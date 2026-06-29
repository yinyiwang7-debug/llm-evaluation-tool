"""
回归测试工具
============
每次运行自动保存结果，并与上一次结果对比，发现质量变化。

新增概念：
  1. 时间戳 — 每次运行有一个唯一标签，方便追溯
  2. 历史对比 — 自动找上一次结果做比较
  3. 变化标记 — NEW_FAIL（新失败） / NEW_PASS（修好了） / UNCHANGED（不变）
"""
import os
import json
import datetime
import glob
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ToxicityMetric

# ===== 配置 =====
METRICS = {
    "relevancy": AnswerRelevancyMetric(model="deepseek-chat", threshold=0.5, include_reason=True),
    "faithfulness": FaithfulnessMetric(model="deepseek-chat", threshold=0.5, include_reason=True),
    "toxicity": ToxicityMetric(model="deepseek-chat", threshold=0.5, include_reason=True),
}
CLIENT = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
HISTORY_DIR = "regression_history"  # 历史结果保存目录


def call_model(prompt: str) -> str:
    """调用 DeepSeek 生成回答"""
    resp = CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个知识渊博的AI助手，请简洁准确地回答问题。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def run_all_tests(cases: list) -> list:
    """执行全部测试用例，返回结果列表"""
    results = []
    for i, case in enumerate(cases, 1):
        question = case["input"]
        print(f"  [{i}/{len(cases)}] {case['id']}...", end="")

        # 调用模型
        try:
            answer = call_model(question)
        except Exception as e:
            answer = f"[调用失败] {e}"

        # 评测
        tc = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=case.get("retrieval_context", []),
        )
        check_results = {}
        for check in case.get("check", []):
            metric = METRICS.get(check)
            if not metric:
                continue
            try:
                metric.measure(tc)
                score = round(metric.score, 2)
                check_results[check] = {
                    "score": score,
                    "passed": metric.is_successful() if check != "toxicity" else score < 0.5,
                    "reason": metric.reason[:150],
                }
            except Exception as e:
                check_results[check] = {"score": -1, "passed": False, "reason": str(e)[:100]}

        result = {
            "id": case["id"],
            "category": case["category"],
            "input": question,
            "actual_output": answer,
            "checks": check_results,
            "all_passed": all(c["passed"] for c in check_results.values()),
        }
        results.append(result)
        print(" PASS" if result["all_passed"] else " FAIL")
    return results


def analyze_regression(current: list, previous: list) -> dict:
    """对比本次和上次的结果，找出变化"""
    # 建立索引：id → result
    prev_map = {r["id"]: r for r in previous}

    changes = {"new_passes": [], "new_fails": [], "new_tests": [], "unchanged": []}

    for r in current:
        prev = prev_map.get(r["id"])
        if not prev:
            changes["new_tests"].append(r)  # 新增用例（独立统计）
            continue

        cur_pass = r["all_passed"]
        prev_pass = prev["all_passed"]

        if cur_pass and not prev_pass:
            changes["new_passes"].append(r)   # 之前失败→现在通过（修复了）
        elif not cur_pass and prev_pass:
            changes["new_fails"].append(r)    # 之前通过→现在失败（回退了）
        else:
            changes["unchanged"].append(r)    # 结果没变

    return changes


def save_results(results: list, timestamp: str) -> str:
    """保存结果到历史目录"""
    filepath = f"{HISTORY_DIR}/{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return filepath


def find_previous_run() -> list:
    """查找最近一次历史运行的结果"""
    files = sorted(glob.glob(f"{HISTORY_DIR}/*.json"), reverse=True)
    if not files:
        return []
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def print_report(current: list, changes: dict, timestamp: str):
    """打印回归测试报告"""
    total = len(current)
    passed = sum(1 for r in current if r["all_passed"])
    rate = passed / total * 100

    print(f"\n{'='*55}")
    print(f"  回归测试报告 — {timestamp}")
    print(f"{'='*55}")
    print(f"  通过率：{passed}/{total} = {rate:.1f}%")
    print()

    if changes["new_fails"]:
        print(f"  [回退] 以下用例之前通过，现在失败：")
        for r in changes["new_fails"]:
            fails = [f"{n}({c['score']})" for n, c in r["checks"].items() if not c["passed"]]
            print(f"    [FAIL] {r['id']} -- {', '.join(fails)}")

    if changes["new_passes"]:
        print(f"  [修复] 以下用例之前失败，现在通过：")
        for r in changes["new_passes"]:
            print(f"    [PASS] {r['id']}")

    if changes["new_tests"]:
        print(f"  [新增] 以下为新增加的用例：")
        for r in changes["new_tests"]:
            status = "[PASS]" if r["all_passed"] else "[FAIL]"
            print(f"    {status} {r['id']}")

    unchanged_count = len(changes["unchanged"])
    print(f"\n  未变化：{unchanged_count} 条用例")
    print(f"  历史记录：{HISTORY_DIR}/")
    print(f"{'='*55}")


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("[错误] 未检测到 API Key")
        return

    # 加载用例
    with open("live_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 生成时间戳（格式：2026-06-29_18-30-00）
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"开始回归测试：{len(cases)} 条用例\n")

    # 执行测试
    results = run_all_tests(cases)

    # 找上次结果做对比（先找再保存，避免找到自己）
    previous = find_previous_run()

    # 保存本次结果
    savefile = save_results(results, timestamp)
    print(f"已保存：{savefile}")

    if previous:
        changes = analyze_regression(results, previous)
        print_report(results, changes, timestamp)
    else:
        # 第一次运行，没有历史对比
        passed = sum(1 for r in results if r["all_passed"])
        print(f"\n{'='*55}")
        print(f"  首次运行 — 无历史数据可对比")
        print(f"  通过率：{passed}/{len(results)} = {passed/len(results)*100:.1f}%")
        print(f"  后续运行会自动对比本次结果")
        print(f"{'='*55}")

    # 同时生成 HTML 报告（复用 live_report.html）
    # 但保留时间戳版本
    print(f"\n提示：运行 python run_regression.py 即可重复测试并自动对比")


if __name__ == "__main__":
    main()
