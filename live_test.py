"""
大模型实时评测工具
=====================
流程：读取测试问题 → 调用大模型回答 → 自动评测 → 生成报告

这模拟了真实的测试场景：
  你测的是"模型对这个问题的真实回答"，而不是预设好的答案
"""
import os
import json
import datetime
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


def evaluate_case(input_text: str, actual_output: str, case: dict) -> dict:
    """对模型回答进行多维度评测"""
    result = {
        "id": case["id"],
        "category": case["category"],
        "input": input_text,
        "actual_output": actual_output,
        "checks": {},
    }

    tc = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=case.get("retrieval_context", []),
    )

    for check in case.get("check", []):
        metric = METRICS.get(check)
        if not metric:
            continue
        try:
            metric.measure(tc)
            score = round(metric.score, 2)
            result["checks"][check] = {
                "score": score,
                "passed": metric.is_successful() if check != "toxicity" else score < 0.5,
                "reason": metric.reason,
            }
        except Exception as e:
            result["checks"][check] = {
                "score": -1,
                "passed": False,
                "reason": f"出错：{str(e)[:150]}",
            }

    return result


def generate_html(results: list) -> str:
    """生成 HTML 报告"""
    total = len(results)
    passed = sum(1 for r in results if all(c["passed"] for c in r["checks"].values()))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>大模型实时评测报告</title>
<style>
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
.summary {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
.pass-rate {{ font-size: 36px; font-weight: bold; color: #4CAF50; }}
.pass-rate.low {{ color: #f44336; }}
.dim-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 15px 0; }}
.dim-card {{ padding: 15px; border-radius: 6px; text-align: center; }}
.dim-card.good {{ background: #e8f5e9; }}
.dim-card.warn {{ background: #fff3e0; }}
.dim-card.bad {{ background: #ffebee; }}
.dim-card h3 {{ margin: 0 0 5px; font-size: 14px; color: #666; }}
.dim-card .score {{ font-size: 28px; font-weight: bold; }}
.model-response {{ background: #f9f9f9; padding: 12px; border-radius: 6px; border-left: 3px solid #1976d2; margin: 8px 0; font-size: 14px; line-height: 1.6; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #fafafa; font-size: 13px; color: #666; }}
.passed {{ color: #4CAF50; font-weight: bold; }}
.failed {{ color: #f44336; font-weight: bold; }}
.metric-badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; margin: 1px; }}
.metric-pass {{ background: #e8f5e9; color: #2e7d32; }}
.metric-fail {{ background: #ffebee; color: #c62828; }}
.footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 20px; }}
</style>
</head>
<body>
<h1>大模型实时评测报告</h1>
<p>生成时间：{now} | 调用模型：DeepSeek deepseek-chat | 测试用例：{total} 条</p>
<div class="summary">
    <h2>总览</h2>
    <div class="pass-rate {'low' if passed/total < 0.6 else ''}">{passed} / {total}</div>
    <p>通过率：{passed}/{total} = {passed/total*100:.1f}%</p>
    <h3>各维度平均分</h3>
    <div class="dim-grid">
"""
    dims = {}
    for r in results:
        for name, info in r["checks"].items():
            if name not in dims:
                dims[name] = []
            if info["score"] >= 0:
                dims[name].append(info["score"])
    for name, scores in dims.items():
        avg = sum(scores) / len(scores)
        css = "good" if avg >= 0.8 else "warn" if avg >= 0.5 else "bad"
        html += f'<div class="dim-card {css}"><h3>{name}</h3><div class="score">{avg:.2f}</div></div>\n'

    html += """</div></div>
<h2>逐条结果</h2>
<table>
<tr><th>ID</th><th>类别</th><th>问题</th><th>评测结果</th></tr>
"""
    for r in results:
        all_pass = all(c["passed"] for c in r["checks"].values())
        status = "通过" if all_pass else "未通过"
        input_short = r["input"][:20] + "..." if len(r["input"]) > 20 else r["input"]
        checks_html = ""
        for name, info in r["checks"].items():
            cls = "metric-pass" if info["passed"] else "metric-fail"
            checks_html += f'<span class="metric-badge {cls}">{name}: {info["score"]}</span> '

        html += f"""<tr>
<td>{r["id"]}</td>
<td>{r["category"]}</td>
<td>{input_short}</td>
<td class="{'passed' if all_pass else 'failed'}">{status} {checks_html}</td>
</tr>
<tr style="background:#fafafa;"><td colspan="4"><div class="model-response"><strong>模型回答：</strong>{r["actual_output"]}</div>"""
        for name, info in r["checks"].items():
            if info.get("reason"):
                reason = info["reason"][:300]
                html += f'<div style="font-size:12px;color:#666;margin:4px 0;">{name}: {reason}</div>'
        html += "</td></tr>"

    html += f"""</table>
<div class="footer"><p>Generated by DeepEval + DeepSeek</p></div>
</body></html>"""
    return html


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("[错误] 未检测到 API Key！")
        return

    with open("live_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"[开始] 共 {len(cases)} 个测试用例，将依次调用 DeepSeek 生成回答...\n")
    results = []

    for i, case in enumerate(cases, 1):
        question = case["input"]
        print(f"[{i}/{len(cases)}] 问题：{question[:30]}...")

        try:
            answer = call_model(question)
            print(f"  回答：{answer[:60]}...")
        except Exception as e:
            print(f"  模型调用失败：{e}")
            answer = f"[模型调用失败] {str(e)[:80]}"

        result = evaluate_case(question, answer, case)
        results.append(result)

        for name, info in result["checks"].items():
            icon = "[PASS]" if info["passed"] else "[FAIL]"
            print(f"  {icon} {name}: {info['score']}")
        print()

    html = generate_html(results)
    with open("live_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    passed = sum(1 for r in results if all(c["passed"] for c in r["checks"].values()))
    print(f"{'='*50}")
    print(f"[完成] 评测完成！")
    print(f"通过率：{passed}/{len(results)} = {passed/len(results)*100:.1f}%")
    print(f"报告已生成：live_report.html")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
