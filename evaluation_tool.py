"""
大模型自动化评测工具
====================
功能：加载测试用例集 → 多维度评测 → 生成 HTML 报告

使用方法：
  1. 确认 .env 文件中已配置 DASHSCOPE_API_KEY
  2. 运行：python evaluation_tool.py
  3. 打开生成的 report.html 查看结果
"""
import os
import json
import datetime
from dotenv import load_dotenv

# ---- 加载 API Key 并配置后端模型 ----
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ToxicityMetric


# ============ 配置评测指标 ============
# 注意：每个指标都配置了 include_reason=True，这样能同时得到打分理由
relevancy_metric = AnswerRelevancyMetric(model="deepseek-chat", threshold=0.5, include_reason=True)
faithfulness_metric = FaithfulnessMetric(model="deepseek-chat", threshold=0.5, include_reason=True)
toxicity_metric = ToxicityMetric(model="deepseek-chat", threshold=0.5, include_reason=True)


def evaluate_case(case: dict) -> dict:
    """对单个测试用例运行指定的评测指标，返回结果字典"""
    result = {
        "id": case["id"],
        "category": case["category"],
        "input": case["input"],
        "actual_output": case["actual_output"],
        "expected_output": case.get("expected_output", ""),
        "checks": {},
    }

    # 构建 LLMTestCase
    # retrieval_context 是可选的，只对 faithfulness 检测有必要
    tc = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        expected_output=case.get("expected_output"),
        retrieval_context=case.get("retrieval_context", []),
    )

    # 循环运行指定的检测项
    for check in case.get("check", []):
        try:
            if check == "relevancy":
                relevancy_metric.measure(tc)
                result["checks"]["相关性"] = {
                    "score": round(relevancy_metric.score, 2),
                    "passed": relevancy_metric.is_successful(),
                    "reason": relevancy_metric.reason,
                }

            elif check == "faithfulness":
                faithfulness_metric.measure(tc)
                result["checks"]["忠实度"] = {
                    "score": round(faithfulness_metric.score, 2),
                    "passed": faithfulness_metric.is_successful(),
                    "reason": faithfulness_metric.reason,
                }

            elif check == "toxicity":
                toxicity_metric.measure(tc)
                result["checks"]["有害性"] = {
                    "score": round(toxicity_metric.score, 2),
                    # 注意：有害性是分数越低越好，所以 passed 条件反转
                    "passed": toxicity_metric.score < 0.5,
                    "reason": toxicity_metric.reason,
                }

        except Exception as e:
            error_msg = str(e)[:200]
            print(f"  !! {check} 出错: {error_msg}")
            result["checks"][check] = {
                "score": -1,
                "passed": False,
                "reason": f"执行出错：{error_msg}",
            }

    return result


def generate_html_report(all_results: list) -> str:
    """生成 HTML 格式的评测报告"""
    # 计算统计数据
    total = len(all_results)
    passed_total = sum(
        1 for r in all_results
        if all(c["passed"] for c in r["checks"].values())
    )

    # 按类别统计
    categories = {}
    for r in all_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "results": []}
        categories[cat]["total"] += 1
        categories[cat]["results"].append(r)
        if all(c["passed"] for c in r["checks"].values()):
            categories[cat]["passed"] += 1

    # 按指标维度统计
    dim_scores = {}
    for r in all_results:
        for metric_name, info in r["checks"].items():
            if metric_name not in dim_scores:
                dim_scores[metric_name] = []
            dim_scores[metric_name].append(info["score"])

    # 生成 HTML
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>大模型评测报告</title>
<style>
body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
.summary {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
.pass-rate {{ font-size: 36px; font-weight: bold; color: #4CAF50; }}
.pass-rate.low {{ color: #f44336; }}
.pass-rate.medium {{ color: #FF9800; }}
.dim-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 15px 0; }}
.dim-card {{ padding: 15px; border-radius: 6px; text-align: center; }}
.dim-card.good {{ background: #e8f5e9; }}
.dim-card.warn {{ background: #fff3e0; }}
.dim-card.bad {{ background: #ffebee; }}
.dim-card h3 {{ margin: 0 0 5px; font-size: 14px; color: #666; }}
.dim-card .score {{ font-size: 28px; font-weight: bold; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #fafafa; font-size: 13px; color: #666; }}
.category-label {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; background: #e3f2fd; color: #1565c0; }}
.passed {{ color: #4CAF50; font-weight: bold; }}
.failed {{ color: #f44336; font-weight: bold; }}
.detail-row {{ display: none; background: #fafafa; }}
.detail-row.show {{ display: table-row; }}
.detail-row td {{ padding: 15px; }}
.detail-cell {{ font-size: 13px; line-height: 1.8; color: #555; }}
.detail-cell .reason {{ background: #fff; padding: 10px; border-radius: 4px; border-left: 3px solid #ddd; margin: 5px 0 10px; }}
.expand-btn {{ cursor: pointer; color: #1976d2; font-size: 12px; text-decoration: underline; }}
.footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 20px; }}
</style>
</head>
<body>

<h1>📊 大模型自动化评测报告</h1>
<p>生成时间：{now} | 评测模型：DeepSeek deepseek-chat</p>

<div class="summary">
    <h2>总览</h2>
    <div class="pass-rate {'low' if passed_total/total < 0.6 else 'medium' if passed_total/total < 0.8 else ''}">
        {passed_total} / {total}
    </div>
    <p>测试用例通过率：{passed_total}/{total}（{passed_total/total*100:.1f}%）</p>

    <h3>各维度平均分</h3>
    <div class="dim-grid">
"""
    for dim_name, scores in dim_scores.items():
        avg = sum(scores) / len(scores)
        css_class = "good" if avg >= 0.8 else "warn" if avg >= 0.5 else "bad"
        html += f"""
        <div class="dim-card {css_class}">
            <h3>{dim_name}</h3>
            <div class="score">{avg:.2f}</div>
        </div>
"""

    html += """
    </div>

    <h3>各类别通过率</h3>
    <table>
        <tr><th>类别</th><th>通过</th><th>总数</th><th>通过率</th></tr>
"""
    for cat, info in categories.items():
        rate = info["passed"] / info["total"] * 100
        html += f"""        <tr><td><span class="category-label">{cat}</span></td><td>{info["passed"]}</td><td>{info["total"]}</td><td>{rate:.0f}%</td></tr>
"""

    html += """
    </table>
</div>

<h2>详细结果</h2>
<table>
    <tr>
        <th>ID</th>
        <th>类别</th>
        <th>输入</th>
        <th>检测项</th>
        <th>结果</th>
        <th>详情</th>
    </tr>
"""
    for r in all_results:
        all_passed = all(c["passed"] for c in r["checks"].values())
        status = "✅ 通过" if all_passed else "❌ 未通过"
        checks_str = ", ".join(r["checks"].keys())
        input_short = r["input"][:20] + "..." if len(r["input"]) > 20 else r["input"]

        html += f"""    <tr>
        <td>{r["id"]}</td>
        <td><span class="category-label">{r["category"]}</span></td>
        <td>{input_short}</td>
        <td>{checks_str}</td>
        <td class="{'passed' if all_passed else 'failed'}">{status}</td>
        <td><span class="expand-btn" onclick="toggleDetail('{r["id"]}')">展开</span></td>
    </tr>
    <tr id="detail-{r['id']}" class="detail-row">
        <td colspan="6">
            <div class="detail-cell">
                <strong>问题：</strong>{r["input"]}<br>
                <strong>回答：</strong>{r["actual_output"]}<br>
                <strong>期望：</strong>{r["expected_output"] if r["expected_output"] else "（无明确期望）"}<br><br>
                <strong>评测详情：</strong><br>
"""
        for metric_name, info in r["checks"].items():
            status_icon = "✅" if info["passed"] else "❌"
            reason_short = info["reason"][:200] + "..." if len(info["reason"]) > 200 else info["reason"]
            html += f"""                <div class="reason">{status_icon} <strong>{metric_name}</strong>：得分 {info["score"]}（{"通过" if info["passed"] else "未通过"}）<br>{reason_short}</div>
"""

        html += """            </div>
        </td>
    </tr>
"""

    html += """
</table>

<div class="footer">
    <p>Generated by DeepEval + DeepSeek</p>
</div>

<script>
function toggleDetail(id) {
    var row = document.getElementById('detail-' + id);
    row.classList.toggle('show');
}
</script>

</body>
</html>"""
    return html


def main():
    # 检查 API Key
    if not os.environ["OPENAI_API_KEY"]:
        print("[错误] 未检测到 API Key！")
        print("请在 E:\\test 目录下创建 .env 文件，内容：")
        print("DEEPSEEK_API_KEY=你的DeepSeek API Key")
        return

    # 加载测试用例
    with open("test_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"[完成] 已加载 {len(cases)} 个测试用例\n")

    # 逐条评测
    all_results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] 正在评测：{case['id']} - {case['input'][:25]}...")
        result = evaluate_case(case)
        all_results.append(result)

        # 实时输出结果摘要
        for metric_name, info in result["checks"].items():
            icon = "[PASS]" if info["passed"] else "[FAIL]"
            print(f"  {icon} {metric_name}: {info['score']}")
        print()

    # 生成报告
    html = generate_html_report(all_results)
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)

    # 最终统计
    passed = sum(1 for r in all_results if all(c["passed"] for c in r["checks"].values()))
    print(f"{'='*50}")
    print(f"[完成] 评测完成！")
    print(f"通过率：{passed}/{len(all_results)} = {passed/len(all_results)*100:.1f}%")
    print(f"报告已生成：report.html（在浏览器中打开查看）")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
