"""
DeepEval + DeepSeek 评测示例

使用前提：
  1. 在 E:\test 目录下创建 .env 文件，内容：
     DEEPSEEK_API_KEY=你的DeepSeek API Key
  2. DeepSeek API Key 获取地址：
     https://platform.deepseek.com
"""
import os
import json
from dotenv import load_dotenv

# 加载 .env 文件中的 API Key
load_dotenv()

# 设置通义千问为 DeepEval 的后端模型
# 通义千问兼容 OpenAI SDK，所以设置 OPENAI_BASE_URL 指向它
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric


# ========== 测试用例 ==========
test_cases = [
    LLMTestCase(
        input="什么是 Python 的 GIL？",
        actual_output="GIL（全局解释器锁）是 CPython 中的一种机制，"
                      "它确保同一时间只有一个线程执行 Python 字节码。"
                      "这简化了内存管理，但也限制了多线程的并行执行。",
        expected_output="全局解释器锁，限制多线程并行执行",
    ),
    LLMTestCase(
        input="HTTP 和 HTTPS 的区别是什么？",
        actual_output="HTTPS 是 HTTP 的安全版本，通过 SSL/TLS 协议"
                      "对通信内容进行加密，防止数据被窃取或篡改。",
        expected_output="HTTPS 有加密，HTTP 没有加密",
    ),
    LLMTestCase(
        input="简述机器学习中的过拟合现象",
        actual_output="过拟合是指模型在训练数据上表现很好，"
                      "但在新数据（测试集）上表现差的现象。",
        expected_output="模型在训练集表现好，测试集表现差",
    ),
]


def main():
    # 检查 API Key 是否已设置
    if not os.environ["OPENAI_API_KEY"]:
        print("\n❌ 未检测到 API Key！")
        print("请在 E:\\test 目录下创建 .env 文件，内容如下：")
        print("DASHSCOPE_API_KEY=你的通义千问API Key")
        return

    # 选择评测指标
    # AnswerRelevancyMetric：回答是否与问题相关
    relevancy_metric = AnswerRelevancyMetric(
        model="deepseek-chat",  # 使用 DeepSeek 作为裁判模型
        threshold=0.5,
        include_reason=True,
    )
    # FaithfulnessMetric：回答是否忠实于给定上下文，不产生幻觉
    faithfulness_metric = FaithfulnessMetric(
        model="qwen-plus",
        threshold=0.5,
        include_reason=True,
    )

    print(f"\n{'='*60}")
    print(f"开始评测 {len(test_cases)} 个测试用例...")
    print(f"裁判模型：通义千问 qwen-plus")
    print(f"{'='*60}\n")

    for i, tc in enumerate(test_cases, 1):
        print(f"--- 测试 {i} ---")
        print(f"问题：{tc.input}")
        print(f"回答：{tc.actual_output[:50]}...")

        try:
            # 评测相关性
            relevancy_metric.measure(tc)
            print(f"  相关性得分：{relevancy_metric.score:.2f}")
            if relevancy_metric.reason:
                print(f"  原因：{relevancy_metric.reason}")

            # 评测忠实度（如果提供了 context）
            if hasattr(tc, 'context') and tc.context:
                faithfulness_metric.measure(tc)
                print(f"  忠实度得分：{faithfulness_metric.score:.2f}")

        except Exception as e:
            print(f"  ⚠️ 评测出错：{e}")

        print()

    print(f"{'='*60}")
    print("评测完成！")


if __name__ == "__main__":
    main()
