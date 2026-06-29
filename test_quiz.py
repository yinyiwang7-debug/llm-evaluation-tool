import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

  # 评测指标：忠实度（检查回答是否基于事实，有没有编造）
metric = FaithfulnessMetric(
    model="deepseek-chat",
    threshold=0.5,
    include_reason=True,
  )

  # ===== 请你补全测试用例 =====
  # 要求：
  # - test_case_1：回答与上下文一致（应该是高分）
  # - test_case_2：回答编造了上下文没有的内容（应该是低分）

test_case_1 = LLMTestCase(
    input="python的GIL是什么",          # 问题
    actual_output="GIL 是 CPython 的全局解释器锁",  # 模型回答
    retrieval_context=["GIL（全局解释器锁）是 CPython 中的一种机制，确保同一时间只有一个线程执行 Python 字节码"],      # 已知事实（列表）
  )

test_case_2 = LLMTestCase(
    input="python的GIL是什么",
    actual_output="GIL 是 Google开发的语言",
    retrieval_context=["GIL（全局解释器锁）是 CPython 中的一种机制，确保同一时间只有一个线程执行 Python 字节码"],
  )

  # ===== 运行 =====
if __name__ == "__main__":
    for i, tc in enumerate([test_case_1, test_case_2], 1):
        try:
            metric.measure(tc)
            print(f"测试 {i}：得分 {metric.score:.2f}")
            print(f"原因：{metric.reason}\n")
        except Exception as e:
            print(f"测试 {i} 出错：{e}")