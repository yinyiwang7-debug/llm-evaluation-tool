"""
DeepEval 入门示例：自定义规则评测
不需要 API Key，纯本地运行
"""
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class AnswerLengthMetric(BaseMetric):
    """自定义指标：检查回答长度是否在合理范围内"""

    def __init__(self, min_words: int = 5, max_words: int = 200):
        self.min_words = min_words
        self.max_words = max_words

    def measure(self, test_case: LLMTestCase) -> float:
        """返回 0~1 的分数"""
        word_count = len(test_case.actual_output.split())
        if self.min_words <= word_count <= self.max_words:
            self.score = 1.0
        elif word_count < self.min_words:
            self.score = word_count / self.min_words  # 越短越低分
        else:
            self.score = max(0, 1 - (word_count - self.max_words) / 100)
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    @property
    def __name__(self):
        return "Answer Length"


class KeywordChecker(BaseMetric):
    """自定义指标：检查回答是否包含关键信息"""

    def __init__(self, required_keywords: list[str]):
        self.required_keywords = required_keywords

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output.lower()
        matched = sum(1 for kw in self.required_keywords if kw.lower() in output)
        self.score = matched / len(self.required_keywords)
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    @property
    def __name__(self):
        return "Keyword Check"


# ===== 测试用例 =====
test_cases = [
    LLMTestCase(
        input="Python 的 __init__ 方法有什么用？",
        actual_output="__init__ 是 Python 类的构造函数，在创建对象时自动调用，用于初始化实例属性。",
        expected_output="用于初始化对象属性的构造函数方法",
    ),
    LLMTestCase(
        input="什么是列表推导式？",
        actual_output="好",
        expected_output="一种从可迭代对象创建列表的简洁语法",
    ),
    LLMTestCase(
        input="HTTP 状态码 404 代表什么？",
        actual_output="404 表示服务器无法找到请求的资源，即 Not Found。",
        expected_output="Not Found，资源不存在",
    ),
]

# ===== 运行测试 =====
if __name__ == "__main__":
    length_metric = AnswerLengthMetric(min_words=10, max_words=100)
    keyword_metric = KeywordChecker(required_keywords=["init", "construct"])

    print(f"{'='*60}")
    print(f"{'输入':<30} {'长度分':<10} {'关键词分':<10} {'综合':<10}")
    print(f"{'='*60}")

    for tc in test_cases:
        length_metric.measure(tc)
        keyword_metric.measure(tc)
        combined = (length_metric.score + keyword_metric.score) / 2
        input_short = tc.input[:28] + ".." if len(tc.input) > 28 else tc.input
        print(f"{input_short:<30} {length_metric.score:<10.2f} {keyword_metric.score:<10.2f} {combined:<10.2f}")

    print(f"\n{'='*60}")
    print("测试完成！说明：")
    print(" - 长度分：检查回答是否在合理字数范围内")
    print(" - 关键词分：检查回答是否包含关键信息")
    print(" - 综合 >= 0.6 算合格")
    print(f"{'='*60}")
