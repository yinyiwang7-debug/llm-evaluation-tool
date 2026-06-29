# 大模型自动化评测工具

基于 Python + DeepSeek API + DeepEval 构建的大模型质量评测框架，支持**相关性、忠实度、有害性**三个维度的自动化评测，并具备**回归测试**能力。

## 功能

- ✅ 三维度评测：相关性（是否答非所问）、忠实度（是否产生幻觉）、有害性（是否包含不当内容）
- ✅ 测试用例覆盖三类场景：正常、边界、对抗（含有害诱导和幻觉检测）
- ✅ 实时评测：调用大模型生成回答后自动打分
- ✅ 回归测试：每次运行自动与历史结果对比，发现质量回退
- ✅ HTML 报告：可视化展示通过率和各维度得分

## 项目结构

```
├── evaluation_tool.py      # 评测引擎（预设答案 → 评测 → 报告）
├── live_test.py            # 实时评测（调模型回答 → 评测 → 报告）
├── run_regression.py       # 回归测试（自动对比本次 vs 上次结果）
├── test_cases.json         # 20 条预设答案的测试用例
├── live_cases.json         # 7 条实时测试用例
└── .gitignore              # 排除 API Key 等敏感文件
```

## 快速开始

```bash
# 1. 安装依赖
pip install deepeval openai python-dotenv

# 2. 配置 API Key
# 在项目根目录创建 .env 文件：
# DEEPSEEK_API_KEY=你的DeepSeek API Key

# 3. 运行评测
python evaluation_tool.py    # 预设用例评测
python live_test.py          # 实时调用模型评测
python run_regression.py     # 回归测试
```

## 技术栈

- Python 3.12
- DeepSeek API（LLM-as-Judge）
- DeepEval（评测框架）
- JSON（数据与代码分离）

## 测试结果示例

| 场景 | 用例数 | 通过率 |
|------|--------|--------|
| 正常场景 | 6 | 100% |
| 边界场景 | 6 | 100% |
| 对抗场景（有害） | 2 | 50% |
| 对抗场景（幻觉） | 4 | 25% |
| **总计** | **20** | **80%** |

评测工具成功检测出模型幻觉（答错红楼梦作者、GIL 含义等）和有害回答（教唆作弊）等质量问题。
