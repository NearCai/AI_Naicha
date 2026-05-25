# beverage_ai — 茶饮研发闭环 AI 系统 v1

[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![Status](https://img.shields.io/badge/status-v1%20skeleton-orange)]()

按照 [技术方案书 v1.4](docs/茶饮研发闭环AI系统_技术方案书.md) 和 [v1 实现方案](docs/茶饮研发闭环AI系统_v1实现方案.md) 实现的代码骨架。

## v1 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 词表 (附录 D) | ✅ 完整(50 条 demo subset) | 全 schema 校验通过,生产需扩到 207 |
| Recipe schema | ✅ 完整 | pydantic v2 强校验 |
| 健康计算器 | ✅ 完整 | 纯查表 + cooking factor + 致敏原 |
| 约束检查器 | ✅ 完整 | 6 类硬约束 + 致敏原排除 |
| 9 步生成器 + 守恒重标定 | ✅ 完整 | Dirichlet 采样 + 风格/糖度档位映射 |
| 上下文条件 Dirichlet + Bayesian 更新 | ✅ 完整 | 共轭后验 + snapshot 落盘 |
| MMR + LCB/UCB acquisition | ✅ 完整 | 优化器多样性 + 不确定性感知 |
| NSGA-II 优化器 (MixedVariableGA) | ✅ 完整 | pymoo 集成 |
| 感官 GNN | 🟡 架构完成 + Mock 预测器 | GAT 模型类定义完整,真实训练需 GPU + 数据 |
| 销量 LightGBM (品牌 FE + K 折) | 🟡 架构完成 + Mock 预测器 | 同上 |
| 复购 v1 (加权公式) | ✅ 完整 | |
| LLM Planner | ✅ 完整(Claude + Mock) | 设了 API key 就用真实, 否则 Mock |
| 反馈记录 (DuckDB) | ✅ 完整 | |
| 端到端 pipeline | ✅ 完整 | 可跑通 |
| CLI | ✅ 完整 | `beverage-ai generate` / `health` / `demo` |
| Gradio demo | ✅ 完整 | `python demo/app.py` |

**真实训练所需**:50K+ 评论 (路径 A)、200+ 盲品 (路径 C)、1000+ SKU 销量数据。本骨架交付**架构 + 评估接口 + mock 预测器**,任何模块都能独立用真实模型替换。

## 安装

```bash
# 推荐 uv (https://docs.astral.sh/uv/)
uv venv && uv pip install -e ".[dev]"

# 或纯 pip
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -e ".[dev]"
```

可选 extras:`.[ml]` 装 torch/lightgbm,`.[llm]` 装 anthropic,`.[demo]` 装 gradio,`.[all]` 全装。

## 快速上手

### 1. 跑测试(无需任何 API key)

```bash
pytest
```

### 2. CLI 演示

```bash
# 健康计算 + 约束检查
beverage-ai health examples/recipe_example.yaml

# 9 步生成器产出候选
beverage-ai generate --request "夏季年轻女性低糖" --n 10

# 端到端 pipeline(用 mock 预测器)
beverage-ai pipeline --request "夏季年轻女性低糖" --top-k 5
```

### 3. Gradio 网页 demo

```bash
pip install -e ".[demo]"
python demo/app.py
# 打开 http://localhost:7860
```

## 项目结构

```
beverage_ai/                # python 包
├── ingredients/            # 词表 (§附录 D)
├── recipes/                # Recipe schema + 9 步生成器 + 守恒
├── constraints/            # §3.4 约束检查器
├── simulators/             # 4 个仿真器
│   ├── sensory/            # §3.3.1 GNN
│   ├── sales/              # §3.3.2 LightGBM
│   ├── repurchase/         # §3.3.3 加权
│   └── health/             # §3.3.4 查表
├── optimizer/              # §3.5 NSGA-II + LCB + MMR
├── planner/                # §3.1 LLM Planner
├── priors/                 # §E.9 条件 Dirichlet + Bayesian
├── feedback/               # §3.7 DuckDB
└── pipeline/               # 端到端

data/                       # 数据资产 (YAML/parquet)
tests/                      # pytest
demo/                       # Gradio app
scripts/                    # 一次性脚本(训练/数据采集)
```

## 与文档的对应

详见 v1 实现方案 §1 实现地图。每个文件头注释标了对应技术方案章节。

## v1 不在范围

参考 [技术方案书 §10 + 实现方案 §10](docs/茶饮研发闭环AI系统_v1实现方案.md):
- Conditional VAE 生成器 → v2
- DeepSurv 复购模型 → v2 (需真实 POS)
- Double ML 销量预测 → v2 (需 ≥3000 SKU)

## License

MIT
