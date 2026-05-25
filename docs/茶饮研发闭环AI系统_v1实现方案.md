# 茶饮研发闭环 AI 系统 — v1 实现方案

> **版本**:v1.0(实现方案首版)
> **日期**:2026-05-25
> **对应技术方案**:[茶饮研发闭环AI系统_技术方案书.md](茶饮研发闭环AI系统_技术方案书.md) v1.4
> **目标读者**:开发团队 / 实习生 / 学生
> **预计周期**:8 周(单人)/ 5–6 周(2 人)
> **预算上限**:¥2400(LLM API + 30 SKU + 物料)

---

## 0. 文档说明

本文档把技术方案书 v1.4 的设计、方法、词表、参数表**落地为可执行的工程实施方案**。包括:

- 项目结构与命名约定
- 各模块的实现骨架(关键类、函数签名、可直接复用的代码片段)
- 8 周排期细化到周(每周明确 deliverable + 验收标准)
- 测试策略与 CI
- 数据/代码资产清单

**章节引用约定**:本文档每处 "§X.Y" 默认指向 [技术方案书 v1.4](茶饮研发闭环AI系统_技术方案书.md) 的对应章节;本文档内引用用 "本文 §X.Y"。

**与技术方案书的分工**:
- 技术方案书 = WHAT(系统是什么 / 各模块的方法 / 风险与评估)
- 本文档     = HOW(怎么写代码 / 怎么排周 / 怎么验收)

---

## 1. 与技术方案书的对应关系(实现地图)

| 技术方案章节 | 实现位置 | 上线周 |
|---|---|---|
| §3.1 LLM Planner | `planner/llm_planner.py` | Week 3 |
| §3.2 配方生成器 + §E.2 9 步 | `recipes/generator.py` + `recipes/reconciler.py` | Week 3 |
| §3.3.1 感官 GNN(Stage 1) | `simulators/sensory/{model,train,data}.py` | Week 4 |
| §3.3.1 感官 GNN(Stage 2 微调) | 同上 | Week 7 |
| §3.3.2 销量预测器(品牌固定效应 + K 折) | `simulators/sales/{model,train}.py` | Week 4 |
| §3.3.3 复购 v1(加权) | `simulators/repurchase/v1_weighted.py` | Week 4 |
| §3.3.4 健康计算器 + §D.4 nutrition | `simulators/health/calculator.py` | Week 1 |
| §3.4 约束检查器 + §E.3 物理约束 | `constraints/checker.py` | Week 1 |
| §3.5 优化器(MixedVariableGA + LCB + MMR) | `optimizer/{nsga2,acquisition,mmr}.py` | Week 5 |
| §3.6 人类盲品(BIBD + Latin square) | `scripts/panel_*.py` + DuckDB schema | Week 7 |
| §3.7 反馈管道 | `feedback/{recorder,updater}.py` | Week 7-8 |
| §6 评估(Wilcoxon / Friedman / 混合效应) | `scripts/analyze_panel.py` | Week 8 |
| 附录 D 原料词表 | `data/ingredients/ingredient_vocab.yaml` + `ingredients/vocab.py` | Week 1 |
| 附录 E.1–E.8 用量与比例 | `recipes/generator.py` + `data/priors/` | Week 2-3 |
| 附录 E.9 灵活化机制(条件 α / Bayesian 更新) | `priors/{engine,dirichlet,typical_serving}.py` | Week 3 + Week 7-8 |

---

## 2. v1 范围

### 2.1 必须交付(8 周内)

- [x] 207 条原料词表(YAML 落地)+ enrichment schema 校验
- [x] Recipe schema + 9 步生成器 + 守恒重标定
- [x] 健康计算器(纯查表)
- [x] 约束检查器(8 条硬约束 + 致敏原 + 排除列表)
- [x] LLM Planner(Claude Haiku + 严格 JSON)
- [x] 感官 GNN(GAT,Stage 1 + Stage 2 双层输出头)
- [x] 销量预测器(LightGBM + 品牌固定效应 + K 折交叉拟合)
- [x] 复购 v1(加权公式)
- [x] NSGA-II + LCB + MMR + MixedVariableGA
- [x] 上下文条件 Dirichlet + Bayesian 后验更新
- [x] 端到端 pipeline + Gradio demo
- [x] 盲品试验(35+ 评委,Latin square BIBD)+ Wilcoxon / Friedman / 混合效应分析
- [x] DuckDB 反馈记录 + 增量更新触发

### 2.2 不在 v1(留 v2)

- Conditional VAE 生成器(需 ≥ 5000 样本)
- DeepSurv 复购模型(需真实 POS 数据)
- Double ML 销量预测(数据量到 3000 SKU 后)
- 个性化推荐(系统对象是 SKU,不是用户)
- A/B 测试自动化与门店部署
- 移动端 / 多语言

---

## 3. 技术栈与环境

### 3.1 Python 环境(用 `uv` 管理)

```toml
# pyproject.toml(完整版见本文附录 A)
[project]
name = "beverage_ai"
version = "0.1.0"
requires-python = ">=3.11,<3.13"

dependencies = [
    # 数据 & schema
    "pydantic>=2.5",
    "pyyaml>=6.0",
    "pandas>=2.2",
    "numpy>=1.26,<2.0",      # torch_geometric 兼容性
    "pyarrow>=14",
    "duckdb>=0.9",

    # ML 主干
    "torch>=2.1",
    "torch-geometric>=2.4",
    "lightgbm>=4.1",
    "scikit-learn>=1.4",

    # 优化
    "pymoo>=0.6.1",
    "botorch>=0.10",          # 可选,用于 Pareto 局部细化

    # 统计
    "scipy>=1.11",
    "statsmodels>=0.14",

    # LLM
    "anthropic>=0.8",

    # Demo & 追踪
    "gradio>=4.0",
    "wandb>=0.16",

    # 工程
    "typer>=0.9",
    "loguru>=0.7",
]
```

### 3.2 GPU 需求与时长

| 任务 | GPU 要求 | 单次耗时 |
|---|---|---|
| GNN Stage 1 预训练(50K 配方-评分对) | RTX 3060+ 12GB | 4–8 h |
| GNN Stage 2 微调(200–300 盲品样本) | 同上 | < 30 min |
| Sales LightGBM | CPU 即可 | < 10 min |
| 推理(单次端到端 pipeline) | CPU 即可 | < 30 s |

合计每周 GPU < 10 h,可用学校实验室 / 个人卡 / Colab Pro。

### 3.3 第三方服务与成本

| 服务 | 用途 | 8 周预算 |
|---|---|---|
| Anthropic Claude API (Haiku) | LLM Planner + 路径 A aspect 抽取 | ~$100 (~¥720) |
| OpenAI GPT-4o-mini | 对比 / 鲁棒性测试(可选) | ~$20 (~¥150) |
| Weights & Biases | 实验追踪(免费) | ¥0 |
| GitHub Actions | CI(公共 repo 免费) | ¥0 |
| 头部 30 SKU 购买 + 折光仪 | §E.4-A 逆向工程 | ~¥950 |
| 盲品物料 + 评委补贴 | Week 7 | ~¥500 |
| 食品研发书籍(2–3 本) | §E.4-B 配方收集 | ~¥150 |
| **合计** | — | **~¥2470** |

---

## 4. 项目结构

```
beverage_ai/
├── pyproject.toml
├── README.md
├── .python-version           # 3.11
├── .env.example              # ANTHROPIC_API_KEY=...
├── .gitignore
├── .github/workflows/ci.yml
│
├── data/                     # 数据资产(只读,git LFS 大文件)
│   ├── ingredients/
│   │   ├── ingredient_vocab.yaml          # §附录 D 207 条
│   │   ├── aliases.yaml                   # 同义词归一化
│   │   ├── topping_compatibility.yaml     # §E.5.3
│   │   └── nutrition_sources/             # 营养数据来源原文
│   ├── recipes/
│   │   ├── reference_recipes_v1.yaml      # §E.4-B 100+ 公开复刻
│   │   └── reverse_engineered_v1.yaml     # §E.4-A 30 SKU 称重
│   ├── reviews/                           # 路径 A
│   │   └── reviews_v1.parquet
│   ├── panel/                             # 路径 C 盲品
│   │   └── tasting_v1.parquet
│   ├── products/                          # 销量端
│   │   └── sku_features_v1.parquet
│   └── priors/
│       ├── dirichlet_alpha_v1.yaml        # §E.5.1 初始 α
│       └── prior_history/                 # 每次更新的 snapshot
│
├── beverage_ai/                           # Python 包源码
│   ├── __init__.py
│   ├── ingredients/                       # §附录 D
│   │   ├── vocab.py                       # 加载 + 查询 + 校验
│   │   ├── validator.py                   # schema 校验
│   │   └── aliases.py                     # 别名归一化
│   ├── recipes/                           # §3.2 + §E.2
│   │   ├── schema.py                      # Recipe pydantic 模型
│   │   ├── generator.py                   # 9 步生成
│   │   └── reconciler.py                  # Step 10 守恒重标定
│   ├── constraints/                       # §3.4 + §E.3
│   │   └── checker.py
│   ├── simulators/
│   │   ├── sensory/                       # §3.3.1
│   │   │   ├── model.py                   # GAT 模型 + 双层输出头
│   │   │   ├── data.py                    # 路径 A/C 数据加载
│   │   │   ├── train.py                   # Stage 1 + Stage 2
│   │   │   └── predict.py
│   │   ├── sales/                         # §3.3.2
│   │   │   ├── model.py                   # LightGBM + K 折
│   │   │   ├── train.py
│   │   │   └── predict.py
│   │   ├── repurchase/
│   │   │   └── v1_weighted.py             # §3.3.3 加权公式
│   │   └── health/
│   │       └── calculator.py              # §3.3.4 纯查表
│   ├── optimizer/                         # §3.5
│   │   ├── problem.py                     # pymoo Problem 定义
│   │   ├── nsga2.py                       # MixedVariableGA wrapper
│   │   ├── acquisition.py                 # LCB / UCB
│   │   └── mmr.py                         # Top-K 多样性
│   ├── planner/                           # §3.1
│   │   └── llm_planner.py                 # Claude API + JSON schema
│   ├── priors/                            # §E.5 + §E.9
│   │   ├── engine.py                      # PriorEngine 统一接口
│   │   ├── dirichlet.py                   # 条件 α + Bayesian 更新
│   │   └── typical_serving.py             # 在线校准
│   ├── feedback/                          # §3.7
│   │   ├── recorder.py                    # DuckDB 写入
│   │   └── updater.py                     # 触发各模块更新
│   ├── pipeline/                          # 端到端编排
│   │   ├── end_to_end.py
│   │   └── config.py
│   └── utils/
│       ├── io.py
│       └── logging.py
│
├── scripts/                               # 一次性脚本
│   ├── ingest_reviews.py                  # 爬虫 + LLM 抽 aspect
│   ├── ingest_reference_recipes.py
│   ├── reverse_engineer_skus.py
│   ├── train_sensory_gnn_stage1.py
│   ├── train_sensory_gnn_stage2.py
│   ├── train_sales_model.py
│   ├── fit_dirichlet_priors.py
│   ├── run_pipeline_demo.py
│   ├── panel_design.py                    # 生成 BIBD + Latin square
│   ├── analyze_panel.py                   # Wilcoxon + Friedman + 混合效应
│   └── update_from_feedback.py
│
├── tests/
│   ├── unit/
│   │   ├── test_vocab.py
│   │   ├── test_recipe_schema.py
│   │   ├── test_generator.py
│   │   ├── test_reconciler.py
│   │   ├── test_constraints.py
│   │   ├── test_health.py
│   │   ├── test_acquisition.py
│   │   ├── test_priors.py
│   │   └── test_mmr.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_feedback_loop.py
│   │   └── test_optimizer.py
│   └── fixtures/
│       ├── sample_vocab.yaml
│       └── sample_recipes.yaml
│
├── notebooks/
│   ├── 01_explore_reviews.ipynb
│   ├── 02_eda_reference_recipes.ipynb
│   ├── 03_gnn_training_log.ipynb
│   └── 04_panel_results_analysis.ipynb
│
└── demo/
    ├── app.py                             # Gradio
    └── examples/
        ├── input_summer_youth.json
        ├── input_winter_mature.json
        └── input_lowsugar_healthy.json
```

---

## 5. 数据资产清单

| 文件 | Schema | 来源 | 责任人 | 截止 | 量级 |
|---|---|---|---|---|---|
| `ingredient_vocab.yaml` | §D.4 | 人工整理 + USDA + 中国食物成分表 | 数据 | Week 1 | 207 条 |
| `aliases.yaml` | `{alias: canonical_id}` | 人工(品牌产品页爬取) | 数据 | Week 1 | ~100 |
| `topping_compatibility.yaml` | `{(t1, t2): score in [-1, 1]}` | 人工 + 后期挖掘 | 数据 | Week 3 | ~50 对 |
| `reference_recipes_v1.yaml` | `[Recipe + score?]` | 小红书复刻 + 培训教材 | 数据 | Week 3 | ≥ 100 |
| `reverse_engineered_v1.yaml` | 同上 + 测量元信息 | 头部 30 SKU 实物称重 | 数据 | Week 2 | 30 |
| `reviews_v1.parquet` | `(sku, customization, text, ts)` | 大众点评 / 小红书爬虫 | 数据 | Week 2(并行启动) | 5–10 万 |
| `sku_features_v1.parquet` | 销量代理特征 | 品牌产品页 | 数据 | Week 2 | ≥ 1000 |
| `tasting_v1.parquet` | `(panelist, recipe, dim, score)` | 盲品 | 数据 + 评委 | Week 7 | ~1200 行 |
| `dirichlet_alpha_v1.yaml` | §E.5.1 | 从 `reference_recipes_v1` 拟合 | 模型 | Week 3 | 6 风格 |

**约定**:所有数据文件版本化(`_v1` 后缀);更新时建 `_v2` 文件 + 在 `data/CHANGELOG.md` 记录差异。

---

## 6. 模块实现细节

下面每节给:**职责一句话 + 关键类签名 + 必看代码片段 + 测试要点**。完整代码骨架见本文附录 C。

### 6.1 `ingredients/vocab.py`

**职责**:加载 `ingredient_vocab.yaml`,提供 `get(id)`、`search(name)`、`validate()` 三类接口。

```python
from pydantic import BaseModel, Field
from typing import Literal
import yaml

Category = Literal[
    "tea_base", "dairy_base", "alt_milk_base", "coffee_base",
    "sweetener", "fruit", "topping", "flavoring",
    "auxiliary", "gel", "grain",
]

class IngredientNutrition(BaseModel):
    energy_kcal: float | None = None
    sugar_g: float | None = None
    fat_g: float | None = None
    trans_fat_g: float = 0.0
    caffeine_mg: float = 0.0
    sodium_mg: float | None = None

class Ingredient(BaseModel):
    id: str = Field(pattern=r"^[a-z]+_[a-z0-9_]+$")
    name_zh: str
    name_en: str
    category: Category
    subcategory: str | None = None
    default_form: str
    typical_serving_g: float = Field(gt=0)
    allergens: list[str] = []
    cost_tier: Literal["low", "medium", "high", "premium"]
    supply: Literal["stable", "seasonal", "volatile"]
    shelf_life_days: int | None = None
    nutrition_per_100g: IngredientNutrition
    flavor_descriptors: list[str] = []
    notes_zh: str | None = None
    source: str
    deprecated: bool = False

class Vocab:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._items: dict[str, Ingredient] = {
            r["id"]: Ingredient(**r) for r in raw
        }

    def get(self, id_: str) -> Ingredient:
        if id_ not in self._items:
            raise KeyError(f"Ingredient not in vocab: {id_}")
        return self._items[id_]

    def by_category(self, cat: Category) -> list[Ingredient]:
        return [i for i in self._items.values() if i.category == cat and not i.deprecated]

    def __contains__(self, id_: str) -> bool:
        return id_ in self._items
```

**测试要点**:词表 YAML 加载零错误;所有 207 条 schema 校验通过;缺失 `nutrition_per_100g` 必填字段触发警告。

---

### 6.2 `recipes/schema.py`

**职责**:统一 `Recipe` 数据契约,所有模块的输入输出。

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal

SugarLevel = Literal["无糖", "三分", "五分", "七分", "全糖"]
Style = Literal["纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调"]
CupSize = Literal[380, 500, 700]

class Process(BaseModel):
    extraction_temp_c: float = 90
    extraction_time_s: int = 240
    shake_count: int = 12
    serving_temp_c: float = 4

class Recipe(BaseModel):
    recipe_id: str
    style: Style
    cup_volume_ml: CupSize
    ingredients: dict[str, float]    # {vocab_id: gram}
    process: Process = Field(default_factory=Process)
    sugar_level: SugarLevel
    metadata: dict = {}

    @model_validator(mode="after")
    def check_nonempty(self):
        if not self.ingredients:
            raise ValueError("ingredients must not be empty")
        return self
```

**测试要点**:序列化 → YAML → 反序列化 round-trip 等价;不合法 `cup_volume_ml=600` 被 pydantic 拒绝。

---

### 6.3 `simulators/health/calculator.py`

**职责**:输入 Recipe → 输出 nutrition dict。纯查表 + 加权,无模型。

```python
from collections import defaultdict
from ..ingredients.vocab import Vocab
from ..recipes.schema import Recipe

# 吸水/缩水修正系数(干态 → 湿态 / 即食态)
COOKING_FACTOR = {
    "topping_brown_pearl": 2.5,      # 干珍珠泡发后增重
    "topping_taro_ball": 1.4,
    "topping_red_bean": 2.2,
    "grain_chia": 8.0,                # 奇亚籽吸水巨大
    "grain_basil_seed": 10.0,
    # 其他默认 1.0
}

def compute_nutrition(recipe: Recipe, vocab: Vocab) -> dict:
    totals = defaultdict(float)
    allergens = set()
    has_trans_fat = False
    missing = []

    for ing_id, mass_g in recipe.ingredients.items():
        if ing_id not in vocab:
            missing.append(ing_id)
            continue
        ing = vocab.get(ing_id)
        actual_g = mass_g * COOKING_FACTOR.get(ing_id, 1.0)
        for nutrient, val in ing.nutrition_per_100g.model_dump().items():
            if val is not None:
                totals[nutrient] += val * actual_g / 100

        allergens.update(ing.allergens)
        if ing.nutrition_per_100g.trans_fat_g > 0:
            has_trans_fat = True

    return {
        "energy_kcal": totals["energy_kcal"],
        "sugar_g": totals["sugar_g"],
        "fat_g": totals["fat_g"],
        "trans_fat_g": totals["trans_fat_g"],
        "caffeine_mg": totals["caffeine_mg"],
        "sodium_mg": totals["sodium_mg"],
        "allergens": sorted(allergens),
        "has_trans_fat": has_trans_fat,
        "missing_nutrition_for": missing,
    }
```

**测试要点**:
- 一杯 500ml 全糖珍珠奶茶热量在 [400, 500] kcal(行业典型值校验)
- 含 `alt_milk_creamer` 的配方 `has_trans_fat=True`
- 缺失原料触发 `missing_nutrition_for` 非空

---

### 6.4 `constraints/checker.py`

**职责**:执行 §E.3 物理硬约束 + §3.4 业务约束,返回违规列表。

```python
from ..recipes.schema import Recipe

class ConstraintViolation(BaseModel):
    code: str
    severity: Literal["hard", "soft"]
    message: str

def check_constraints(
    recipe: Recipe,
    nutrition: dict,
    targets: dict,
    vocab: Vocab,
) -> list[ConstraintViolation]:
    v = []

    # 1. 杯容守恒
    total_volume = sum(recipe.ingredients.values())  # 简化:1g ≈ 1ml
    if not (0.85 * recipe.cup_volume_ml <= total_volume <= 1.10 * recipe.cup_volume_ml):
        v.append(ConstraintViolation(
            code="VOLUME_OVERFLOW", severity="hard",
            message=f"总量 {total_volume:.0f}g 不在 [{0.85*recipe.cup_volume_ml:.0f}, {1.10*recipe.cup_volume_ml:.0f}]"
        ))

    # 2. 法规
    if nutrition["caffeine_mg"] > 200:
        v.append(ConstraintViolation(code="CAFFEINE_GB", severity="hard",
                                      message=f"咖啡因 {nutrition['caffeine_mg']:.0f}mg 超国标"))
    if nutrition["has_trans_fat"] and targets.get("trans_fat_zero", False):
        v.append(ConstraintViolation(code="TRANS_FAT", severity="hard",
                                      message="含反式脂肪,违反目标"))

    # 3. 用户硬约束
    if nutrition["sugar_g"] > targets.get("sugar_limit", 999):
        v.append(ConstraintViolation(code="SUGAR_LIMIT", severity="hard",
                                      message=f"含糖 {nutrition['sugar_g']:.1f}g 超 {targets['sugar_limit']}g"))

    # 4. 化学不兼容
    has_soda = any(i in recipe.ingredients for i in ["aux_soda_water", "aux_sparkling_water"])
    has_dairy = any(vocab.get(i).category == "dairy_base"
                    for i in recipe.ingredients if i in vocab)
    if has_soda and has_dairy:
        v.append(ConstraintViolation(code="SODA_DAIRY", severity="hard",
                                      message="苏打水 + 乳基会凝固"))

    # 5. 配料上限
    n_toppings = sum(1 for i in recipe.ingredients
                     if i in vocab and vocab.get(i).category == "topping")
    if n_toppings > 3:
        v.append(ConstraintViolation(code="TOPPING_COUNT", severity="soft",
                                      message=f"配料 {n_toppings} 种 > 3"))

    # 6. 致敏原排除
    excluded = set(targets.get("excluded_allergens", []))
    if excluded & set(nutrition["allergens"]):
        v.append(ConstraintViolation(code="ALLERGEN", severity="hard",
                                      message=f"含排除致敏原: {excluded & set(nutrition['allergens'])}"))

    return v

def is_feasible(violations: list[ConstraintViolation]) -> bool:
    return not any(v.severity == "hard" for v in violations)
```

**测试要点**:每条约束至少一个 positive case 一个 negative case;`is_feasible` 在无硬违规时 True。

---

### 6.5 `recipes/generator.py` + `recipes/reconciler.py`

**职责**:§E.2 的 9 步层级参数化生成 + Step 10 守恒重标定。

```python
import numpy as np
from numpy.random import default_rng
from ..priors.engine import PriorEngine
from .schema import Recipe, Process

class RecipeGenerator:
    def __init__(self, vocab: Vocab, prior: PriorEngine, rng_seed: int = 42):
        self.vocab = vocab
        self.prior = prior
        self.rng = default_rng(rng_seed)

    def generate(self, planner_output: dict, n_candidates: int = 200) -> list[Recipe]:
        recipes = []
        for _ in range(n_candidates):
            r = self._one_recipe(planner_output)
            if r is not None:
                recipes.append(r)
        return self._dedupe(recipes)

    def _one_recipe(self, planner: dict) -> Recipe | None:
        # Step 1: 风格(从 planner 提示或采样)
        style = planner.get("style_hint") or self.rng.choice(
            ["纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调"]
        )
        # Step 2: 杯型
        cup = planner.get("cup_volume_ml", 500)
        # Step 3: Dirichlet 体积分配
        alpha = self.prior.get_dirichlet_alpha(style, planner.get("context", {}))
        partition = self.rng.dirichlet(alpha)   # (tea, milk, fruit, water, coffee, ice)
        vols = (partition * cup).tolist()
        tea_v, milk_v, fruit_v, water_v, coffee_v, ice_g = vols

        # Step 4-9 (示意,实际更复杂)
        tea_id = self._pick_tea(planner)
        ingredients = {tea_id: tea_v}

        if milk_v > 5:
            dairy_id = self._pick_dairy(planner)
            ingredients[dairy_id] = milk_v
        # ... fruit / sweetener / topping / flavoring / ice 同理

        # 糖度档位
        level = planner.get("sugar_level", "五分")
        sugar_g = {"无糖": 0, "三分": 8, "五分": 13, "七分": 18, "全糖": 25}[level]
        ingredients[self._pick_sweetener(planner)] = sugar_g

        # 配料
        n_topping = self.rng.integers(0, 3, endpoint=True)
        toppings = self._sample_compatible_toppings(n_topping, style)
        for t in toppings:
            ingredients[t] = self.vocab.get(t).typical_serving_g * self.rng.normal(1.0, 0.2)

        if ice_g > 5:
            ingredients["aux_ice_cube"] = ice_g

        recipe = Recipe(
            recipe_id=f"gen_{self.rng.integers(1e9):09d}",
            style=style, cup_volume_ml=cup,
            ingredients={k: round(max(v, 0.1), 1) for k, v in ingredients.items()},
            sugar_level=level,
            metadata={"planner": planner},
        )
        return reconcile(recipe)  # Step 10

    # ... helpers omitted

def reconcile(recipe: Recipe) -> Recipe | None:
    """Step 10: 总量超 cup × 1.10 → 按比例缩液体;总量低于 cup × 0.85 → 补水"""
    total = sum(recipe.ingredients.values())
    upper = recipe.cup_volume_ml * 1.10
    lower = recipe.cup_volume_ml * 0.85
    if total > upper:
        scale = upper / total
        # 只缩液体(非 topping/sweetener)
        liquid_keys = [k for k in recipe.ingredients
                       if k.startswith(("tea_", "dairy_", "alt_milk_",
                                        "coffee_", "fruit_", "aux_pure", "aux_soda"))]
        for k in liquid_keys:
            recipe.ingredients[k] *= scale
    elif total < lower:
        recipe.ingredients["aux_pure_water"] = (
            recipe.ingredients.get("aux_pure_water", 0) + (lower - total)
        )
    return recipe
```

**测试要点**:
- 1000 次 `generate()` 调用,Reconciler 后 100% 满足杯容约束
- 同一 planner_output 不同 seed 产出多样化候选(Jaccard > 0.3 平均)
- 风格 "纯茶" 产出的配方不含 `dairy_*`

---

### 6.6 `simulators/sensory/model.py`(GNN)

**职责**:§3.3.1 双层输出头 GAT。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

CORE_DIMS = ["甜度", "苦度", "茶香", "奶香", "喜爱度"]               # 5,双通路共训
EXT_DIMS  = ["涩", "酸", "回甘", "顺滑", "果香",
             "咸", "油腻", "清新", "浓郁", "层次"]                    # 10,仅路径 A

class SensoryGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden, heads=heads)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=heads)
        self.proj = nn.Linear(hidden * heads * 2, 256)
        # 双层输出头
        self.head_core_mean = nn.Linear(256 + N_CUSTOM_FEAT, len(CORE_DIMS))
        self.head_core_logvar = nn.Linear(256 + N_CUSTOM_FEAT, len(CORE_DIMS))
        self.head_ext_mean = nn.Linear(256 + N_CUSTOM_FEAT, len(EXT_DIMS))
        self.head_ext_logvar = nn.Linear(256 + N_CUSTOM_FEAT, len(EXT_DIMS))

    def forward(self, data: Batch, customization: torch.Tensor):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.elu(self.conv2(x, ei))
        # 读出
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        g = F.elu(self.proj(g))
        g = torch.cat([g, customization], dim=-1)
        return {
            "core_mean":   self.head_core_mean(g),
            "core_logvar": self.head_core_logvar(g),
            "ext_mean":    self.head_ext_mean(g),
            "ext_logvar":  self.head_ext_logvar(g),
        }

def nll_gaussian(target, mean, logvar):
    """异方差高斯 NLL,用于 Stage 2"""
    inv_var = torch.exp(-logvar)
    return 0.5 * (inv_var * (target - mean)**2 + logvar).mean()
```

**训练循环要点**:
- Stage 1:用 `nll_gaussian` 训 4 个头(core + ext);路径 A 的"unknown 维度"用 mask 跳过
- Stage 2:冻结 `conv1/2/proj` + `head_ext_*`,只 fine-tune `head_core_*`;NLL loss + 早停

---

### 6.7 `simulators/sales/model.py`

**职责**:§3.3.2 LightGBM + 品牌固定效应 + K 折交叉拟合残差。

```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

class SalesPredictor:
    def __init__(self, k: int = 5):
        self.k = k
        self.baseline_models: list[lgb.LGBMRegressor] = []
        self.recipe_model: lgb.LGBMRegressor | None = None

    def fit(self, df_baseline, y, df_recipe_embed):
        """两步训练: baseline (with 品牌 FE) + recipe (拟合残差)"""
        residuals = np.zeros_like(y, dtype=float)
        for tr, va in KFold(self.k, shuffle=True, random_state=42).split(df_baseline):
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
            m.fit(df_baseline.iloc[tr], y[tr])
            residuals[va] = y[va] - m.predict(df_baseline.iloc[va])
            self.baseline_models.append(m)

        # recipe 模型拟合残差
        self.recipe_model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03,
            objective="quantile", alpha=0.5,  # 中位数预测,稳健
        )
        self.recipe_model.fit(df_recipe_embed, residuals)

        # 同时训 0.05 / 0.95 分位数用于 LCB
        self.q05 = lgb.LGBMRegressor(objective="quantile", alpha=0.05,
                                      n_estimators=500).fit(df_recipe_embed, residuals)
        self.q95 = lgb.LGBMRegressor(objective="quantile", alpha=0.95,
                                      n_estimators=500).fit(df_recipe_embed, residuals)

    def predict(self, baseline_feat, recipe_embed) -> dict:
        base = np.mean([m.predict(baseline_feat) for m in self.baseline_models], axis=0)
        recipe_contrib = self.recipe_model.predict(recipe_embed)
        # σ 估计
        q05 = self.q05.predict(recipe_embed)
        q95 = self.q95.predict(recipe_embed)
        sigma = (q95 - q05) / 3.29  # ≈ 95% CI 的标准差
        return {"mean": base + recipe_contrib, "sigma": sigma,
                "baseline": base, "recipe_contrib": recipe_contrib}
```

**测试要点**:
- K 折 baseline 不在自身样本上预测(防数据泄漏)
- `sigma > 0` 在所有预测点上
- Spearman ρ ≥ 0.4 on held-out test set

---

### 6.8 `priors/engine.py` + `priors/dirichlet.py`

**职责**:§E.9.2/9.3 上下文条件 Dirichlet + Bayesian 后验更新。

```python
from datetime import datetime
import json
import numpy as np
import yaml

ROLES = ["tea", "milk", "fruit", "water", "coffee", "ice"]

class PriorEngine:
    def __init__(self, alpha_path: str, deltas_path: str, history_dir: str):
        with open(alpha_path) as f:
            self.base_alpha = yaml.safe_load(f)              # {style: [α_role...]}
        with open(deltas_path) as f:
            self.context_deltas = yaml.safe_load(f)
        self.history_dir = history_dir

    def get_dirichlet_alpha(self, style: str, context: dict) -> np.ndarray:
        a = np.array(self.base_alpha[style], dtype=float)
        for feat, val in context.items():
            if feat in self.context_deltas and val in self.context_deltas[feat]:
                delta = self.context_deltas[feat][val]
                a += np.array([delta.get(r, 0) for r in ROLES])
        return np.clip(a, 0.05, None)

    def update_dirichlet_posterior(
        self,
        style: str,
        observed_recipes: list,
        scores: np.ndarray,
        learning_rate: float = 0.3,
    ) -> np.ndarray:
        """Bayesian 共轭更新,只用 Top 30% 配方"""
        if len(observed_recipes) < 5:
            return np.array(self.base_alpha[style])
        thresh = np.percentile(scores, 70)
        good = [r for r, s in zip(observed_recipes, scores) if s >= thresh]
        if len(good) < 3:
            return np.array(self.base_alpha[style])

        partitions = np.stack([_partition_of(r) for r in good])  # (n_good, 6)
        a_prior = np.array(self.base_alpha[style])
        a_obs   = len(good) * partitions.mean(axis=0)
        a_new   = a_prior + learning_rate * a_obs

        # 写 snapshot
        snap = {
            "style": style, "timestamp": datetime.utcnow().isoformat(),
            "n_good": len(good), "learning_rate": learning_rate,
            "alpha_prior": a_prior.tolist(), "alpha_new": a_new.tolist(),
        }
        path = f"{self.history_dir}/{style}_{snap['timestamp']}.json"
        with open(path, "w") as f:
            json.dump(snap, f, indent=2)

        # 更新内存(下次调用立即生效)
        self.base_alpha[style] = a_new.tolist()
        return a_new

def _partition_of(recipe) -> np.ndarray:
    """从 Recipe 提取 6 角色体积比"""
    role_mass = {r: 0.0 for r in ROLES}
    for ing_id, mass in recipe.ingredients.items():
        if ing_id.startswith("tea_"):     role_mass["tea"] += mass
        elif ing_id.startswith(("dairy_", "alt_milk_")):  role_mass["milk"] += mass
        elif ing_id.startswith("fruit_"): role_mass["fruit"] += mass
        elif ing_id.startswith("coffee_"): role_mass["coffee"] += mass
        elif ing_id in ("aux_pure_water", "aux_soda_water"): role_mass["water"] += mass
        elif ing_id.startswith("aux_ice"): role_mass["ice"] += mass
    total = sum(role_mass.values())
    return np.array([role_mass[r] / total if total > 0 else 0.0 for r in ROLES])
```

**测试要点**:
- α 始终 > 0
- 后验更新连续调用 10 次,α 不会发散(KL bound test)
- Snapshot 文件可正确反序列化恢复

---

### 6.9 `optimizer/`(NSGA-II + LCB + MMR)

```python
# optimizer/problem.py
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice

class BeverageProblem(ElementwiseProblem):
    def __init__(self, vocab, simulators, planner_output, kappa=1.0):
        # vars 由 §3.5 search_space 决定
        vars_ = {
            "tea_id":   Choice(options=[i.id for i in vocab.by_category("tea_base")]),
            "milk_id":  Choice(options=["none"] + [i.id for i in vocab.by_category("dairy_base")]),
            "tea_v":    Real(bounds=(150, 350)),
            "milk_v":   Real(bounds=(0, 200)),
            "sugar_g":  Real(bounds=(0, 30)),
            # ... 等
        }
        super().__init__(vars=vars_, n_obj=4, n_constr=0)
        self.simulators = simulators
        self.kappa = kappa
        self.planner = planner_output

    def _evaluate(self, x: dict, out, *args, **kwargs):
        recipe = self._x_to_recipe(x)
        sensory = self.simulators["sensory"].predict(recipe)
        sales = self.simulators["sales"].predict(recipe)
        nutrition = self.simulators["health"](recipe)

        # LCB / UCB
        pref_lcb  = sensory["喜爱度"]["mean"] - self.kappa * sensory["喜爱度"]["sigma"]
        sales_lcb = sales["mean"] - self.kappa * sales["sigma"]
        cost = self._compute_cost(recipe)
        sugar = nutrition["sugar_g"]

        out["F"] = [-pref_lcb, -sales_lcb, cost, sugar]
```

```python
# optimizer/mmr.py
import numpy as np

def mmr_select(pareto_front, embeddings, k: int, lam: float = 0.6) -> list[int]:
    """λ × 相关性 + (1-λ) × 多样性"""
    n = len(pareto_front)
    scores = np.array([-p.objective_sum for p in pareto_front])
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    selected = [int(np.argmax(scores))]
    remaining = set(range(n)) - set(selected)
    while len(selected) < k and remaining:
        best_i, best_score = None, -np.inf
        for i in remaining:
            div = min(np.linalg.norm(embeddings[i] - embeddings[j]) for j in selected)
            s = lam * scores[i] + (1 - lam) * div
            if s > best_score:
                best_score, best_i = s, i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected
```

**测试要点**:1000 次 `evaluate` 调用全部产出有效目标值;Top-K 在 GNN embedding 距离上有真正差异(平均 L2 > 阈值)。

---

### 6.10 `planner/llm_planner.py`

**职责**:§3.1。Claude API + 严格 JSON Schema 输出。

```python
import json
import os
from anthropic import Anthropic

PLANNER_SCHEMA = {
    "type": "object",
    "required": ["style_hint", "cup_volume_ml", "sugar_level", "health"],
    "properties": {
        "style_hint": {"enum": ["纯茶", "奶茶", "果茶", "咖啡奶茶", "冰沙", "特调"]},
        "cup_volume_ml": {"enum": [380, 500, 700]},
        "sugar_level": {"enum": ["无糖", "三分", "五分", "七分", "全糖"]},
        "health": {
            "type": "object",
            "properties": {
                "sugar_limit": {"type": "number"},
                "calorie_limit": {"type": "number"},
                "trans_fat_zero": {"type": "boolean"},
                "excluded_allergens": {"type": "array", "items": {"type": "string"}},
            },
        },
        "context": {"type": "object"},  # season, target_age, ...
        "flavor_keywords": {"type": "array", "items": {"type": "string"}},
        "price_range_cny": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    },
}

class LLMPlanner:
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def plan(self, user_request: str) -> dict:
        msg = self.client.messages.create(
            model=self.model, max_tokens=1024,
            system=(
                "你是茶饮研发助手, 把用户的自然语言需求转成结构化目标。"
                "严格按给定 JSON Schema 输出, 不要任何额外文字。"
                f"Schema: {json.dumps(PLANNER_SCHEMA, ensure_ascii=False)}"
            ),
            messages=[{"role": "user", "content": user_request}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
```

**测试要点**:50 个 fixture 输入,JSON 解析成功率 ≥ 95%;关键约束召回率 ≥ 90%。

---

### 6.11 `feedback/recorder.py`

**职责**:§3.7 反馈结构化存盘 + 触发更新。

```python
import duckdb
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    session_id  VARCHAR,
    recipe_id   VARCHAR,
    recipe_json JSON,
    pred_json   JSON,            -- 系统预测
    actual_json JSON,            -- 实际盲品 + 销量(如有)
    context_json JSON,
    ts          TIMESTAMP,
);

CREATE TABLE IF NOT EXISTS panel_score (
    session_id  VARCHAR,
    recipe_id   VARCHAR,
    panelist_id VARCHAR,
    dimension   VARCHAR,
    score       SMALLINT,        -- Likert 1-5
    cup_order   SMALLINT,        -- Latin square 顺序
    block       SMALLINT,        -- BIBD 块号
    session_dt  TIMESTAMP,
);
"""

class FeedbackRecorder:
    def __init__(self, db_path="data/feedback.duckdb"):
        self.con = duckdb.connect(db_path)
        self.con.execute(SCHEMA)

    def record_recipe(self, session_id, recipe, predicted, context):
        self.con.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?,?)",
            [session_id, recipe.recipe_id, recipe.model_dump_json(),
             json.dumps(predicted), None, json.dumps(context), datetime.utcnow()]
        )

    def record_panel(self, session_id, recipe_id, panelist_id, dim, score, cup_order, block):
        self.con.execute(
            "INSERT INTO panel_score VALUES (?,?,?,?,?,?,?,?)",
            [session_id, recipe_id, panelist_id, dim, score, cup_order, block, datetime.utcnow()]
        )

    def get_panel_for_update(self, session_id) -> "pd.DataFrame":
        return self.con.execute(
            "SELECT * FROM panel_score WHERE session_id = ?", [session_id]
        ).df()
```

---

### 6.12 `pipeline/end_to_end.py`

**职责**:把上面所有模块串起来。

```python
def run_pipeline(user_request: str, top_k: int = 5) -> dict:
    # Stage 1: 解析
    planner = LLMPlanner()
    spec = planner.plan(user_request)

    # Stage 2: 生成候选 (warm start for NSGA-II)
    vocab = Vocab("data/ingredients/ingredient_vocab.yaml")
    prior = PriorEngine(...)
    generator = RecipeGenerator(vocab, prior)
    seeds = generator.generate(spec, n_candidates=300)

    # Stage 3: 加载仿真器
    simulators = {
        "sensory": SensoryPredictor("models/sensory_gnn_v1.pt", vocab),
        "sales":   SalesPredictor.load("models/sales_v1.pkl"),
        "health":  lambda r: compute_nutrition(r, vocab),
    }

    # Stage 4: 多目标优化
    problem = BeverageProblem(vocab, simulators, spec, kappa=1.0)
    algorithm = build_mixed_variable_ga(seeds=seeds, pop_size=200)
    res = minimize(problem, algorithm, ("n_gen", 50), seed=42)

    # Stage 5: MMR Top-K
    pareto = res.pop
    embeddings = [simulators["sensory"].embed(p.recipe) for p in pareto]
    top_indices = mmr_select(pareto, embeddings, k=top_k)
    top_recipes = [pareto[i].recipe for i in top_indices]

    # Stage 6: 落盘 + 返回
    recorder = FeedbackRecorder()
    session_id = f"s_{int(time.time())}"
    for r in top_recipes:
        recorder.record_recipe(session_id, r, predictions_of(r), spec.get("context"))

    return {
        "session_id": session_id,
        "spec": spec,
        "top_recipes": [r.model_dump() for r in top_recipes],
        "predictions": [predictions_of(r) for r in top_recipes],
    }
```

---

### 6.13 `demo/app.py`(Gradio)

```python
import gradio as gr
from beverage_ai.pipeline import run_pipeline

def gradio_handler(text):
    result = run_pipeline(text, top_k=5)
    cards = []
    for i, (r, p) in enumerate(zip(result["top_recipes"], result["predictions"])):
        cards.append(
            f"### 配方 {i+1}\n"
            f"- 风格: {r['style']} {r['cup_volume_ml']}ml {r['sugar_level']}\n"
            f"- 主料: {', '.join(f'{k} {v:.0f}g' for k, v in r['ingredients'].items())}\n"
            f"- 预测喜爱度: {p['sensory']['喜爱度']['mean']:.2f} ± {p['sensory']['喜爱度']['sigma']:.2f}\n"
            f"- 含糖: {p['nutrition']['sugar_g']:.1f}g  热量: {p['nutrition']['energy_kcal']:.0f}kcal\n"
        )
    return "\n\n---\n\n".join(cards)

demo = gr.Interface(
    fn=gradio_handler,
    inputs=gr.Textbox(label="需求描述", lines=3,
                      placeholder="想为夏季年轻女性开发一款主打健康轻负担的茶饮新品..."),
    outputs=gr.Markdown(label="Top-5 候选配方"),
    title="茶饮研发闭环 AI 系统 v1",
    examples=[
        "夏季年轻女性低糖轻负担,定价 18-22 元",
        "冬天上班族暖身高蛋白,接受 30 元",
        "无糖控碳水,只要茶味鲜明",
    ],
)

if __name__ == "__main__":
    demo.launch()
```

---

## 7. 8 周排期(每周明确 deliverable 与验收标准)

### Week 1 — 项目脚手架 + 词表 + Schema + 健康/约束

**任务**

- [ ] `pyproject.toml` + `uv venv` 环境初始化
- [ ] GitHub repo + CI(ruff + pytest 在 push 时跑)
- [ ] 录入 207 条原料词表至 `ingredient_vocab.yaml`(数据 owner 主导)
- [ ] 实现 `ingredients/vocab.py`(本文 §6.1)
- [ ] 实现 `recipes/schema.py`(本文 §6.2)
- [ ] 实现 `simulators/health/calculator.py`(本文 §6.3)
- [ ] 实现 `constraints/checker.py`(本文 §6.4)
- [ ] 启动 reviews 爬虫(后台 24h 跑,Week 2 收割)

**Deliverable**

- 端到端 CLI:`python -m beverage_ai.cli health <recipe.yaml>` 输出营养标签 + 约束检查结果

**验收标准**

- 所有单元测试通过(覆盖率 ≥ 80%)
- 一杯 500ml 全糖珍珠奶茶热量预测在 [400, 500] kcal
- 含 `alt_milk_creamer` 的 fixture 被识别出反式脂肪

---

### Week 2 — 数据基础设施 + 用量先验拟合

**任务**

- [ ] DuckDB schema 上线(`feedback.duckdb`)
- [ ] W&B 项目初始化,跑通一次 dummy log
- [ ] 完成 30 SKU 逆向称重(数据 owner 实地)→ `reverse_engineered_v1.yaml`
- [ ] 完成 100+ 公开复刻配方收集 → `reference_recipes_v1.yaml`
- [ ] 实现 `scripts/fit_dirichlet_priors.py`,从参考配方拟合 α
- [ ] 校准 207 条原料的 `typical_serving_g`
- [ ] reviews 爬虫产出 5–10 万条原始数据
- [ ] 用 Claude Haiku 批量抽 aspect(50K × $0.002 ≈ $100)

**Deliverable**

- `data/priors/dirichlet_alpha_v1.yaml` 已落盘
- `data/reviews/reviews_v1.parquet` 已落盘
- 完成 Phase 1(技术方案 §7.1)

**验收标准**

- Dirichlet α 拟合脚本可复现(同 seed 二次执行 KL < 1e-6)
- Reviews aspect 抽取一致性审计(100 条人工对比)≥ 80%

---

### Week 3 — 配方生成器 + LLM Planner + 条件 Dirichlet

**任务**

- [ ] 实现 `priors/engine.py` + `priors/dirichlet.py`(本文 §6.8)
- [ ] 实现 `recipes/generator.py` + `recipes/reconciler.py`(本文 §6.5)
- [ ] 实现 `planner/llm_planner.py`(本文 §6.10)
- [ ] 编写 `topping_compatibility.yaml`(~50 对)
- [ ] 编写 `aliases.yaml`(~100 别名)
- [ ] 与 §3.4 约束检查器集成(生成 → 重标定 → 检查)

**Deliverable**

- `python -m beverage_ai.cli generate "<需求>" --n 200` 输出 200 个合规候选

**验收标准**

- 1000 次生成,经 reconciler 后 100% 满足杯容约束
- LLM Planner 在 50 条 fixture 输入下,JSON 解析成功率 ≥ 95%
- 风格 = "纯茶" 的产出 0% 含乳基

---

### Week 4 — 仿真器 v1 训练

**任务**

- [ ] 实现 `simulators/sensory/{model,data,train}.py`(本文 §6.6)
- [ ] GNN Stage 1 预训练(GPU 4–8h)
- [ ] 实现 `simulators/sales/{model,train}.py`(本文 §6.7)
- [ ] 训销量预测器(品牌 FE + K=5 折)
- [ ] 实现 `simulators/repurchase/v1_weighted.py`(加权公式 + 网格搜索 α/β/γ)
- [ ] 训练日志全部上 W&B

**Deliverable**

- `models/sensory_gnn_stage1.pt`, `models/sales_v1.pkl` 落盘
- 单输入 Recipe → 4 个仿真器各自的 prediction dict

**验收标准**

- 感官 GNN 在 path A test set 上,5 个 core 维度 Pearson r ≥ 0.5
- 销量预测器 Spearman ρ ≥ 0.4(已下架 SKU 子集 recall ≥ 0.6)
- 复购模块输出在 [0, 1] 范围

---

### Week 5 — 优化器 + 端到端集成

**任务**

- [ ] 实现 `optimizer/{problem,nsga2,acquisition,mmr}.py`(本文 §6.9)
- [ ] 实现 `pipeline/end_to_end.py`(本文 §6.12)
- [ ] 跑通 3 个不同 planner_output 的场景:
  - "夏季年轻女性低糖"
  - "冬季上班族暖身"
  - "无糖纯茶"
- [ ] 每场景产出 Top-5 配方 + 预测可视化

**Deliverable**

- `scripts/run_pipeline_demo.py` 输出 3 个场景的 Markdown 报告

**验收标准**

- 1000 次 NSGA-II evaluate 调用 0 次 crash
- Top-5 在 GNN embedding 距离上,任两个的 L2 ≥ 阈值(平均 > 中位数 × 1.5)
- 端到端单次运行 < 30 秒(CPU)

---

### Week 6 — Gradio Demo + 文档收尾(Phase 3 闭关)

**任务**

- [ ] 实现 `demo/app.py`(本文 §6.13)
- [ ] 编写 `README.md`(快速上手 + 完整复现指南)
- [ ] 文档站点(可选):mkdocs + GitHub Pages
- [ ] 录制 5 分钟 demo 视频(OBS / Loom)
- [ ] Phase 3 中期评审

**Deliverable**

- `https://<user>.gradio.live/...` 可访问(临时)
- Demo 视频上传

**验收标准**

- 评委从 README 0 知识起步,30 分钟内能跑通 demo
- Gradio 界面在 3 个 example 输入下都给出合理候选

---

### Week 7 — 盲品执行 + Stage 2 微调

**任务**

- [ ] 准备 21 杯配方:Top-5 系统 + 7 随机基线 + 7 专家对照(其中 5 个市售热销 + 2 自调)
- [ ] 采购原料 + 厨房制作(2 天)
- [ ] 招募 35+ 评委(目标人群匹配)
- [ ] 用 `scripts/panel_design.py` 生成 BIBD 分配 + Latin square 顺序
- [ ] 执行 2 场盲品(每人每场 ≤ 3 杯,中间 ≥ 1 天)
- [ ] 评分录入 DuckDB
- [ ] GNN Stage 2 微调(GPU < 30 min)
- [ ] 首次 Bayesian Dirichlet 后验更新(`learning_rate=0.3`)

**Deliverable**

- `data/panel/tasting_v1.parquet`(~1200 行)
- `models/sensory_gnn_v1.pt`(Stage 2 后)
- `data/priors/prior_history/2026-W7_*.json`

**验收标准**

- 评委有效完成率 ≥ 90%(剔除中途退出)
- Stage 2 微调后,core 5 维 Pearson r ≥ 0.6(LOO 交叉验证)

---

### Week 8 — 闭环验证 + 报告 + 交付

**任务**

- [ ] 实现 `scripts/analyze_panel.py`:
  - 主指标:A vs B 整体喜爱度 Wilcoxon signed-rank
  - 次要:A vs C 等价性检验(TOST)
  - 多维度:Friedman + Nemenyi(Bonferroni α = 0.05/6)
  - 嵌套结构:`statsmodels` 混合效应模型
  - **必报告效应量**(rank-biserial r 或 Cohen's d)
- [ ] 写实验报告 `docs/v1_experiment_report.md`
- [ ] `typical_serving_g` 在线校准(Week 7 数据驱动)
- [ ] 最终 demo 视频 + 项目交付包

**Deliverable**

- `docs/v1_experiment_report.md`(含主试验 p 值 + 效应量 + 95% CI 图)
- `release/v1.0.zip`(代码 + 模型 + 数据 + 文档)
- 完成度 100% checkpoint

**验收标准**

- A 组喜爱度均值 > B 组,Wilcoxon p < 0.05,效应量 r ≥ 0.3
- 闭环回流端到端测试通过(`tests/integration/test_feedback_loop.py`)
- `release/v1.0.zip` 在干净虚拟机上 30 分钟内 reproduce 主结果

---

## 8. 测试策略

### 8.1 单元测试(每模块独立)

| 模块 | 覆盖率目标 | 关键 case |
|---|---|---|
| `ingredients/vocab` | ≥ 95% | 207 条全校验、缺字段告警 |
| `recipes/schema` | ≥ 95% | round-trip、不合法 cup 拒绝 |
| `recipes/generator` | ≥ 80% | property test:1000 次生成全合规 |
| `recipes/reconciler` | ≥ 90% | 边界 case(全溢出 / 全亏空) |
| `constraints/checker` | ≥ 95% | 每条约束 ≥ 1 pos + 1 neg |
| `simulators/health` | ≥ 90% | 行业典型值 sanity check |
| `simulators/sensory` | ≥ 70% | forward 不 crash + shape 正确 |
| `simulators/sales` | ≥ 80% | K 折无数据泄漏 |
| `priors/dirichlet` | ≥ 90% | KL bound、snapshot 反序列化 |
| `optimizer/mmr` | ≥ 90% | 数学性质(k=1 = argmax,k=n = 全选) |
| `optimizer/acquisition` | ≥ 95% | LCB 公式正确 |
| `planner/llm_planner` | ≥ 70% | mock API,50 条 fixture 测试 |
| `feedback/recorder` | ≥ 85% | DuckDB schema 兼容 |

**总目标**:整体覆盖率 ≥ 80%。

### 8.2 集成测试

- `tests/integration/test_pipeline.py`:端到端,用 stub LLM 跑通
- `tests/integration/test_optimizer.py`:NSGA-II 1000 次 evaluate 不 crash
- `tests/integration/test_feedback_loop.py`:模拟 30 个评委盲品 → 后验更新 → 下一轮生成器 α 变化

### 8.3 数据质量测试

- `scripts/qc_data.py`:每次 pull 数据后自动跑
  - 词表 schema 100% 校验
  - reviews aspect 抽取一致性抽样审计
  - 销量数据缺失率 < 10%

### 8.4 性能测试

- 端到端单次 < 30s(CPU)
- 内存峰值 < 4GB
- 优化器 50 代 × pop=200 < 60s

### 8.5 CI(GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with: { python-version: "3.11" }
      - run: pip install uv && uv sync --dev
      - run: ruff check .
      - run: mypy beverage_ai
      - run: pytest --cov=beverage_ai --cov-report=term-missing
```

---

## 9. 部署与运维(v1 学生项目级)

### 9.1 本地运行

```bash
# 一次性环境
uv venv && uv sync
cp .env.example .env  # 填 ANTHROPIC_API_KEY

# 一次推理
python -m beverage_ai.cli generate "夏季年轻女性低糖" --n 200

# 端到端 demo
python -m beverage_ai.pipeline.end_to_end "夏季年轻女性低糖"

# Gradio Web Demo
python demo/app.py
```

### 9.2 模型 / 数据版本管理

- 模型:`models/*.pt` + `models/MANIFEST.yaml`(记录训练 commit + 训练数据 snapshot id)
- 数据:`data/*_v{N}.parquet`,所有更新走 PR + `data/CHANGELOG.md`
- 先验:`data/priors/prior_history/<timestamp>_<style>.json` 永久保留

### 9.3 闭环更新触发

```bash
# Week 7+ 盲品后每月跑一次
python scripts/update_from_feedback.py --session $SID
# 内部:
#   1. 拉取 DuckDB 中 session_id 的所有盲品
#   2. 触发感官 GNN Stage 2 增量微调
#   3. 触发 Dirichlet 后验更新(每个 style 单独)
#   4. 触发 typical_serving_g 中位数校准
#   5. 把新模型/先验落盘到带版本号的路径
#   6. 旧版自动归档,可一行命令回滚
```

---

## 10. 已知限制 + v2 路径

### 10.1 v1 不能做的事

1. **首次冷启动**:Phase 4 之前,Pareto 前沿可能包含模型幻觉解(LCB κ=1 只能部分缓解)
2. **失败 SKU 覆盖**:销量代理标签的幸存者偏差只能部分修正(见 §3.3.2 R11)
3. **复购预测独立性**:v1 本质是喜爱度的别名,不应在报告中夸大
4. **大众点评爬虫合规风险**:学术豁免在中国法律下不明确(见 §附录 C2)
5. **盲品评委代表性**:35 人同学/室友群体不代表全国市场目标人群

### 10.2 v2 路径(本期之后)

| v2 项 | 触发条件 | 预估工时 |
|---|---|---|
| Conditional VAE 生成器 | ≥ 5000 配方-评分对(约 1 年闭环) | 3 周 |
| Double ML 销量预测 | ≥ 3000 SKU 含失败品 | 2 周 |
| DeepSurv 复购模型 | 真实 POS 数据接入 | 4 周 |
| Active learning 闭环采样 | 不确定性校准 ECE < 0.05 | 1 周 |
| 门店 A/B 测试自动化 | 合作品牌 + 后端对接 | 4 周 |

---

## 附录 A. `pyproject.toml` 完整版

(本文 §3.1 已基本完整,补 dev 与 scrape extras 见 §3.1。)

---

## 附录 B. 关键 YAML schema 示例

### B.1 `dirichlet_alpha_v1.yaml`(初始)

```yaml
# 6 风格 × 6 角色 (tea, milk, fruit, water, coffee, ice)
纯茶:        [8.0, 0.0, 0.3, 1.0, 0.0, 2.0]
奶茶:        [3.0, 1.5, 0.2, 0.3, 0.0, 1.5]
果茶:        [2.0, 0.1, 2.0, 1.0, 0.0, 1.5]
咖啡奶茶:    [1.0, 1.5, 0.0, 0.3, 1.5, 1.0]
冰沙:        [1.0, 1.0, 1.0, 0.5, 0.0, 4.0]
特调:        [1.0, 1.0, 1.0, 1.0, 0.5, 1.5]
```

### B.2 `context_deltas.yaml`

```yaml
season:
  summer: {tea: 0.0, milk: -0.3, fruit: +0.2, water: 0.0, coffee: 0.0, ice: +0.8}
  winter: {tea: 0.0, milk: +0.5, fruit: -0.2, water: 0.0, coffee: 0.0, ice: -0.6}
target_age:
  youth:  {tea: -0.3, milk: -0.3, fruit: +0.5, water: 0.0, coffee: 0.0, ice: +0.2}
  mature: {tea: +0.5, milk: +0.5, fruit: -0.3, water: 0.0, coffee: 0.0, ice: 0.0}
health_strict:
  true:   {tea: +0.5, milk: -0.2, fruit: +0.3, water: +0.3, coffee: 0.0, ice: 0.0}
```

### B.3 `topping_compatibility.yaml`(片段)

```yaml
# 互补结构 score=1.0, 冲突 score=-1.0, 中性 0.0
- {pair: [topping_brown_pearl, topping_taro_ball], score: 0.9}    # 都是嚼劲, 但口感互补
- {pair: [topping_brown_pearl, topping_pop_mango], score: 0.8}    # 嚼劲 + 爆破
- {pair: [topping_grass_jelly, topping_red_bean], score: 0.7}     # 经典烧仙草
- {pair: [topping_pop_mango, topping_pop_strawberry], score: -0.5}  # 同质化
```

---

## 附录 C. 完整代码骨架获取方式

由于代码量较大,完整可运行骨架的 git repo 模板存放在(占位):

```
git clone https://github.com/<your-org>/beverage_ai_template.git
cd beverage_ai_template
uv venv && uv sync
pytest    # 验证骨架可跑通
```

模板包含:
- 上述所有文件的空实现(`pass` 或 `NotImplementedError`)
- 已可运行的 `pyproject.toml` + `.github/workflows/ci.yml`
- 单元测试的 fixture 示例
- 一个 minimal Gradio demo(返回 dummy 数据)

填代码即可,无需从零搭脚手架。

---

**变更记录**

| 版本 | 日期 | 作者 | 变更内容 |
|---|---|---|---|
| v1.0 | 2026-05-25 | — | 首版实现方案,对应技术方案书 v1.4 |
