# `data/products/` — 茶饮品牌与 SKU 销量数据

> 对应 [技术方案书 §3.3.2](../../../茶饮研发闭环AI系统_技术方案书.md) 销量端 +
> [v1 实现方案 §5](../../../茶饮研发闭环AI系统_v1实现方案.md) 数据资产清单。
>
> 这是 `sku_features_v1.parquet` 的 YAML 前身,留 YAML 是为了可读 + 易追溯来源。
> 等需要喂给 LightGBM 时再写脚本转 parquet。

## 文件清单

| 文件 | 说明 |
|---|---|
| `sku_features_v1.yaml` | 主数据文件:10 个头部品牌 + 33 个 SKU + 行业级数据 |
| `README.md` | 本文档:数据来源、可信度、扩展方法 |

## 这是什么 / 不是什么

✅ **是**:
- 公开渠道(招股书、年报、官方公关稿、行业报告)能验证的品牌 / SKU 元数据
- 每条记录都有 `data_sources` 字段引用具体出处
- 适合做销量预测模型的**baseline 特征**(品牌、价格、品类、上市年、招牌标志)
- 适合做"配方相对市场的可比较性"分析(类似 SKU 的价格区间 / 风味偏向)

❌ **不是**:
- 各品牌的精确日均 / 月均杯量(品牌核心商密,公开渠道拿不到)
- 各店实际营业额 / 利润(同上)
- 实时数据(数据陈年度 2024H2 - 2025H2,建议每季度刷新)
- 完整的全品牌全 SKU 覆盖(只覆盖头部 10 个品牌的代表性 SKU)

## 数据来源可信度分级

每条 `data_sources[].type` 标注以下等级:

| Type | 可信度 | 来源 | 例子 |
|---|---|---|---|
| `ipo_prospectus` | **HIGH** | 上市公司招股书(港交所/纳斯达克监管披露) | 蜜雪冰城 2024 招股书, 古茗 2024 招股书 |
| `annual_report` | **HIGH** | 年报 / 中期报告 | 蜜雪 2025 中期, 奈雪 2024 年报 |
| `press_release` | HIGH (限对自己的声明) | 品牌官方公关稿 | 霸王茶姬伯牙绝弦 12.5 亿杯, 书亦草莓奶云麻薯 3800 万杯 |
| `industry_report` | MEDIUM | 艾瑞 / CBNData / 36 氪 / 华泰研究 | 解码 40 万家门店, KERRY 风味图谱 |
| `news_media` | LOW-MEDIUM | 媒体转载(常引用其他源) | 36 氪 / 新浪财经 / 腾讯新闻 |
| `estimated` | LOW | 衍生计算 / 估算 | 喜茶 / 奈雪 / 茶颜门店数(非上市) |

**审计建议**:训销量模型前,把所有 `estimated` 字段单独打 flag,验收时单独评估它们的影响。

## 字段说明

### 行业级 (`industry`)
- `market_size_cny_2024`: 2024 全国市场规模(元)
- `top_flavor_3yr` / `fastest_growing_flavor_1yr`: 来自 KERRY × 饿了么 2025 风味图谱
- `total_stores_china_2025_09`: 全国所有现制茶饮门店

### 品牌级 (`brands[]`)
- `tier`: budget / mid / premium / regional — 用作销量模型的分层特征
- `stores_total` + `stores_year`: 门店数 + 数据年份
- `revenue_cny_*`: 营收(若上市)
- `total_cups_*`: 出杯量(若披露)
- `atv_cny_*`: 客单价(Average Transaction Value)
- `category_mix_by_cups`: 品类杯量占比(如古茗 果茶:奶茶:咖啡 = 41:47:12)
- `positioning`: 品牌定位 tag 列表

### SKU 级 (`skus[]`)
- `style`: 对齐 `RecipeGenerator` 的 6 大风格分类 — 可以直接 join 配方生成
- `signature`: 是否品牌招牌
- `trending_2025`: 是否当下热销
- `proxies.*`: 公开杯量 / 销售额 / 占比等代理指标

## 已知缺口(`gaps` 字段也记录)

| 缺口 | 影响 | 补救路径 |
|---|---|---|
| 蜜雪 Top 5 SKU 占总杯量 41.2% 但未拆分到 SKU | 蜜雪具体 SKU 占比要估算 | 招股书追加披露 / 长期监测季报 |
| 古茗 / 沪上阿姨只给品类合计 | 同上 | 同上 |
| 喜茶 / 奈雪 / 茶颜非上市 | 详细数据靠公关稿 + 媒体爆料 | 持续监测品牌官方公众号 / 微博 |
| 复购率大多不披露 | 复购模型 v1 没真值校准 | 仅古茗 53% 是少数公开数据 |
| 单店日均 / 地域分布 / 季节性 | 销量模型几何特征缺失 | 大众点评单店数据(注意 ToS,见 `docs/SCRAPING_NOTICE.md`) |
| 新品衰减曲线 | 无法预测新品 12 / 24 周后衰减 | 长期追踪同一 SKU 多个时点 |

## 怎么扩展这个文件

### 加一个新 SKU(品牌已存在)

```yaml
# 在 skus: 列表里加一条
- sku_id: <brand_id>_<short_name>       # 如 mixue_summer_peach
  brand_id: mixue
  sku_name_zh: 夏日蜜桃
  style: 果茶                            # 必须用 Recipe.style 的 6 种之一
  price_cny: 8.0
  launch_year: 2025
  signature: false                       # 默认 false
  trending_2025: true
  proxies:
    # 任何能找到的公开数据, 命名为 *_disclosed 标明出处
    annual_cups_disclosed: null
    weekly_cups_disclosed: null
    other_proxy: null
  data_sources:
    - cite: "来源标题"
      type: news_media                   # 选一个 type
      year: 2025
      url: "https://..."
  notes_zh: "可选, 描述这条 SKU 的特殊背景"
```

### 加一个新品牌

```yaml
# 在 brands: 列表加一条
- brand_id: lelecha
  brand_name_zh: 乐乐茶
  brand_name_en: Lelecha
  tier: premium
  avg_price_cny: 19.0
  stores_total: null                     # 不知道就 null, 不要瞎写
  stores_year: null
  listed: false
  positioning: [premium, bakery_tea]
  notes_zh: "茶饮 + 烘焙复合业态"
  data_sources: []                        # 没 source 就空列表, 不要编
```

### 加新的 industry-level 字段

```yaml
industry:
  # 已有字段...
  new_field_2025:
    value: ...
    source: "..."
```

## 怎么用这个文件训销量预测器

```python
import yaml
import pandas as pd

with open("data/products/sku_features_v1.yaml", encoding="utf-8") as f:
    raw = yaml.safe_load(f)

# 拍平成 DataFrame
rows = []
brand_map = {b["brand_id"]: b for b in raw["brands"]}
for sku in raw["skus"]:
    b = brand_map[sku["brand_id"]]
    rows.append({
        "sku_id": sku["sku_id"],
        "brand": sku["brand_id"],
        "tier": b["tier"],
        "style": sku["style"],
        "price_cny": sku.get("price_cny"),
        "signature": sku.get("signature", False),
        "trending_2025": sku.get("trending_2025", False),
        "launch_year": sku.get("launch_year"),
        "brand_stores": b.get("stores_total"),
        "brand_atv": b.get("avg_price_cny"),
        "annual_cups_disclosed": sku.get("proxies", {}).get("annual_cups_disclosed"),
    })

df = pd.DataFrame(rows)
df.to_parquet("data/products/sku_features_v1.parquet", index=False)
# → 送进 SalesPredictorLGB.fit(baseline_df=..., recipe_embed_df=..., y=...)
```

详见 [技术方案书 §3.3.2 销量预测器 + R10/R11 修复](../../../茶饮研发闭环AI系统_技术方案书.md):
- baseline 必须包含品牌固定效应(`brand` 列)
- K 折交叉拟合避免 baseline 在自己样本上过拟合
- 已下架 SKU 需要单独采集(本文件目前不含失败品 → 幸存者偏差)

## 数据 vintage 与刷新

| 字段类别 | 当前 vintage | 建议刷新频率 |
|---|---|---|
| 品牌门店数 | 2024 H2 - 2025 H2 | 季度 |
| 品牌财务(上市公司) | 2024-2025 中期 | 财报发布日 +1 周 |
| SKU 杯量披露 | 2023-2025 | 招股书 / 年报 / 公关稿 +1 周 |
| 行业风味趋势 | 2024-2025 | 半年 |

每次刷新建议:复制 `sku_features_v1.yaml` → `sku_features_v2_<YYYY-MM>.yaml`,在新文件里改,保留旧版本对比。

## 法律 / 伦理

- 本文件所有数据均来自公开渠道(上市公司监管披露、品牌官方公关稿、媒体公开报道)
- 不含任何爬虫数据 / 用户个人信息 / 商业机密
- 仅用于研究 / 教学 / 产品研发参考,不构成投资建议
- 如品牌方认为本文件有错漏,欢迎提 issue 校正
