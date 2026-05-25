# Path A Data Acquisition — Strategy & Operations Guide

> Per 技术方案书 §3.3.1 路径 A: collect 50,000–100,000 reviews for GNN Stage 1
> pretraining. None of the routes alone hits the target; this doc lays out
> how to combine them.

## Realistic yield per source

| # | Source | Yield (realistic) | Cost | Time | Legal risk |
|---|---|---|---|---|---|
| 1 | **HuggingFace datasets** | 20,000–40,000 (after keyword filter) | ¥0 | 1-2 hours setup + download | None |
| 2 | **Weibo Open API** | 1,000–5,000/day, capped ~30K/month free | ¥0 | 1-2 weeks elapsed | None (within ToS) |
| 3 | **Partner / brand collab** | depends on partner (0–100K) | personnel | weeks–months | None |
| 4 | **Manual collection** | 5–20/hour realistic | personnel | ~50 hours / 500 records | None |
| 5 | **Scraper** | unlimited in theory | server + risk | ongoing | **High** — see SCRAPING_NOTICE.md |

## Recommended combined plan to hit 50K

```
HuggingFace        25,000   (1 day)       baseline language transfer
Weibo API           5,000   (1 week)      domain-specific signal
Partner / mock     15,000   (variable)    high-quality if partnership exists
Manual                500   (~50 hours)   highest-quality calibration
Synthetic gap-fill  5,000+  (1 hour)      if all else short — mark clearly
-----------                                
Total              50,500
```

You **do not** need 50K of equal quality. GNN Stage 1 is pretraining with
noisy labels; the 500 manual + 5K Weibo serve as the higher-quality anchors,
while the 25K HF lets the model learn the *language* of sensory description.

## Workflow

```bash
# 1. See current state
python scripts/build_path_a_dataset.py

# 2. Bulk: HuggingFace (recommended first, biggest yield)
pip install -e '.[hf]'
python scripts/ingest_hf_reviews.py --recipe scripts/hf_recipe.yaml

# 3. Domain: Weibo (after registering at open.weibo.com)
export WEIBO_ACCESS_TOKEN=...
python scripts/ingest_weibo.py --keywords '奶茶,茶饮新品,鲜奶茶,鸭屎香' --max-records 2000 --shard weibo_w1

# 4. Partner export (if available)
python scripts/ingest_partner.py --path data/inbox/2026_q2.csv --shard partner_q2

# 5. Manual fill-in (high-quality calibration)
python scripts/collect_manual.py --shard manual_q3

# 6. After data: extract aspects (incremental, cached, $-aware)
beverage-ai aspects extract --self-consistency 3 --cost-ceiling-usd 50

# 7. Audit a sample
beverage-ai aspects audit --n 50

# 8. Final stats
beverage-ai scrape stats
beverage-ai aspects stats
```

## Per-source notes

### 1. HuggingFace datasets

Edit `scripts/hf_recipe.yaml` to pick datasets. Tested combinations:

- `XiangPan/waimai_10k` — most relevant (food delivery reviews; ~8K after 奶茶 keyword)
- `seamew/ChnSentiCorp` — general Chinese sentiment (~3K after keyword filter)

If keyword filtering returns too few, drop the `keywords` field (or set it
broader like `[茶, 咖啡, 饮料]`) and accept some off-topic data — GNN
Stage 1 can still learn language transfer from it.

**Important**: each dataset has its own license. Read it before redistributing.

### 2. Weibo Open API

Register at https://open.weibo.com:
1. Create developer account
2. Create application → get App Key + Secret
3. Complete OAuth2 web-flow to get user access token
4. `export WEIBO_ACCESS_TOKEN=...`

Free tier quota is restrictive (~150 requests/hour, ~50 statuses/request).
A full day of polite scraping yields 1–3K statuses. Be patient.

**Keywords that work well**:
- 品牌名:`喜茶`, `奈雪`, `蜜雪冰城`, `茶颜悦色`, `霸王茶姬`
- 品类:`奶茶`, `茶饮新品`, `鲜奶茶`, `果茶`, `厚乳茶`
- 趋势词:`鸭屎香`, `油柑`, `桂花乌龙`, `轻乳茶`

### 3. Partner / brand collaboration

The only **scalable, zero-risk** route. Approach:
1. Contact brand R&D or marketing
2. Pitch the academic / research angle
3. Sign data-sharing agreement (specify: research use only, no redistribution)
4. Receive export in any tabular format
5. Run `ingest_partner.py`

If you have **no** brand contact, this is a long-tail option. Combine with HF
+ Weibo for v1 and pursue partnerships for v2.

### 4. Manual collection

Use for the last 100–500 high-quality entries. Workflow:
1. Open small-red-book / Weibo / community forum in browser
2. Run `python scripts/collect_manual.py --shard manual_xxx`
3. For each post you find interesting:
   - Copy/paste the text
   - Fill in brand, SKU, customization, rating
   - Optionally paste source URL
4. Tool auto-saves every 5 entries; Ctrl-C is safe

5-20 entries/hour realistic. Don't try to hit 50K this way.

### 5. Scraper (NOT recommended)

See `docs/SCRAPING_NOTICE.md` for the full legal/ToS analysis. The
`DianpingScraper` and `XiaohongshuScraper` are skeletons left as
documentation; running them at scale is your own risk.

## When to declare Path A "done"

```
beverage-ai scrape stats
```

Goals:
- Total ≥ 30,000 (minimum for GNN Stage 1)
- ≥ 3 sources represented (data-source diversity)
- Manual fraction ≥ 1% (calibration anchor)
- LLM aspect extraction completed (`beverage-ai aspects stats` shows full coverage)
- Audit pass rate ≥ 80% (`beverage-ai aspects audit --n 50` and check by eye)

If those are green, proceed to Stage 1 training.
