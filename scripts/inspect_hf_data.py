"""Inspect what was actually ingested into the hf_waimai shard."""


from beverage_ai.scrapers.store import RawReviewStore

store = RawReviewStore("data/reviews/raw")
df = store.read("hf_waimai")
print(f"Total rows: {len(df)}\n")

print("Rating distribution (binary label → 1.5=neg / 4.5=pos):")
print(df["rating"].value_counts().to_string())
print()

text_lens = df["text"].str.len()
print(f"Text length: mean={text_lens.mean():.0f}, median={text_lens.median():.0f}, "
      f"min={text_lens.min()}, max={text_lens.max()}")
print()

print("--- First 5 rows ---")
for _, row in df.head(5).iterrows():
    print(f"  [r={row['rating']}] {row['text'][:80]}")
print()

print("--- Last 5 rows ---")
for _, row in df.tail(5).iterrows():
    print(f"  [r={row['rating']}] {row['text'][:80]}")
print()

print("--- Tea-related keyword hits (奶茶 | 茶饮 | 拿铁 | 咖啡 | 茶) ---")
tea_pattern = "奶茶|茶饮|拿铁|咖啡|茶"
mask = df["text"].str.contains(tea_pattern, regex=True, na=False)
print(f"  matched: {mask.sum()} / {len(df)} ({100 * mask.mean():.1f}%)")
print()

print("--- Sample tea-related reviews ---")
for _, row in df[mask].head(8).iterrows():
    print(f"  [r={row['rating']}] {row['text'][:120]}")
