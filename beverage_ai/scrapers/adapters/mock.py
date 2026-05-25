"""Synthetic review generator.

Produces realistic-looking Chinese reviews with embedded sensory cues
(甜度/苦度/茶香/奶香/喜爱度) and customization mentions (糖度档位、配料).
Used for testing the LLM aspect extractor + downstream GNN training prototype.
"""
from __future__ import annotations

import random
from typing import Iterable

from ..base import ReviewRecord, make_review_id, normalize_text


_BRANDS_SKUS = [
    ("喜茶", ["多肉葡萄", "芝士绿妍", "桂花乌龙", "鸭屎香拿铁", "酪酪桃桃"]),
    ("奈雪", ["霸气芝士葡萄", "宝藏茶", "金色山脉烤奶", "鹰嘴豆拿铁"]),
    ("茶颜悦色", ["幽兰拿铁", "声声乌龙", "桂花弄"]),
    ("蜜雪冰城", ["珍珠奶茶", "柠檬水", "棒打鲜橙"]),
    ("书亦烧仙草", ["招牌烧仙草", "芋圆奶茶"]),
    ("一点点", ["波霸奶茶", "四季春", "阿华田"]),
    ("古茗", ["超A芝士葡萄", "茉莉鲜奶茶"]),
    ("CoCo都可", ["珍珠奶茶", "百香果双响炮"]),
]

_OPENERS = [
    "今天点了{brand}的{sku},",
    "种草已久的{brand}{sku},终于尝到了,",
    "下午茶买了一杯{brand}的{sku},",
    "{brand}的{sku}回购了第三次,",
    "{brand}新品{sku}打卡,",
    "排了二十分钟买的{brand}{sku},",
]

_CUSTOMIZATIONS = [
    "三分糖去冰", "五分糖少冰", "七分糖正常冰", "全糖正常冰", "无糖少冰", "三分糖加波霸",
    "五分糖加芋圆", "无糖加椰果", "七分糖加红豆", "全糖加双倍珍珠",
]

# Quality descriptors with embedded sensory cues
_QUALITY_GOOD = [
    "茶味很浓厚,奶香也足,整体非常顺滑,推荐",
    "桂花香气扑鼻,微微回甘,完全没有齁甜",
    "奶盖咸甜平衡,底茶清爽,可以无脑回购",
    "果香清新,酸甜适中,夏天必备",
    "层次很丰富,先是茶香后是奶味,确实好喝",
    "厚乳很惊艳,顺滑不腻,茶底也压得住",
]

_QUALITY_BAD = [
    "甜得发齁,根本喝不出茶味,不推荐",
    "茶味太涩,后味发苦,真心一般",
    "奶味假假的,像是植脂末做的,没什么记忆点",
    "果肉很少,基本只有糖水味,踩雷",
    "整体偏寡淡,层次也不够,不会回购",
    "凉了之后特别腻,香精味很重",
]

_QUALITY_MIXED = [
    "茶香不错但偏甜,可以接受",
    "奶味顺滑,但果茶部分酸味突出有点冲突",
    "配料挺足的,只是茶底有点淡了",
    "颜值在线,味道一般般,见仁见智吧",
    "刚喝是惊艳,放一会儿就普通了",
]


class MockScraper:
    """Generates synthetic reviews on demand. Always available, no deps."""

    source_name = "mock"

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def scrape(
        self,
        *,
        keywords: list[str] | None = None,
        brand: str | None = None,
        max_records: int | None = None,
    ) -> Iterable[ReviewRecord]:
        n = max_records or 50
        for _ in range(n):
            yield self._one()

    def _one(self) -> ReviewRecord:
        brand, skus = self._rng.choice(_BRANDS_SKUS)
        sku = self._rng.choice(skus)
        customization = self._rng.choice(_CUSTOMIZATIONS)

        # Bias quality distribution: 50% good, 25% mixed, 25% bad
        roll = self._rng.random()
        if roll < 0.5:
            quality = self._rng.choice(_QUALITY_GOOD)
            rating = self._rng.uniform(4.0, 5.0)
        elif roll < 0.75:
            quality = self._rng.choice(_QUALITY_MIXED)
            rating = self._rng.uniform(2.8, 3.8)
        else:
            quality = self._rng.choice(_QUALITY_BAD)
            rating = self._rng.uniform(1.0, 2.5)

        opener = self._rng.choice(_OPENERS).format(brand=brand, sku=sku)
        text = normalize_text(f"{opener}{customization}。{quality}。")

        return ReviewRecord(
            review_id=make_review_id(self.source_name, brand, text),
            source=self.source_name,
            brand=brand,
            sku=sku,
            text=text,
            customization_raw=customization,
            rating=round(rating, 1),
            metadata={"synthetic": True, "quality_roll": roll},
        )
