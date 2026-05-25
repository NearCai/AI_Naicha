# 奶茶配方 Skill 库

这个目录把 `/Users/wangzixing/Desktop/黑客松/beverage_ai/data/recipes/reference_recipes_v1.yaml` 中的 110 条现有配方整理成 7 个可检索 Markdown 文档，供 AiNaiCha 项目后续生成配方时作为参考知识库。

分类指标：季节、价格带、成本上限、出杯时间、目标人群、甜度倾向、温度形态。

说明：原始 reference recipe 只包含风格、杯量、糖度、原料克重和名称；本目录中的季节、价格带、成本、出杯时间、目标人群、温度形态为基于原料、风格和克重的规则化推断，用于 AI 生成配方时检索参考，不应直接视为真实经营数据。

## 文档

- [季节](season.md)
- [价格带](price_band.md)
- [成本上限](cost_cap.md)
- [出杯时间](make_time.md)
- [目标人群](target_audience.md)
- [甜度倾向](sweetness.md)
- [温度形态](temperature.md)
