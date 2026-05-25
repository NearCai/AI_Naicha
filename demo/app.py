"""Gradio web demo — v1 实现方案 §6.13.

Run with:
    pip install -e ".[demo]"
    python demo/app.py
"""
from __future__ import annotations

import json

try:
    import gradio as gr
except ImportError as e:
    raise SystemExit(
        "gradio not installed. Run: pip install -e '.[demo]'"
    ) from e

from beverage_ai.ingredients.vocab import load_default_vocab
from beverage_ai.pipeline.end_to_end import run_pipeline

_VOCAB = load_default_vocab()


def gradio_handler(request_text: str, top_k: int, kappa: float):
    if not request_text or not request_text.strip():
        return "请输入需求描述", "", ""
    result = run_pipeline(
        user_request=request_text,
        top_k=int(top_k),
        n_candidates=150,
        kappa=float(kappa),
        seed=42,
    )

    # Spec block
    spec_md = "### 解析后的需求\n```json\n" + json.dumps(
        result.spec, ensure_ascii=False, indent=2
    ) + "\n```"

    # Stats block
    stats_md = (
        f"### 统计\n"
        f"- 生成候选: **{result.n_generated}**\n"
        f"- 合规候选: **{result.n_feasible}**\n"
        f"- Pareto 前沿: **{result.n_pareto}**\n"
        f"- 用时: **{result.elapsed_sec:.2f}s**\n"
    )

    # Recipe cards
    cards = []
    for i, c in enumerate(result.top_recipes, start=1):
        r = c["recipe"]
        means = c["means"]
        sigmas = c["sigmas"]
        nut = c["nutrition"]
        ing_lines = []
        for ing_id, mass in r["ingredients"].items():
            name = _VOCAB.get(ing_id).name_zh if ing_id in _VOCAB else ing_id
            ing_lines.append(f"- {name}: **{mass}g**")
        card = (
            f"### 候选 #{i}\n"
            f"**{r['style']} · {r['cup_volume_ml']}ml · {r['sugar_level']}**\n\n"
            f"| 指标 | 值 |\n|---|---|\n"
            f"| 喜爱度 | {means['preference']:.2f} ± {sigmas['preference']:.2f} (1–5) |\n"
            f"| 销量分 | {means['sales_proxy']:.1f} ± {sigmas['sales_proxy']:.1f} |\n"
            f"| 复购分 | {means['repurchase']:.3f} (0–1) |\n"
            f"| 成本 | ¥{means['cost_cny']:.2f} |\n"
            f"| 含糖 | {nut['sugar_g']:.1f}g |\n"
            f"| 热量 | {nut['energy_kcal']:.0f}kcal |\n"
            f"| 咖啡因 | {nut['caffeine_mg']:.0f}mg |\n"
            f"| 致敏原 | {', '.join(nut['allergens']) if nut['allergens'] else '无'} |\n\n"
            f"**原料明细**:\n" + "\n".join(ing_lines) + "\n"
        )
        cards.append(card)
    cards_md = "\n\n---\n\n".join(cards) if cards else "(无候选)"

    return spec_md, stats_md, cards_md


demo = gr.Interface(
    fn=gradio_handler,
    inputs=[
        gr.Textbox(
            label="需求描述",
            lines=3,
            placeholder="例: 想为夏季年轻女性开发一款主打健康轻负担的茶饮新品, 定价 18-22 元",
        ),
        gr.Slider(1, 10, value=5, step=1, label="Top-K"),
        gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="κ (LCB 保守度)"),
    ],
    outputs=[
        gr.Markdown(label="需求解析"),
        gr.Markdown(label="统计"),
        gr.Markdown(label="Top-K 候选配方"),
    ],
    title="🧋 茶饮研发闭环 AI 系统 v1",
    description=(
        "输入自然语言需求,系统会:解析 → 生成 200 候选 → "
        "多目标优化 (LCB) → MMR 选 Top-K。详见技术方案书与实现方案文档。"
    ),
    examples=[
        ["夏季年轻女性低糖轻负担, 定价 18-22 元", 5, 1.0],
        ["冬天上班族暖身高蛋白, 接受 30 元", 5, 1.0],
        ["无糖控碳水, 只要茶味鲜明", 5, 1.0],
    ],
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
