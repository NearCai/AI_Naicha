# Panel Analysis — Session `s_demo_w1`

_Generated 2026-05-25T03:13:49.443829+00:00_

## Session overview
- Total ratings: 1,050
- Panelists: 35
- Recipes: 21
- Dimensions: 喜爱度, 奶香, 甜度, 苦度, 茶香
- Groups: {'A': 7, 'B': 7, 'C': 7}

## Per-dimension summary
| Dim | n_obs | Mean | Std |
|---|---|---|---|
| 甜度 | 210 | 4.27 | 0.92 |
| 苦度 | 210 | 2.43 | 0.99 |
| 茶香 | 210 | 2.70 | 0.85 |
| 奶香 | 210 | 1.99 | 1.07 |
| 喜爱度 | 210 | 3.12 | 0.79 |

## Wilcoxon paired tests (A vs B, one-sided A > B)
| Dim | n | mean_A | mean_B | Δ | p | sig? | Cohen's d | r |
|---|---|---|---|---|---|---|---|---|
| 甜度 | 29 | 4.54 | 3.84 | +0.70 | 0.0003 | ✅ | +0.80 | +0.75 |
| 苦度 | 29 | 2.43 | 2.33 | +0.10 | 0.2836 | ❌ | +0.15 | +0.14 |
| 茶香 | 29 | 2.80 | 2.76 | +0.04 | 0.4330 | ❌ | +0.05 | +0.04 |
| 奶香 | 29 | 2.34 | 1.66 | +0.68 | 0.0013 | ✅ | +0.65 | +0.65 |
| 喜爱度 | 29 | 3.48 | 3.16 | +0.32 | 0.0250 | ❌ | +0.37 | +0.46 |

## TOST equivalence (A ≈ C)
- **甜度**: Δ=+0.12 (bound ±0.5), p_low=0.000, p_high=0.006 → ✅ equivalent
- **苦度**: Δ=+0.13 (bound ±0.5), p_low=0.002, p_high=0.039 → ✅ equivalent
- **茶香**: Δ=+0.26 (bound ±0.5), p_low=0.000, p_high=0.022 → ✅ equivalent
- **奶香**: Δ=+0.54 (bound ±0.5), p_low=0.000, p_high=0.582 → ❌ not equivalent
- **喜爱度**: Δ=+0.62 (bound ±0.5), p_low=0.000, p_high=0.758 → ❌ not equivalent

## Friedman test (across 5 sensory dims)
- χ²(4) = 118.50, p=0.0000 vs Bonferroni α=0.0083 → ✅

## Mixed-effects model (`score ~ group + (1|panelist)`)
- **甜度**: log-lik=-273.3, panelist var=0.000, fixed effects: {'group[T.B]': -0.614285715819766, 'group[T.C]': -0.07142857189349884}, p: {'group[T.B]': 3.798855463448347e-05, 'group[T.C]': 0.6306088088710271}
- **苦度**: log-lik=-297.4, panelist var=0.055, fixed effects: {'group[T.B]': -0.0853359380713634, 'group[T.C]': -0.0244564619125495}, p: {'group[T.B]': 0.6067998543567378, 'group[T.C]': 0.8842890770053408}
- **茶香**: log-lik=-253.9, panelist var=0.191, fixed effects: {'group[T.B]': -0.11111801862997354, 'group[T.C]': -0.22653617623479888}, p: {'group[T.B]': 0.39296212966456956, 'group[T.C]': 0.08039358904316575}
- **奶香**: log-lik=-306.5, panelist var=0.046, fixed effects: {'group[T.B]': -0.6788379583040045, 'group[T.C]': -0.543565786441689}, p: {'group[T.B]': 9.317423335853031e-05, 'group[T.C]': 0.0017819764377903992}
- **喜爱度**: log-lik=-240.8, panelist var=0.127, fixed effects: {'group[T.B]': -0.2561022378666346, 'group[T.C]': -0.4693256545249552}, p: {'group[T.B]': 0.037432500335396655, 'group[T.C]': 0.00013614254370673424}

## Power analysis
- For d=0.5, α=0.05, power=0.8: **n ≥ 25** (paired t/Wilcoxon)