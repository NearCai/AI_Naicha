# AutoDL GPU 训练操作手册

完整的"本地 ↔ AutoDL"工作流,让 GNN 训练在 RTX 3090/4090 上跑而不是 CPU。

## TL;DR

```bash
# 一次性:开 AutoDL 实例 → 拿 ssh url(类似 root@region-X.seetacloud.com:25731)

# 在本地(Windows: Git Bash 或 WSL)
bash scripts/upload_to_autodl.sh root@region-X.seetacloud.com:25731

# 在 AutoDL 上
ssh -p 25731 root@region-X.seetacloud.com
cd /root/beverage_ai
bash scripts/setup_autodl.sh                       # 5-8 分钟
bash scripts/run_training_autodl.sh                # mock extractor, 默认 50 epoch

# 训完拉回本地
bash scripts/download_from_autodl.sh root@region-X.seetacloud.com:25731
python scripts/report_training.py
```

---

## 1. 选实例

进 [autodl.com](https://www.autodl.com) → 算力市场:

| GPU | 显存 | 价格 | 适合 |
|---|---|---|---|
| RTX 3060 | 12 GB | ¥0.8-1.5/h | 本项目当前规模(<5K graphs)足够 |
| **RTX 3090** | **24 GB** | **¥1.5-2.5/h** | **推荐**:GNN 训练 + 真 Claude aspect 并行 |
| RTX 4090 | 24 GB | ¥2.5-3.5/h | 更快,与 3090 性能比 ~1.5x |
| A40 | 48 GB | ¥2-4/h | 大模型微调可用 |
| A100 | 40/80 GB | ¥4-7/h | 本项目用不上,浪费钱 |

**镜像选择**:`PyTorch 2.x` 或 `Miniconda + CUDA 12.x`(任选,我们的脚本会重装合适版本)。

**存储**:默认 `/root/autodl-tmp/` 是 NVMe scratch(快但只在实例运行期保留),`/root/` 是系统盘 50GB。建议把项目放在 `/root/beverage_ai/`,模型与缓存放 `/root/autodl-tmp/models/` 避免占系统盘。

---

## 2. 本地准备(Windows + Git Bash 或 WSL)

```bash
# 检查 ssh / rsync 都装了
which ssh rsync
```

Windows 没 rsync 的话,装 Git for Windows 内置的 `MSYS rsync`,或用 WSL。

---

## 3. 上传代码 + 数据

从你本地项目根目录:

```bash
cd D:/2026new/paper/beverage_ai      # PowerShell: Set-Location D:\2026new\paper\beverage_ai

# Git Bash:
bash scripts/upload_to_autodl.sh root@region-X.seetacloud.com:25731
```

会同步:

| 内容 | 大小 | 必要性 |
|---|---|---|
| 代码 `beverage_ai/` `scripts/` `tests/` `pyproject.toml` | ~500KB | 必需 |
| `data/ingredients/` (vocab 等) | ~100KB | 必需 |
| `data/priors/` (Dirichlet α) | ~5KB | 必需 |
| `data/recipes/reference_recipes_v1.yaml` | ~30KB | 必需 |
| `data/reviews/raw/` (~15K 已抓评论) | ~10MB | 必需 |
| `data/reviews/aspects_cache.duckdb` | 1-50MB | 可选(没有就在 AutoDL 重新抽) |

**不会同步**:`.venv/`, `__pycache__/`, `models/*.pt`(目标会重新训)

---

## 4. SSH 上去 setup

```bash
ssh -p 25731 root@region-X.seetacloud.com
cd /root/beverage_ai
bash scripts/setup_autodl.sh
```

脚本做的事:

1. `export HF_ENDPOINT=https://hf-mirror.com`(国内访问加速,持久写入 `~/.bashrc`)
2. 检测 CUDA 版本 → 选对应 torch wheel(cu121 / cu124 / 等)
3. 建 `.venv` + 装 `beverage_ai[ml,hf,llm]`
4. 打印 verification:`torch.cuda.is_available()` 应该是 `True`

预期输出末尾:
```
torch: 2.x.y+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
GPU mem: 24.0 GB
torch_geometric: 2.x.y
beverage_ai: 0.1.0
```

---

## 5. 跑训练

### 默认(mock extractor,免费,~3 分钟)

```bash
bash scripts/run_training_autodl.sh
```

等价于:
```bash
python scripts/train_sensory_gnn_stage1.py \
    --epochs 50 --extractor mock --device auto --amp \
    --patience 10 --tag autodl
```

### 用真 Claude(更高质量标签,~$30 跑 15K)

```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
EXTRACTOR=claude COST=30 bash scripts/run_training_autodl.sh
```

### 配 WandB 远程监控

```bash
pip install wandb
wandb login   # 粘 token
WANDB_PROJECT=beverage_ai bash scripts/run_training_autodl.sh
```

然后在本地浏览器开 `https://wandb.ai/<user>/beverage_ai` 看曲线。

### 性能预期(15K 评论,~940 graphs,50 epoch)

| 设备 | 单 epoch | 总时长 | 备注 |
|---|---|---|---|
| 本地 CPU (Intel i7) | ~2.3s | 115s | 已验证,跑通 |
| RTX 3060 | ~0.4s | 20s | 5-6× 加速 |
| **RTX 3090** | **~0.2s** | **10s** | **10-12× 加速,推荐** |
| RTX 4090 | ~0.15s | 8s | 与 3090 比性价比一般 |

注:本项目当前规模太小,GPU 加速效益主要来自:
1. AMP 混合精度
2. 大 batch size(GPU 默认 128,CPU 默认 32)
3. DataLoader 多进程(GPU 默认 4 worker)

**真正的 GPU 必要性**会在以下场景显现:
- 真 Claude 抽 50K aspects(~30 分钟 GPU 网络 IO)
- Stage 2 微调(增大 hidden + heads)
- 数据扩到 100K+ graphs(典型生产规模)

---

## 6. 拉模型回本地

训完后,从本地:
```bash
bash scripts/download_from_autodl.sh root@region-X.seetacloud.com:25731
python scripts/report_training.py
```

会拉回:
- `models/sensory_gnn_stage1_prototype.pt` (final state)
- `models/sensory_gnn_stage1_best.pt` (best val checkpoint)
- `models/sensory_gnn_stage1_log.json` (训练曲线)
- `data/reviews/aspects_cache.duckdb` (如果在 AutoDL 上抽过 aspect)

---

## 7. 成本估算(50K 数据完整训练)

| 项目 | 时长 | AutoDL ¥ | API $ |
|---|---|---|---|
| 实例租用(RTX 3090) | 1-2 小时 | ¥3-5 | — |
| 真 Claude aspect 抽取 50K | 包含在上面时长 | — | ~$30 |
| 训练 100 epoch | <30 分钟 | 包含 | — |
| **总计** | **2 小时** | **~¥5** | **~$30** |

对比 CPU 本地:抽取 50K Claude aspects 要 4-6 小时(纯 IO 等待),训练要 30+ 分钟。AutoDL 节省的不是钱,是时间。

---

## 8. 常见问题

### Q: 实例关机后数据丢吗?
A: `/root/` 和 `/root/autodl-tmp/` 都在实例的"数据盘",**关机不丢**。释放(销毁)实例才丢。如果按小时计费,**记得用完关机不释放**,下次开机数据还在。

### Q: HuggingFace 拉不下来 / 超时
A: 已经设了 `HF_ENDPOINT=https://hf-mirror.com`。如果还失败:
```bash
echo $HF_ENDPOINT      # 验证
HF_HUB_DOWNLOAD_TIMEOUT=60 python scripts/ingest_hf_reviews.py --recipe scripts/hf_recipe.yaml
```

### Q: ANTHROPIC_API_KEY 在 AutoDL 上怎么 export 才持久?
A: 写进 `~/.bashrc`:
```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-xxx' >> ~/.bashrc
source ~/.bashrc
```

### Q: 训练中断 / SSH 断了怎么办
A: 用 `tmux` 或 `screen` 防断:
```bash
tmux new -s train
bash scripts/run_training_autodl.sh
# Ctrl-B 然后 D 分离
# 重新 SSH 后: tmux attach -t train
```

### Q: WandB 也访问不了
A: WandB 国内访问有时慢。可以用 tensorboard 替代:
```bash
pip install tensorboard
# 修改训练脚本里的 wandb 那段 → SummaryWriter
# 然后在 AutoDL 上启动 tensorboard --logdir runs --host 0.0.0.0 --port 6006
# 浏览器开 http://实例外网IP:6006/
```

### Q: 显存炸了 OOM
A: 减小 batch size:
```bash
python scripts/train_sensory_gnn_stage1.py --batch-size 32 --device cuda --amp ...
```

---

## 9. 进阶:多卡 / 多任务

本项目 GNN 极小(<1M 参数),不需要多卡。但如果要并行做多个超参实验:

```bash
# 不同的 cost ceiling
TAG=ceiling_5  COST=5  bash scripts/run_training_autodl.sh &
TAG=ceiling_30 COST=30 bash scripts/run_training_autodl.sh &
wait
```

注意单 GPU 会被两个 job 抢资源,batch 各设小一些。

---

## 10. 关机省钱

跑完后立即:

```bash
# AutoDL 网页 → 容器实例 → 关机 (不要"释放")
```

关机后不计算 GPU 费用,只收数据盘存储费(¥0.01-0.02/GB/天)。50GB 项目数据 = ¥0.5-1/天保留。
