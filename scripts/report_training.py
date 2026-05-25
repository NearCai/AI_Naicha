"""Pretty-print the GNN Stage 1 training log."""
import json
import sys
from pathlib import Path

log_path = Path("models/sensory_gnn_stage1_log.json")
if not log_path.exists():
    print(f"ERROR: {log_path} not found", file=sys.stderr)
    sys.exit(1)

log = json.loads(log_path.read_text(encoding="utf-8"))

print("=" * 75)
print(f"GNN Stage 1 prototype training — {log['device']}")
print(f"Train graphs: {log['n_train']}   Val graphs: {log['n_val']}   Elapsed: {log['elapsed_sec']}s")
print("=" * 75)

dims = ["甜度", "苦度", "茶香", "奶香", "喜爱度"]
print(f"\n{'ep':>3}  {'train':>8}  {'val':>8}  ", end="")
for d in dims:
    print(f"{d:>6}", end="  ")
print()
print("-" * 75)

for e in log["epochs"]:
    rstr = ""
    for d in dims:
        v = e["val_pearson"].get(d)
        rstr += f"{v if v is not None else 'n/a':>6}  " if v is None else f"{v:>+6.3f}  "
    print(f"{e['epoch']:>3}  {e['train_loss']:>8.4f}  {e['val_loss']:>8.4f}  {rstr}")

print()
last = log["epochs"][-1]["val_pearson"]
print("Final validation Pearson r (per dim):")
for d in dims:
    v = last.get(d)
    if v is None:
        print(f"  {d}: n/a (insufficient val labels)")
    else:
        marker = " ← strong" if abs(v) > 0.5 else " ← weak" if abs(v) > 0.2 else ""
        print(f"  {d}: {v:+.3f}{marker}")
