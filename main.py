import re
import pandas as pd

pattern = re.compile(
    r"depth = (\d+).*?n_active = (\d+).*?t_hist = ([\d.e+-]+).*?t_scan = ([\d.e+-]+).*?t_write = ([\d.e+-]+).*?t_find = ([\d.e+-]+)",
    re.S
)

data = []
with open("out.txt") as f:
    text = f.read()
    for m in pattern.finditer(text):
        data.append({
            "depth": int(m.group(1)),
            "n_active": int(m.group(2)),
            "t_hist": float(m.group(3)),
            "t_scan": float(m.group(4)),
            "t_write": float(m.group(5)),
            "t_find": float(m.group(6)),
        })

df = pd.DataFrame(data)

# concise totals and shares
totals = df[["t_hist", "t_scan", "t_write", "t_find"]].sum()
T = float(totals.sum()) if len(totals) else 0.0
if T > 0:
    print(
        "totals  "
        f"t_hist={totals['t_hist']:.3f}s ({100*totals['t_hist']/T:.1f}%)  "
        f"t_scan={totals['t_scan']:.3f}s ({100*totals['t_scan']/T:.1f}%)  "
        f"t_write={totals['t_write']:.3f}s ({100*totals['t_write']/T:.1f}%)  "
        f"t_find={totals['t_find']:.3f}s ({100*totals['t_find']/T:.1f}%)  "
        f"total={T:.3f}s"
    )

# top depths by total time
if not df.empty:
    per_depth = df.groupby("depth")[["t_hist", "t_scan", "t_write", "t_find"]].sum()
    per_depth["total"] = per_depth.sum(axis=1)
    hits = df.groupby("depth").size()
    top = per_depth["total"].sort_values(ascending=False).head(10)
    print("\nTop depths by total time:")
    for d, s in top.items():
        h = int(hits.get(d, 0))
        avg = s / h if h else 0.0
        print(f"  depth={d} total={s:.4f}s avg={avg:.6f}s hits={h}")

# keep original detailed stats (optional)
if not df.empty:
    print("\nDetailed stats:")
    print(df.describe())
    print("Longest operation:", df.max())

