import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# 1. Load the segment metrics
df = pd.read_csv(os.path.join(HERE, "step3_heuristic_baseline_metrics.csv"))

# 2. Quadrant thresholds
RISK_SPLIT = 7.6
VALUE_ZERO = 0
VALUE_HIGH = 9550

# 3. Assign each segment to a quadrant and pick its colour
def quadrant(risk, value):
    if value < VALUE_ZERO:  return "exit",     "#c0392b"   # red
    if risk  > RISK_SPLIT:  return "contain",  "#e67e22"   # orange
    if value > VALUE_HIGH:  return "grow",     "#27ae60"   # green
    return                         "maintain", "#3498db"   # blue

df[["posture", "color"]] = df.apply(
    lambda r: pd.Series(quadrant(r.default_rate_pct, r.mean_value_proxy_inr)),
    axis=1
)

# 4. Bubble sizes
sizes = df.n_loans / df.n_loans.max() * 4000 + 300

# 5. Build the chart
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df.default_rate_pct, df.mean_value_proxy_inr,
           s=sizes, c=df.color, alpha=0.75,
           edgecolors="black", linewidths=1.2)

# 6. Quadrant dividers
ax.axvline(RISK_SPLIT, color="gray", lw=1, alpha=0.6)
ax.axhline(VALUE_ZERO, color="gray", lw=1, alpha=0.6)
ax.axhline(VALUE_HIGH, color="gray", lw=1, alpha=0.4, ls="--")

# 7. Bubble labels (segment name + % of book)
for _, r in df.iterrows():
    ax.annotate(f"{r.segment}\n({r.pct_of_book}% book)",
                xy=(r.default_rate_pct, r.mean_value_proxy_inr),
                xytext=(10, 8), textcoords="offset points", fontsize=9)

# 8. Quadrant corner labels
ax.text(0.5, 41000, "GROW",     fontsize=16, fontweight="bold", color="#27ae60", alpha=0.35)
ax.text(0.5,  2000, "MAINTAIN", fontsize=16, fontweight="bold", color="#3498db", alpha=0.35)
ax.text(13,   2000, "CONTAIN",  fontsize=16, fontweight="bold", color="#e67e22", alpha=0.35)
ax.text(13,  -3000, "EXIT",     fontsize=16, fontweight="bold", color="#c0392b", alpha=0.35)

# 9. Axis labels and cosmetics
ax.set_xlabel("Default rate (%)", fontsize=12)
ax.set_ylabel("Mean value per loan (₹)", fontsize=12)
ax.set_title("Risk × Value Matrix — 6 segments  (bubble size = # loans)",
             fontsize=14, pad=15)
ax.set_xlim(0, 15)
ax.set_ylim(-5000, 45000)
ax.grid(alpha=0.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

plt.tight_layout()
plt.savefig(os.path.join(HERE, "out", "slide7_risk_value_matrix.png"),
            dpi=200, bbox_inches="tight")
plt.show()