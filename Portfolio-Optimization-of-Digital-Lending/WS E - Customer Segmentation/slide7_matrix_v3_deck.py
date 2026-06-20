"""
slide6_strategic_posture_grid.py
────────────────────────────────
Renders the seven WS E segments into a four-card strategic posture grid
(GROW / CONTAIN / MAINTAIN / EXIT). Each card carries one or two segments
with key economics and the policy lever that operates inside that posture.

This replaces the v2/v3 bubble chart for the diagnosis slide. A bubble
chart squashes against the bimodal x-axis (5 segments at 0–14% loss-making,
2 at 74–79%); a card grid sidesteps the layout problem entirely and reads
as a strategy framework rather than a scatter plot.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# ── Safe Directory & Font Loading ────────────────────────────────────────────
# Handles both terminal execution (__file__) and interactive notebooks
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

# Safely attempt to load local fonts if the directory exists
font_dir = os.path.join(HERE, "fonts")
if os.path.exists(font_dir):
    for f in ["Roboto-Regular.ttf", "Roboto-Bold.ttf"]:
        font_path = os.path.join(font_dir, f)
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
        else:
            print(f"Warning: Font file missing at {font_path}. Matplotlib will fallback to default.")
else:
    print(f"Warning: 'fonts' directory not found at {font_dir}. Matplotlib will fallback to default.")

# Set the global font to Roboto for this output
plt.rcParams.update({"font.family": "Arial, sans-serif"})  # Fallback to Arial if Roboto is not available

# ── Palette ──────────────────────────────────────────────────────────────────
INK   = "#16242B"
GRAPH = "#5A646B"
HAIR  = "#DCE0E3"
WHITE = "#FFFFFF"

POSTURE_COLOR = {
    "grow":     "#05998c",
    "contain":  "#44c0c5",
    "maintain": "#025043",
    "exit":     "#C52A16",
}

POSTURE_LABEL = {"grow": "GROW", "contain": "CONTAIN",
                 "maintain": "MAINTAIN", "exit": "EXIT"}

POSTURE_SUB = {
    "grow":     "low risk · high value",
    "contain":  "elevated risk · still profitable",
    "maintain": "low risk · moderate value",
    "exit":     "loss-making per loan",
}

POSTURE_INTENT = {
    "grow":     "Expand profitable segments to absorb volume shed elsewhere.",
    "contain":  "Surgically decline the loss tail; keep the profitable majority.",
    "maintain": "No policy change — the working core of the book.",
    "exit":     "Stop new acquisition; let existing book run off.",
}

# ── Source data ──────────────────────────────────────────────────────────────
metrics_path = os.path.join(HERE, "step6_hybrid_metrics.csv")
if os.path.exists(metrics_path):
    df = pd.read_csv(metrics_path)
    def _seg(name_substr):
        m = df[df.segment.str.contains(name_substr, regex=False)]
        return None if m.empty else m.iloc[0]

# Per-card content.
CARDS = {
    "grow": [
        dict(name="SME Prime (A-B)",
             share="9.2% of book",
             numbers="4,558 loans · 3.1% default · +₹33,369 / loan",
             lever=None),
    ],
    "contain": [
        dict(name="SME Non-prime (C-D-E)",
             share="11.0% of book",
             numbers="5,461 loans · 8.1% default · +₹42,631 / loan",
             lever="→ Headline lever  ·  +₹12.97 cr"),
        dict(name="Personal Subprime (D-E)",
             share="11.1% of book",
             numbers="5,486 loans · 13.5% default · +₹5,318 / loan",
             lever="→ Supporting lever  ·  +₹2.33 cr"),
    ],
    "maintain": [
        dict(name="Personal Prime (A-B)",
             share="20.1% of book",
             numbers="9,988 loans · 4.3% default · +₹4,803 / loan",
             lever=None),
        dict(name="Personal Mid (C)",
             share="13.1% of book",
             numbers="6,513 loans · 7.5% default · +₹6,996 / loan",
             lever=None),
    ],
    "exit": [
        dict(name="BNPL (APR ≤ 32.4%)",
             share="26.6% of book",
             numbers="13,180 loans · 6.5% default · −₹782 / loan",
             lever="→ Cease Digital ads + DSA  ·  +₹1.08 cr"),
        dict(name="High-APR Danger (>32.4%)",
             share="8.9% of book",
             numbers="4,414 loans · 42.1% default · −₹1,703 / loan",
             lever="→ Effectively closed  ·  let run off"),
    ],
}

BOTTOM = {
    "grow":     "→ Lever  Grow originations +10%  ·  +₹1.47 cr",
    "maintain": "→ No lever  ·  33.2% of book unchanged",
}

# ── Figure: 11 × 6.2 in (16:9 deck aspect) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 110); ax.set_ylim(0, 62); ax.axis("off")

# Card geometry
CW, CH = 51, 27           # card width × height
MARGIN_X = 3              # left/right margins
GAP_X = 2                 # gap between left/right cards
ROW_BOTTOM_Y = 4          # bottom row y origin
ROW_TOP_Y = 33            # top row y origin
CARD_POS = {
    "grow":     (MARGIN_X,                ROW_TOP_Y),
    "contain":  (MARGIN_X + CW + GAP_X,   ROW_TOP_Y),
    "maintain": (MARGIN_X,                ROW_BOTTOM_Y),
    "exit":     (MARGIN_X + CW + GAP_X,   ROW_BOTTOM_Y),
}

# ── Card renderer ────────────────────────────────────────────────────────────
def draw_card(posture):
    cx, cy = CARD_POS[posture]
    cy_top = cy + CH
    color = POSTURE_COLOR[posture]

    # White card surface with hairline border
    ax.add_patch(mpatches.Rectangle((cx, cy), CW, CH,
                                    fc=WHITE, ec=HAIR, lw=0.6))

    # Coloured accent stripe
    ax.add_patch(mpatches.Rectangle((cx + 2, cy_top - 2.1), 2.6, 0.42,
                                    fc=color, ec="none"))

    # Posture name
    pname = " ".join(POSTURE_LABEL[posture])
    ax.text(cx + 2, cy_top - 3.4, pname,
            fontsize=14, fontweight="bold", color=color, va="top")

    # Sub-label
    ax.text(cx + 2, cy_top - 6.2, POSTURE_SUB[posture],
            fontsize=9, color=GRAPH, va="top")

    # Strategic intent
    ax.text(cx + 2, cy_top - 8.4, POSTURE_INTENT[posture],
            fontsize=9, color=GRAPH, va="top")

    # Segment block(s)
    segments = CARDS[posture]
    seg_y = cy_top - 11.8
    for seg in segments:
        ax.text(cx + 2, seg_y, seg["name"],
                fontsize=11, fontweight="bold", color=INK, va="top")
        ax.text(cx + CW - 2, seg_y, seg["share"],
                fontsize=9, color=GRAPH, va="top", ha="right")

        ax.text(cx + 2, seg_y - 2.4, seg["numbers"],
                fontsize=9, color=GRAPH, va="top")

        if seg.get("lever"):
            ax.text(cx + 2, seg_y - 4.6, seg["lever"],
                    fontsize=9.5, fontweight="bold", color=color, va="top")
            seg_y -= 7.4
        else:
            seg_y -= 5.4

    # Bottom strap
    if posture in BOTTOM:
        ax.plot([cx + 2, cx + CW - 2], [cy + 4.2, cy + 4.2],
                color=HAIR, lw=0.5)
        ax.text(cx + 2, cy + 3.2, BOTTOM[posture],
                fontsize=9.5, fontweight="bold", color=color, va="top")

# Draw all four cards
for p in ("grow", "contain", "maintain", "exit"):
    draw_card(p)

# ── Footer ───────────────────────────────────────────────────────────────────
ax.plot([3, 107], [2.6, 2.6], color=HAIR, lw=0.5)
ax.text(3, 1.5,
        "Seven segments · 49,600 loans · ₹1,398 cr origination",
        fontsize=8.5, color=GRAPH, va="top")
ax.text(107, 1.5,
        "Source: WS E step6_hybrid_metrics.csv · WS I Sections 3 & 6",
        fontsize=8.5, color=GRAPH, va="top", ha="right")

# ── Save ────────────────────────────────────────────────────────────────────
out_dir = os.path.join(HERE, "out")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "slide6_strategic_posture_grid.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=WHITE)
print(f"Saved: {out_path}")
plt.close()