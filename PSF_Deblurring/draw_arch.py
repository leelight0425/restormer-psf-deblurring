#!/usr/bin/env python
"""Restormer PSF Deblurring — clean symmetric U-Net architecture diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

C_ENC  = "#5B9BD5"
C_DEC  = "#ED7D31"
C_LAT  = "#B1B1B1"
C_EMB  = "#FFC000"
C_SKIP = "#70AD47"
C_ROPE = "#9B59B6"
C_EDGE = "#333333"
C_TXT  = "#111111"
C_ARR  = "#555555"
C_BG   = "#FFFFFF"

FW, FH = 9.0, 4.8
B_W    = 1.20          # TransformerBlock width
B_H    = 0.52          # TransformerBlock height
S_W    = 0.58          # small block width
S_H    = 0.30          # small block height

X_ENC  = 2.4           # encoder column center
X_DEC  = 6.4           # decoder column center

# Row centers: Level1, Level2, Level3, Bottleneck
Y = np.array([3.85, 2.70, 1.55, 0.55])

# ── Drawing helpers ─────────────────────────────────────────────────────────
def block(ax, x, y, w, h, color, text="", sub="", fs=7, fs_sub=5,
          tc=None, lw=0.8, z=2):
    if tc is None:
        tc = "white" if color not in (C_EMB, C_LAT) else C_TXT
    r = min(0.09, h * 0.2)
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r:.4f}",
        facecolor=color, edgecolor=C_EDGE, linewidth=lw, zorder=z))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=z+1)
    if sub:
        ax.text(x, y - h/2 - 0.10, sub, ha="center", va="top",
                fontsize=fs_sub, color=C_EDGE, fontstyle="italic", zorder=z+1)

def rope(ax, x, y):
    ax.text(x + B_W/2 + 0.06, y + 0.18, "RoPE",
            fontsize=4.5, color=C_ROPE, fontweight="bold", fontstyle="italic")

def arr_h(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", color=C_ARR, lw=0.7), zorder=1)

def arr_v(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="->", color=C_ARR, lw=0.7), zorder=1)

def skip(ax, x0, y0, x1, y1):
    """n-shaped skip: go up from x0,y0 → arc across → down to x1,y1."""
    apex = max(y0, y1) + 0.55
    path = mpath.Path(
        [(x0, y0), (x0, apex), (x1, apex), (x1, y1)],
        [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.CURVE3])
    ax.add_patch(mpatches.PathPatch(path, fc="none", ec=C_SKIP, lw=0.8, zorder=0))

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, 9)
ax.set_ylim(0, 5.0)
ax.set_aspect("equal")
ax.set_facecolor(C_BG)
ax.axis("off")

# ═══════════ ENCODER ═══════════

# --- Input ---
xin = X_ENC - 1.40
block(ax, xin, Y[0], S_W, S_H, C_EMB, "5×H×W", fs=5.5)
ax.text(xin, Y[0] + S_H/2 + 0.08, "RGB+XY", ha="center", fontsize=4.5,
        color=C_TXT, fontstyle="italic")

# --- Embed ---
xemb = X_ENC - 0.30
block(ax, xemb, Y[0], S_W, S_H, C_EMB, "Embed", "3×3 Conv", fs=5.5, fs_sub=4.5)
arr_h(ax, xin + S_W/2, xemb - S_W/2, Y[0])

# --- E1 ---
block(ax, X_ENC, Y[0], B_W, B_H, C_ENC, "TB ×4", "d=48  h=1")
arr_h(ax, xemb + S_W/2, X_ENC - B_W/2, Y[0])
rope(ax, X_ENC, Y[0])

# E1 → E2
arr_v(ax, X_ENC, Y[0]-B_H/2, Y[1]+B_H/2)
ax.text(X_ENC-0.05, (Y[0]+Y[1])/2, "↓  PixelUnshuffle", fontsize=4.8,
        color=C_ARR, fontstyle="italic", ha="right", va="center")

# --- E2 ---
block(ax, X_ENC, Y[1], B_W, B_H, C_ENC, "TB ×6", "d=96  h=2")
rope(ax, X_ENC, Y[1])

# E2 → E3
arr_v(ax, X_ENC, Y[1]-B_H/2, Y[2]+B_H/2)
ax.text(X_ENC-0.05, (Y[1]+Y[2])/2, "↓  PixelUnshuffle", fontsize=4.8,
        color=C_ARR, fontstyle="italic", ha="right", va="center")

# --- E3 ---
block(ax, X_ENC, Y[2], B_W, B_H, C_ENC, "TB ×6", "d=192  h=4")
rope(ax, X_ENC, Y[2])

# E3 → Latent
arr_v(ax, X_ENC, Y[2]-B_H/2, Y[3]+B_H/2)
ax.text(X_ENC-0.05, (Y[2]+Y[3])/2, "↓  PixelUnshuffle", fontsize=4.8,
        color=C_ARR, fontstyle="italic", ha="right", va="center")

# --- Latent ---
block(ax, X_ENC, Y[3], B_W, B_H, C_LAT, "TB ×8", "d=384  h=8")
rope(ax, X_ENC, Y[3])

# --- Latent → D3 ---
arr_h(ax, X_ENC + B_W/2, X_DEC - B_W/2, Y[3])

# ═══════════ DECODER ═══════════

# --- D3 ---
block(ax, X_DEC, Y[2], B_W, B_H, C_DEC, "TB ×6", "d=192  h=4")
rope(ax, X_DEC, Y[2])
skip(ax, X_ENC+B_W/2, Y[2], X_DEC-B_W/2, Y[2])  # E3→D3
arr_v(ax, X_DEC, Y[3]+B_H/2, Y[2]-B_H/2)
ax.text(X_DEC+0.05, (Y[3]+Y[2])/2, "PixelShuffle  ↑", fontsize=4.8,
        color=C_ARR, fontstyle="italic", ha="left", va="center")

# --- D2 ---
block(ax, X_DEC, Y[1], B_W, B_H, C_DEC, "TB ×6", "d=96  h=2")
rope(ax, X_DEC, Y[1])
skip(ax, X_ENC+B_W/2, Y[1], X_DEC-B_W/2, Y[1])  # E2→D2
arr_v(ax, X_DEC, Y[2]+B_H/2, Y[1]-B_H/2)
ax.text(X_DEC+0.05, (Y[2]+Y[1])/2, "PixelShuffle  ↑", fontsize=4.8,
        color=C_ARR, fontstyle="italic", ha="left", va="center")

# --- D1 ---
block(ax, X_DEC, Y[0], B_W, B_H, C_DEC, "TB ×4", "d=48  h=1")
rope(ax, X_DEC, Y[0])
skip(ax, X_ENC+B_W/2, Y[0], X_DEC-B_W/2, Y[0])  # E1→D1
arr_v(ax, X_DEC, Y[1]+B_H/2, Y[0]-B_H/2)
ax.text(X_DEC+0.05, (Y[1]+Y[0])/2, "PixelShuffle  ↑", fontsize=4.8,
        color=C_ARR, fontstyle="italic", ha="left", va="center")

# Cat+1×1 labels
for yy in Y[0:3]:
    ax.text(X_DEC, yy - B_H/2 - 0.22, "Cat + 1×1 Conv",
            ha="center", fontsize=5, color=C_EDGE, fontstyle="italic")

# ═══════════ REFINEMENT + OUTPUT ═══════════

xref = X_DEC + 1.25
block(ax, xref, Y[0], B_W*0.85, B_H, C_DEC, "Ref ×4", "d=96  h=1", fs=7.5, fs_sub=5)
arr_h(ax, X_DEC + B_W/2, xref - B_W*0.42, Y[0])

xo = xref + 1.00
block(ax, xo, Y[0], S_W, S_H, C_EMB, "3×H×W", fs=5.5)
ax.text(xo, Y[0] + S_H/2 + 0.08, "RGB", ha="center", fontsize=4.5,
        color=C_TXT, fontstyle="italic")
arr_h(ax, xref + B_W*0.42, xo - S_W/2, Y[0])

# ═══════════ Global Residual ═══════════
ax.annotate("", xy=(xo - S_W/2 + 0.03, Y[0] - 0.42),
            xytext=(xin + S_W/2 - 0.03, Y[0] - 0.42),
            arrowprops=dict(arrowstyle="->", color=C_ROPE, lw=0.7, ls="--",
                            connectionstyle="arc3,rad=-0.22"), zorder=0)
ax.text((xin + xo)/2, Y[0] - 0.65, "Global Residual (1×1 Conv)",
        ha="center", fontsize=5.5, color=C_ROPE, fontstyle="italic")

# ═══════════ Column Labels ═══════════
ax.text(X_ENC, Y[0] + B_H/2 + 0.48, "Encoder", ha="center",
        fontsize=8, fontweight="bold", color=C_ENC)
ax.text(X_DEC, Y[0] + B_H/2 + 0.48, "Decoder", ha="center",
        fontsize=8, fontweight="bold", color=C_DEC)

# Level labels
for label, yy in [("Level 1", Y[0]), ("Level 2", Y[1]),
                   ("Level 3", Y[2]), ("Bottleneck", Y[3])]:
    ax.text(0.30, yy, label, ha="left", va="center", fontsize=6.5,
            fontweight="bold", color=C_TXT)

# RoPE annotation
ax.text(X_ENC + 0.05, Y[0] + B_H/2 + 0.38, "2D RoPE applied in all MDTA blocks",
        fontsize=5.2, color=C_ROPE, fontstyle="italic", ha="center")

# ═══════════ Legend ═══════════
ly, lx0, ldx = 4.55, 0.30, 1.35
for i, (c, lab) in enumerate([
    (C_ENC, "Encoder TB"), (C_DEC, "Decoder TB"), (C_LAT, "Latent TB"),
    (C_EMB, "Embed/Proj"), (C_SKIP, "Skip Conn."), (C_ROPE, "2D RoPE"),
]):
    cx = lx0 + i * ldx
    block(ax, cx, ly, 0.25, 0.14, c, "", lw=0.5, z=1)
    ax.text(cx + 0.20, ly, lab, fontsize=4.8, va="center", color=C_TXT)

# ═══════════ Title ═══════════
mid = (X_ENC + xref) / 2
ax.text(mid, 4.72, "Restormer for PSF Deblurring (XY Variant)",
        ha="center", fontsize=10, fontweight="bold", color=C_TXT)
ax.text(mid, 4.58, "Input: RGB + XY coordinate channels (5ch)  |  2D RoPE in all TransformerBlocks  |  U-Net Transformer",
        ha="center", fontsize=6.2, color=C_EDGE)

# ═══════════ TB detail ═══════════
ax.text(mid, 0.16,
        "TB = TransformerBlock:  LayerNorm → MDTA (+2D RoPE) → LayerNorm → GDFN",
        ha="center", fontsize=5.8, color=C_EDGE, fontstyle="italic")

# ── Save ───────────────────────────────────────────────────────────────────
for fmt in ("pdf", "png"):
    path = f"PSF_Deblurring_RestormerXY_Arch.{fmt}"
    fig.savefig(path, dpi=300, facecolor=C_BG, edgecolor="none")
    print(f"Saved: {path}")
plt.close()
