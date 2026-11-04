"""Draw the editable PoreLoom Figure 1 from the current method implementation.

Run with a Python environment containing matplotlib and numpy:
    python paper/draw_overview.py

Outputs are independent vector artwork, not a tracing or embedding of the
imagegen visual draft. Signals and evidence points are explicitly schematic.
"""
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/poreloom-overview-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "figures"
W, H = 1600, 1080
FIGURE_WIDTH_MM = 165.1
POINTS_PER_UNIT = FIGURE_WIDTH_MM / 25.4 * 72 / W
C = dict(ink="#172C3B", muted="#536674", line="#C9D4DA", pale="#F4F7F9",
         teal="#087F81", teal_light="#E6F4F1", cached="#ADD5CF",
         purple="#7257A3", purple_light="#F0ECF8", blue="#356FA7", blue_light="#EAF2FA",
         coral="#BD5D50", coral_light="#FBECE7", gold="#D7A342", gold_light="#F5DEAA")

def draw(out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "text.color": C["ink"], "savefig.facecolor": "white",
                         "svg.hashsalt": "poreloom-overview-20261016"})
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_WIDTH_MM / 25.4 * H / W))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    all_text = []
    contained_text = []

    def text(x, y, value, size=30, color=None, weight="normal", ha="left", va="center", **kwargs):
        t = ax.text(x, y, value, fontsize=max(10., size * POINTS_PER_UNIT),
                    color=color or C["ink"], weight=weight, ha=ha, va=va,
                    linespacing=1.2, zorder=6, **kwargs)
        all_text.append(t)
        return t

    def line(points, color=None, lw=0.85, **kwargs):
        x, y = zip(*points)
        ax.plot(x, y, color=color or C["line"], lw=lw, zorder=kwargs.pop("zorder", 2),
                solid_capstyle="round", **kwargs)

    def arrow(points, color=None, lw=0.9, ms=5.6, dashed=False, gid=None):
        p = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        patch = FancyArrowPatch(path=p, arrowstyle="-|>", mutation_scale=ms,
                                color=color or C["muted"], lw=lw,
                                linestyle=(0, (3, 2)) if dashed else "solid", zorder=3)
        if gid is not None:
            patch.set_gid(gid)
        ax.add_patch(patch)

    def box(x, y, w, h, label=None, color="teal", size=30, weight="normal", fill=None,
            edge_width=0.75, gid=None):
        stroke = C.get(color, color)
        face = fill or C.get(color + "_light", C["pale"])
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7",
                              facecolor=face, edgecolor=stroke, lw=edge_width, zorder=1)
        if gid is not None:
            patch.set_gid(gid)
        ax.add_patch(patch)
        if label is not None:
            t = text(x + w / 2, y + h / 2, label, size=size, weight=weight, ha="center")
            contained_text.append((t, (x, y, w, h)))

    def circle(x, y, r, fc, ec=None, lw=0.7):
        patch = Circle((x, y), r, facecolor=fc, edgecolor=ec or fc, lw=lw, zorder=3)
        ax.add_patch(patch)
        return patch

    def signal(x, y, width, height, seed, color=None, steps=22):
        rng = np.random.default_rng(seed)
        vals = np.repeat(rng.normal(size=steps), 5) + rng.normal(0, 0.18, steps * 5)
        vals /= max(np.max(np.abs(vals)), 1)
        xs = np.linspace(x, x + width, len(vals))
        ax.plot(xs, y + vals * height / 2, color=color or C["ink"], lw=0.65, zorder=4)

    # Only panel letters and short component labels are visible.
    # a — signal-to-action loop. All explanatory prose is in caption.txt.
    text(32, 40, "a", size=40, weight="bold")

    # Single DNA strand through a flat membrane/pore illustration.
    for x in np.arange(52, 217, 13):
        if 105 < x < 156:
            continue
        line([(x - 2, 182), (x - 4, 194), (x - 2, 202)], C["line"], lw=0.7)
        line([(x + 3, 182), (x + 4, 194), (x + 2, 202)], C["line"], lw=0.7)
        circle(x, 177, 5.6, "#E0EAF0", "#9AB0BF", lw=0.5)
        circle(x, 208, 5.6, "#E0EAF0", "#9AB0BF", lw=0.5)
    for side in (-1, 1):
        verts = [(130 + side * 11, 142), (130 + side * 31, 143), (130 + side * 39, 163),
                 (130 + side * 24, 183), (130 + side * 18, 218), (130 + side * 7, 231),
                 (130 + side * 7, 189), (130 + side * 9, 161), (130 + side * 11, 142)]
        ax.add_patch(PathPatch(MplPath(verts, [MplPath.MOVETO] + [MplPath.CURVE3] * 8),
                               fc="#D8E9F0", ec=C["blue"], lw=0.9, zorder=3))
    yy = np.linspace(91, 260, 260)
    xx = 130 + 7 * np.sin((yy - 90) / 10)
    ax.plot(xx, yy, color=C["teal"], lw=1.1, zorder=4)
    for y in range(98, 256, 12):
        x = 130 + 7 * np.sin((y - 90) / 10)
        line([(x, y), (x + 5, y - 4)], C["coral"], lw=0.65)
    arrow([(217, 190), (232, 190)])

    text(341, 112, "Raw signal", size=30, ha="center")
    for i, col in enumerate([C["pale"], C["blue_light"], C["teal_light"]]):
        ax.add_patch(Rectangle((240 + i * 70, 150), 67, 78, fc=col, ec="none", zorder=0))
    signal(246, 190, 195, 62, seed=13, steps=30)
    box(476, 72, 260, 244, color="blue", gid="panel_a_causal_normalization")
    text(606, 118, "Causal\nnormalization", size=30, ha="center")
    # Blocks of 10p samples: block j (white) is scaled with statistics of the
    # preceding blocks (teal).
    for i in range(3):
        current = i == 2
        ax.add_patch(Rectangle((494 + i * 75, 172), 75, 44, zorder=2,
                               fc="white" if current else C["teal_light"],
                               ec=C["blue"] if current else C["teal"], lw=1.0 if current else 0.6))
    signal(498, 194, 217, 34, seed=7, steps=24)
    line([(496, 222), (496, 230), (642, 230), (642, 222)], C["teal"], lw=0.8)
    arrow([(569, 230), (569, 252), (681, 252), (681, 218)], color=C["blue"], ms=4.5,
          gid="panel_a_block_scale")
    text(606, 284, "median, MAD", size=28, ha="center")
    box(768, 145, 200, 90, "Streaming\nencoder", size=30)
    box(1016, 145, 200, 90, "Adaptive\npolicy", color="purple", size=30)
    text(868, 114, "b", size=30, ha="center", color=C["teal"], weight="bold")
    text(1116, 114, "c", size=30, ha="center", color=C["purple"], weight="bold")
    for x1, x2 in [(450, 468), (742, 760), (976, 1008)]:
        arrow([(x1, 190), (x2, 190)])

    for y, label, col in [(72, "Accept", "teal"), (159, "Reject", "coral"), (246, "Continue", "blue")]:
        box(1272, y, 180, 62, label, color=col, size=30, weight="bold")
        arrow([(1224, 190), (1244, 190), (1244, y + 31), (1264, y + 31)], color=C[col])
        end = 1566 if label == "Accept" else (1519 if label == "Reject" else 1542)
        line([(1478, y + 31), (1566, y + 31)], C["line"], lw=1.5)
        line([(1478, y + 31), (end, y + 31)], C[col], lw=2.0)
        if label == "Reject":
            line([(end - 6, y + 23), (end + 6, y + 39)], C[col], lw=1.3)
            line([(end - 6, y + 39), (end + 6, y + 23)], C[col], lw=1.3)
        elif label == "Continue":
            circle(end, y + 31, 6, "white", C[col], lw=1)
    arrow([(1362, 315), (1362, 337), (868, 337), (868, 238)], color=C["blue"], lw=1)
    line([(32, 367), (1568, 367)], lw=0.6)

    # b — explicit CNN, an adjacent cache state, and a dense readout.
    text(32, 400, "b", size=40, weight="bold")
    line([(814, 389), (814, 1050)], lw=0.55)

    def feature_stack(x, y, w, h, gid, color="teal"):
        for j in (2, 1, 0):
            patch = Rectangle((x + j * 5, y - j * 5), w, h,
                              fc=C[color + "_light"], ec=C[color], lw=0.55, zorder=3)
            if j == 0:
                patch.set_gid(gid)
            ax.add_patch(patch)

    # Stem: 1 -> 32 convolution, max pool, then 32 -> 64 convolution.
    signal(58, 493, 57, 46, seed=31, steps=10)
    arrow([(122, 493), (143, 493)], ms=4.8)
    feature_stack(151, 465, 49, 60, "panel_b_stem_conv32")
    text(176, 439, "Conv", size=28, ha="center")
    text(176, 497, "32", size=28, ha="center", color=C["teal"])
    arrow([(218, 493), (243, 493)], ms=4.8)
    box(251, 477, 38, 33, color="blue", gid="panel_b_stem_maxpool")
    text(270, 439, "Pool", size=28, ha="center")
    arrow([(297, 493), (340, 493)], ms=4.8)
    feature_stack(348, 465, 49, 60, "panel_b_stem_conv64")
    text(373, 439, "Conv", size=28, ha="center")
    text(373, 497, "64", size=28, ha="center", color=C["teal"])
    arrow([(415, 493), (473, 493), (473, 546), (280, 546), (280, 557)], ms=4.8)

    # Each stage icon has two convolutions and a shortcut; the label is channels x blocks.
    for j, (channels, repeats) in enumerate(((64, 5), (128, 5), (256, 2), (512, 2))):
        y = 565 + 83 * j
        center = y + 26.5
        box(105, y, 350, 53, color="teal", fill="white", gid=f"panel_b_res_stage_{channels}")
        t = text(188, center, f"{channels} ×{repeats}", size=28, ha="center")
        contained_text.append((t, (105, y, 350, 53)))
        feature_stack(278, y + 19, 27, 26, f"panel_b_res_{channels}_conv1")
        feature_stack(345, y + 19, 27, 26, f"panel_b_res_{channels}_conv2")
        arrow([(321, center + 5), (337, center + 5)], ms=3.8, lw=0.65)
        line([(262, center + 5), (262, y + 6), (414, y + 6), (414, center - 4)], C["teal"], lw=0.65)
        circle(414, center + 5, 7, "white", C["teal"], lw=0.6)
        line([(410, center + 5), (418, center + 5)], C["teal"], lw=0.6, zorder=4)
        line([(414, center + 1), (414, center + 9)], C["teal"], lw=0.6, zorder=4)
        arrow([(390, center + 5), (403, center + 5)], ms=3.5, lw=0.65)
        if j < 3:
            arrow([(280, y + 57), (280, y + 62)], ms=3.0, lw=0.65)
            patch = Rectangle((245, y + 64), 70, 12, fc=C["blue_light"], ec=C["blue"], lw=0.65)
            patch.set_gid(f"panel_b_maxpool_{j + 1}")
            ax.add_patch(patch)
            text(222, y + 70, "Pool", size=28, ha="right", color=C["blue"])
            arrow([(280, y + 78), (280, y + 81)], ms=2.8, lw=0.65)

    # Cache sits beside the CNN and contributes to global mean pooling.
    cache = FancyBboxPatch((510, 451), 288, 422, boxstyle="round,pad=0,rounding_size=9",
                          fc="white", ec=C["line"], lw=0.7, linestyle=(0, (4, 3)), zorder=0)
    cache.set_gid("panel_b_cache")
    ax.add_patch(cache)
    for x, y, col, label in [(549, 488, C["cached"], "Cached"),
                             (534, 531, C["teal"], "New"), (673, 531, C["gold_light"], "Tail")]:
        ax.add_patch(Rectangle((x, y - 10), 19, 20, fc=col, ec="none", zorder=1))
        text(x + 29, y, label, size=28)
    cell, start = 18, 590
    for y, n_new, label in ((575, 0, "t"), (675, 2, "t + 1")):
        text(550, y + 17, label, size=28, ha="center")
        for i in range(6 + n_new + 3):
            col = C["cached"] if i < 6 else (C["teal"] if i < 6 + n_new else C["gold_light"])
            ax.add_patch(Rectangle((start + i * cell, y), cell - 1, 34, fc=col, ec="white", lw=0.25, zorder=1))
    for i in (0, 2, 4):
        arrow([(start + (i + .5) * cell, 619), (start + (i + .5) * cell, 665)],
              color="#90B9B1", lw=0.75, ms=4.5)
    arrow([(start + 7.5 * cell, 619), (start + 9.5 * cell, 665)], color=C["gold"], ms=4.5)
    arrow([(689, 719), (689, 765), (650, 765), (650, 792)], ms=4.8)
    box(540, 800, 220, 60, "Σ, count", color="teal", size=30, gid="panel_b_cache_statistics")
    arrow([(463, 840), (532, 840)], color=C["teal"], ms=4.8)
    arrow([(650, 868), (650, 891), (205, 891), (205, 922)], color=C["teal"], ms=5)

    box(85, 930, 240, 65, "Mean pool", size=30, gid="panel_b_mean_pool")
    arrow([(333, 963), (383, 966)], ms=4.8)
    text(510, 909, "MLP", size=30, ha="center")
    dense_columns = [(400, [930, 948, 966, 984, 1002], 512),
                     (505, [935, 956, 977, 998], 128), (620, [953, 979], 2)]
    for (x0, ys0, _), (x1, ys1, _) in zip(dense_columns, dense_columns[1:]):
        for y0 in ys0:
            for y1 in ys1:
                line([(x0 + 7, y0), (x1 - 7, y1)], "#B4CDC9", lw=0.42)
    for x, ys, width in dense_columns:
        for j, y in enumerate(ys):
            circle(x, y, 7, C["teal_light"], C["teal"], lw=0.65).set_gid(f"panel_b_mlp_{width}_{j}")
        text(x, 1037, str(width), size=28, ha="center")
    arrow([(636, 966), (693, 966)], ms=4.8)
    circle(715, 966, 13, "white", C["blue"], lw=0.8).set_gid("panel_b_length_addition")
    text(715, 966, "+", size=28, ha="center", color=C["blue"])
    arrow([(736, 966), (760, 966)], ms=4.8)
    text(784, 966, r"$\ell$", size=32, ha="center")
    box(655, 992, 120, 36, "Linear", color="blue", size=28, gid="panel_b_length_linear")
    arrow([(715, 992), (715, 980)], color=C["blue"], ms=3.0, gid="panel_b_length_logits")
    length_features = text(715, 1056, r"$\phi(\tilde n)$", size=28, color=C["blue"], ha="center")
    length_features.set_gid("panel_b_aligned_length_features")
    arrow([(715, 1040), (715, 1031)], color=C["blue"], ms=3.0, gid="panel_b_length_features_to_linear")

    # c — six-feature state, one active head, and factorized action costs.
    text(842, 400, "c", size=40, weight="bold")
    box(900, 445, 450, 110, color="blue", gid="panel_c_summary")
    feature_text = [
        
        text(1125, 477, r"$[\ell,\ n/S,\ \log(1+n/F),$", size=28, ha="center"),
        text(1125, 523, r"$\ell_{\rm prev},\ \ell_{\min},\ k]$", size=28, ha="center"),
    ]
    for t in feature_text:
        contained_text.append((t, (900, 445, 450, 110)))
    text(1482, 415, "n", size=32, ha="center", style="italic", color=C["blue"])
    arrow([(1482, 438), (1482, 457)], color=C["blue"], gid="panel_c_length_to_bucket")
    box(1410, 465, 145, 70, "b(n)", color="purple", size=32, gid="panel_c_router")

    # Gray branches are routing alternatives. Only Head 2 is active in this illustration.
    inactive = "#BCC7D0"
    line([(945, 620), (1510, 620)], inactive, lw=0.8, linestyle=(0, (3, 3)))
    for center in (945, 1340, 1510):
        arrow([(center, 620), (center, 647)], color=inactive, ms=4.3, dashed=True)
    # Features enter the selected head directly; the length bucket only selects that route.
    arrow([(1105, 563), (1105, 647)], color=C["purple"], lw=1.25, gid="panel_c_features_to_head")
    arrow([(1482, 543), (1482, 590), (1114, 590)], color=C["purple"], dashed=True,
          ms=4.5, gid="panel_c_bucket_selects_head")
    text(1295, 573, "Select", size=28, ha="center", color=C["purple"])

    shown_heads = [(885, "H1", "1"), (1045, "H2", "2"),
                   (1280, "HK", "K"), (1450, "Final", "forced")]
    for x, label, key in shown_heads:
        forced = key == "forced"
        active = key == "2"
        color = "gold" if forced else ("purple" if active else inactive)
        fill = C["gold_light"] if forced else (C["purple_light"] if active else C["pale"])
        height = 145 if forced else 105
        box(x, 655, 120, height, color=color, fill=fill,
            edge_width=1.25 if active else 0.75, gid="panel_c_head_" + key)
        center = x + 60
        t = text(center, 684, label, size=28, ha="center", weight="bold" if active else "normal")
        contained_text.append((t, (x, 655, 120, height)))
        if forced:
            t = text(center, 728, "n = N", size=28, ha="center", color="#8A651B")
            contained_text.append((t, (x, 655, 120, height)))
        options = [("accept", "teal", -18), ("reject", "coral", 18)] if forced else [
            ("accept", "teal", -28), ("reject", "coral", 0), ("continue", "blue", 28)]
        for action, hue, dx in options:
            dot = circle(center + dx, 772 if forced else 728, 8, C[hue])
            dot.set_gid("panel_c_head_" + key + "_" + action)
    text(1223, 704, "…", size=38, ha="center", color=C["muted"])

    # The selected head estimates both components for each legal action.
    arrow([(1105, 768), (1105, 815), (1058, 815), (1058, 828)], color=C["purple"], ms=4.7)
    arrow([(1105, 768), (1105, 815), (1395, 815), (1395, 828)], color=C["purple"], ms=4.7)
    box(915, 836, 285, 120, r"$\hat U(a\mid x)$" + "\nHost time", color="purple", size=28, gid="panel_c_host_cost")
    box(1230, 836, 330, 120, r"$\hat R(a\mid x)$" + "\nTarget risk", color="purple", size=28, gid="panel_c_target_cost")
    line([(1058, 964), (1058, 982), (1395, 982), (1395, 964)], C["purple"], lw=0.8)
    arrow([(1265, 982), (1265, 1005)], color=C["purple"], ms=4.5)
    objective = text(1265, 1035, r"$a^*(x) = \mathrm{arg\,min}_{a}\,[\hat U(a\mid x)+\hat R(a\mid x)]$", size=30, ha="center")
    objective.set_gid("panel_c_objective")

    # Blue denotes a Continue result, followed by new signal and updated features.
    circle(900, 1035, 8, C["blue"]).set_gid("panel_c_continue_result")
    arrow([(932, 1035), (915, 1035)], color=C["blue"], ms=4.5)
    arrow([(885, 1035), (868, 1035), (868, 500), (892, 500)], color=C["blue"], lw=1)
    text(848, 768, "Wait for next chunk", size=26, color=C["blue"], ha="center", rotation=90)

    # Verification concerns the exported artwork, not a new unit-test suite.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds_errors = []
    figure_bounds = fig.bbox
    for t in all_text:
        bb = t.get_window_extent(renderer)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > figure_bounds.x1 + 0.5 or bb.y1 > figure_bounds.y1 + 0.5:
            bounds_errors.append({"kind": "outside_figure", "text": t.get_text()})
    for t, (x, y, w, h) in contained_text:
        bb = t.get_window_extent(renderer)
        p0 = ax.transData.transform((x, y + h))
        p1 = ax.transData.transform((x + w, y))
        if bb.x0 < p0[0] + 1 or bb.x1 > p1[0] - 1 or bb.y0 < p0[1] or bb.y1 > p1[1]:
            bounds_errors.append({"kind": "outside_module", "text": t.get_text()})
    if bounds_errors:
        raise RuntimeError(json.dumps(bounds_errors, indent=2))

    fig.savefig(out / "fig01_overview.pdf", metadata={"Title": "PoreLoom method overview", "Author": "PoreLoom"})
    fig.savefig(out / "fig01_overview.svg")
    fig.savefig(out / "fig01_overview.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    draw(DEFAULT_OUT)
