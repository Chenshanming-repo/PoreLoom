"""Draw the editable PoreLoom Figure 1 from the current method implementation.

Run with a Python environment containing matplotlib and numpy:
    python paper/draw_overview.py

Outputs are independent vector artwork, not a tracing or embedding of the
imagegen visual draft. Signals and evidence points are explicitly schematic.
"""
from pathlib import Path
import argparse
import hashlib
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

SOURCES = [
    "docs/method.md", "src/poreloom/method/pipeline.py", "src/poreloom/method/head.py",
    "src/poreloom/method/pretrain.py", "src/poreloom/method/backbone.py",
    "src/poreloom/model.py", "src/poreloom/streaming.py",
    "src/poreloom/stream_encoder.py", "src/poreloom/stream_controller.py",
    "src/poreloom/stream_bank.py", "src/poreloom/stream_config.py", "src/poreloom/stream_training.py",
    "experiments/20261014_final/icarust_live.py",
]

CAPTION = """Figure 1. PoreLoom performs streaming adaptive sampling directly from raw nanopore current.

(a) Current arrives in successive chunks. After skipping the first 1,500 raw samples, normalization uses past signal statistics regularized toward a training-set prior. The streaming encoder updates evidence for an adaptive policy, which selects Accept (sequence the read to its natural end without further classification), Reject (request ejection), or Continue (retain the read state and await more signal). The blue return arrow denotes the arrival of another chunk from the same read. At the first eligible callback reaching the configured raw endpoint N, only Accept or Reject is available. The default N = 7,500 leaves at most 6,000 usable samples after prefix skipping. The sampling rate f_s defaults to 4,000 Hz; N and f_s are supplied at the entrypoint and stored with the trained model. A time budget T can instead specify N = floor(T f_s), including the skipped prefix. Forced action occurs at the first eligible callback reaching N, with command completion also subject to arrival and service delays. The small b and c labels identify the expanded components below.

(b) The streaming encoder uses the ReadCurrent convolutional architecture. The stem consists of a 32-channel Conv1D (kernel 19, stride 3), batch normalization, ReLU, MaxPool (kernel 2, stride 2), and a 64-channel Conv1D (kernel 3). Four residual stages contain 5, 5, 2 and 2 blocks with 64, 128, 256 and 512 channels, respectively; each block contains two kernel-3 convolutions with batch normalization and a residual connection. Labels give channels and block repetitions. MaxPool (kernel 3, stride 2) follows each of the first three stages. The feature-map stacks and residual shortcuts depict these operations schematically. Global mean pooling produces a 512-dimensional vector, followed by a 512-to-128 linear projection, LayerNorm, GELU, and a 128-to-2 linear classifier, grouped as the MLP. Circles illustrate the fully connected layers; the numbers below them give their actual widths. The blue branch uses the aligned raw-signal endpoint n_tilde = 1,500 + 48 floor((n - 1,500)/48), where n = min(n_received, N) is the observed length used in panel c and all lengths are measured in sample points. At eligible decisions, n and n_tilde differ by at most 47 samples; for example, n = 3,000 gives n_tilde = 2,988. The branch forms phi(n_tilde) = [n_tilde/N, log(1+n_tilde/f_s)] and applies a learned 2-to-2 affine layer. With z_signal denoting the two CNN/MLP logits, z_final = z_signal + W_n phi(n_tilde) + b_n, followed by ell = z_final,target - z_final,host. This length term is learned during streaming encoder training so that class scores explicitly account for prefix length. Here f_s is the configured sampling rate, also used to convert sample counts to seconds. The adjacent cache diagram shows two successive updates, t and t+1. Pale teal denotes contributions of already finalized frames, dark teal denotes newly finalized frames, and gold denotes the boundary-dependent tail that is recomputed. The state retains accumulated feature sums and frame counts, rather than a history of all intermediate activations. New final-frame sums are added to the stable sum; the current tail sum is replaced. Their combined sum divided by their combined count gives the global mean. New frames are computed from overlapping signal windows with enough left context; at sufficient prefix length the recomputed tail contains 11 frames. Each read has its own state. Frame counts and drawn neurons are schematic.

(c) Each decision uses six features: x(n) = [ell, n/S, log(1+n/F), ell_prev, ell_min, k], where ell is the current length-adjusted target-minus-host logit from panel b, ell_prev is its value at the preceding valid look, ell_min is its running minimum including the current look, and k counts valid looks. At the first look, ell_prev = ell. Here n is the observed raw-signal length capped at endpoint N. S and F are fixed feature-reference scales, with S = 7,500 and F = 4,000 for the reported main models, including sputum acquired at 5,000 Hz. They do not resample the signal. Features are standardized using fitting-set statistics and supplied directly to the selected head. Independently, b(n) selects that head using only the observed length n and fixed bucket boundaries. For n < N, the usable length is L' = 48 floor((n - 1,500)/48); once L' >= 432, the one-based head index is b(n) = 1 + min(K - 1, floor((L' - 432)/768)), with K = ceil((N - 1,500 - 432)/768). K denotes the configurable number of ordinary decision heads. When a time budget T is specified, N = floor(T f_s), so K depends on both the time budget and sampling rate; for a fixed sample budget N, changing f_s alone does not change K. Before that minimum length no head is eligible. The solid purple arrow carries x(n) into the active head, while the dashed purple Select arrow indicates length-based control of the route. Each of the K ordinary heads is an independent 6-to-64-to-6 MLP with GELU. Heads 1, 2 and K are shown schematically, with an ellipsis for intermediate heads. The separate forced head gives K + 1 heads in total. Purple highlights the selected head; dashed gray branches are inactive alternatives. Teal, coral and blue circles denote Accept, Reject and Continue. At n = N, the forced head has only Accept and Reject (a 6-to-64-to-4 MLP). A read acts at most once per bucket, and arrivals can skip buckets.

For each legal action a, the head predicts two costs. Rollout supervision for read i is U_i(a) = 1[host_i] T_i(a)/T_ref and R_i(a) = 1[target_i and a Reject command is issued in the rollout]. T_i(a) is total pore occupancy under that action and subsequent policy, including decision-service and ejection delays where applicable. T_ref is the mean natural host-read duration in the head-fitting DEV-A data. Thus target reads have zero U label, and R refers to a Reject command, not necessarily a successful physical ejection. The displayed u(a) and r(a) are the learned predictions of U and R: u(a) = softplus(z_u(a)), r(a) = sigmoid(z_r(a)) for Reject/Continue, and r(Accept) = 0. Action selection minimizes their weighted sum:

    Q(a | x, b) = u(a | x, b) + beta r(a | x, b),
    a* = arg min_{a in A_b} Q(a | x, b).

A_b = {Accept, Reject, Continue} for ordinary heads and {Accept, Reject} for the forced head. A larger beta places a greater penalty on target rejection; the default operating point maximizes command accuracy on DEV-K, with ties resolved by lower target command false-rejection rate. Heads are fitted backward from the forced head to earlier buckets. Continue labels therefore include the outcomes of already fitted later heads. The blue return path depicts a Continue result followed by a new observation and updated x(n).

Head training minimizes the class-balanced component loss, averaged over legal actions:

    L_b = (1 / sum_i w_i) sum_i w_i (1 / |A_b|) sum_{a in A_b}
          [(u_i(a) - U_i(a))^2 + 1[a != Accept] BCE(r_i(a), R_i(a))].

Here w_i = 0.5/q for target nodes and 0.5/(1-q) for host nodes, with q the target fraction among the training nodes of that bucket. BCE is implemented on the corresponding pre-sigmoid logits for numerical stability. beta weights the action-selection score, rather than multiplying the BCE term in this supervised loss.
"""


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
    box(482, 145, 238, 90, "Normalize", color="blue", size=30)
    box(768, 145, 200, 90, "Streaming\nencoder", size=30)
    box(1016, 145, 200, 90, "Adaptive\npolicy", color="purple", size=30)
    text(868, 114, "b", size=30, ha="center", color=C["teal"], weight="bold")
    text(1116, 114, "c", size=30, ha="center", color=C["purple"], weight="bold")
    for x1, x2 in [(450, 474), (728, 760), (976, 1008)]:
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
    arrow([(1362, 315), (1362, 337), (341, 337), (341, 238)], color=C["blue"], lw=1)
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
                   (1280, "HK", "K"), (1450, "Forced", "forced")]
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
    arrow([(1105, 768), (1105, 839), (1058, 839), (1058, 883)], color=C["purple"], ms=4.7)
    arrow([(1105, 768), (1105, 839), (1395, 839), (1395, 883)], color=C["purple"], ms=4.7)
    box(915, 891, 285, 65, "u(a)  Host time", color="purple", size=28, gid="panel_c_host_cost")
    box(1230, 891, 330, 65, "r(a)  Target reject", color="purple", size=28, gid="panel_c_target_cost")
    line([(1058, 964), (1058, 982), (1395, 982), (1395, 964)], C["purple"], lw=0.8)
    arrow([(1265, 982), (1265, 1005)], color=C["purple"], ms=4.5)
    objective = text(1265, 1035, r"$a^* = \mathrm{arg\,min}_{a}\,[u(a)+\beta r(a)]$", size=30, ha="center")
    objective.set_gid("panel_c_objective")

    # Blue denotes a Continue result, followed by new signal and updated features.
    circle(930, 1035, 8, C["blue"]).set_gid("panel_c_continue_result")
    arrow([(993, 1035), (945, 1035)], color=C["blue"], ms=4.5)
    arrow([(915, 1035), (868, 1035), (868, 500), (892, 500)], color=C["blue"], lw=1)

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

    stem = out / "fig01_overview"
    fig.savefig(stem.with_suffix(".svg"), metadata={"Title": "PoreLoom method overview", "Date": "2026-10-16"})
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": "PoreLoom method overview", "Author": "PoreLoom"})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(out / "poreloom_main_figure_preview.png", dpi=270)
    plt.close(fig)
    (ROOT / "review/overview_caption_extended.txt").write_text(CAPTION, encoding="utf-8")
    provenance = {
        "method": "PoreLoom with factorized action-cost policy (FACP)",
        "revision": "v10: K ordinary decision heads; B reserved for usable-sample budget",
        "visible_panels": ["a", "b", "c"], "visible_titles": [],
        "encoder": {"stem_channels": [32, 64], "residual_channels": [64, 128, 256, 512],
                    "residual_blocks": [5, 5, 2, 2], "max_pool_layers": 4,
                    "readout": [512, 128, 2], "cache": "stable and tail feature sums and counts",
                    "length_branch": {"input": "n_tilde = 1500 + 48 * floor((n - 1500) / 48)",
                                      "features": ["n_tilde / N", "log(1 + n_tilde / f_s)"],
                                      "linear_dimensions": [2, 2],
                                      "operation": "add W_n phi(n_tilde) + b_n to the two class logits"}},
        "decision_heads": {"regular": "ceil((N - 1500 - 432) / 768)", "symbol": "K", "forced": 1, "total": "K + 1", "shown_regular": [1, 2, "K"],
                           "highlighted_regular": 2, "ordinary_actions": ["accept", "reject", "continue"],
                           "forced_actions": ["accept", "reject"]},
        "decision_inputs": ["current logit", "n/S", "log(1+n/F)", "previous logit", "minimum logit", "valid look count"],
        "head_routing": {"selection_input": "raw observed length n", "head_input": "standardized x(n)",
                         "learned_router": False, "direct_x_to_bucket_edge": False,
                         "used_samples": "48 * floor((min(n, N) - 1500) / 48), for n >= 1500",
                         "ordinary_head_one_based": "1 + min(K - 1, floor((used_samples - 432) / 768))",
                         "forced_at": "N", "default_endpoint": 7500, "default_fs_hz": 4000},
        "decision_costs": {"u_target": "1[host] * total_pore_seconds / host_reference_seconds",
                           "r_target": "1[target and reject command]", "score": "u_hat + beta * r_hat",
                           "training_loss": "class-balanced MSE(u) + BCE(r); Accept risk masked"},
        "date": "2026-10-16", "vector_renderer": "Matplotlib " + matplotlib.__version__,
        "width_mm": FIGURE_WIDTH_MM, "height_mm": FIGURE_WIDTH_MM * H / W,
        "png_dpi": 600, "svg_text": "editable; DejaVu Sans", "pdf_fonts": "embedded TrueType",
        "schematic_only": True, "contains_experimental_measurements": False,
        "illustration_rng_seed": 13,
        "smallest_text_pt": min(t.get_fontsize() for t in all_text),
        "text_count": len(all_text), "text_bounds_errors": bounds_errors,
        "sources": json.loads((ROOT / "provenance/overview_sources.json").read_text()),
        "generator": "scripts/draw_overview.py",
        "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in sorted(out.glob("fig01_overview.*"))},
    }
    (ROOT / "review/overview_render.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "text_count": len(all_text),
                      "bounds_errors": bounds_errors, "formats": ["svg", "pdf", "png"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    draw(args.out)
