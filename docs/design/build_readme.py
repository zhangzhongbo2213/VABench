"""Render the local README figures from the checked-in result tables.

Requires matplotlib, numpy, Pillow, markdown-it-py and ffmpeg.
Run from any directory: python docs/design/build_readme.py [--animate | --gif-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon, Wedge
from matplotlib.textpath import TextPath, TextToPath
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "docs/media"
DATA = json.loads((ROOT / "docs/data/results.json").read_text())
MODELS = DATA["models"]
INK = "#18233B"
MUTED = "#758096"
LINE = "#E4E9F2"
GROUPS = ["#418FAD", "#8570C2", "#C08369"]
GROUP_INK = ["#23647C", "#5E3E91", "#95482E"]
GROUP_NAMES = ["Spatial perception", "Robot manipulation", "Error recovery"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "text.color": INK, "axes.labelcolor": MUTED,
                     "svg.fonttype": "path", "savefig.facecolor": "white"})


def card(fig):
    fig.patch.set_facecolor("white")
    fig.add_artist(FancyBboxPatch((.003, .004), .994, .992,
                   boxstyle="round,pad=0,rounding_size=0.024",
                   transform=fig.transFigure, facecolor="white",
                   edgecolor=LINE, linewidth=1.1, zorder=-10))


def header(fig, number, label, title):
    fig.text(.043, .938, f"{number}  /  {label}", fontsize=9.5,
             color="#7663BD", weight="bold", va="center")
    fig.text(.042, .876, title, fontsize=24, weight="bold", va="center")


def logo(model, size=40):
    im = Image.open(MEDIA / "logos" / (model["family"] + ".png")).convert("RGBA")
    box = im.getbbox()
    if box:
        im = im.crop(box)
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size))
    canvas.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def title_art():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="250" viewBox="0 0 1280 250" role="img" aria-label="VA-Bench: Measuring Embodied Spatial Intelligence">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#F4F7FE"/><stop offset=".55" stop-color="#FAF9FE"/><stop offset="1" stop-color="#F0F8F8"/></linearGradient>
  <linearGradient id="ink" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#263959"/><stop offset=".55" stop-color="#6651BE"/><stop offset="1" stop-color="#467FA0"/></linearGradient>
  <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r=".8" fill="#A5B2C8" opacity=".28"/></pattern>
</defs>
<rect x="1" y="1" width="1278" height="248" rx="25" fill="url(#bg)" stroke="#E6EAF4"/>
<rect x="1" y="1" width="1278" height="248" rx="25" fill="url(#dots)"/>
<g fill="none" stroke="#B9BDDA" stroke-width="1.3" opacity=".48"><path d="M67 172 128 137 193 172 128 207Z M67 102 128 67 193 102 128 137Z M67 102V172 M193 102V172 M128 137V207"/><path d="M1100 70 1165 109 1100 147 1035 109Z M1100 147V211 M1165 109V174L1100 211 1035 174V109"/></g>
<g fill="#8275C5"><circle cx="128" cy="137" r="4"/><circle cx="1100" cy="147" r="4"/></g>
<text x="640" y="90" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="66" font-weight="700" letter-spacing="-3" fill="url(#ink)">VA-Bench</text>
<text x="640" y="137" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="23" font-weight="500" fill="#24344F">Measuring Embodied Spatial Intelligence</text>
<text x="640" y="175" text-anchor="middle" font-family="DejaVu Sans,Arial,sans-serif" font-size="15" fill="#65738C">Visual Demonstrations · Active Perception · Metric Control</text>
<path d="M552 210H728" stroke="#B3A7DC" stroke-width="3" stroke-linecap="round"/>
</svg>'''
    (MEDIA / "title.svg").write_text(svg)


def success_chart():
    fig = plt.figure(figsize=(12.8, 6.6), dpi=100)
    card(fig)
    header(fig, "01", "MAIN RESULTS", "Task success across 14 environments")
    ax = fig.add_axes([.075, .24, .895, .56])
    ax.set_ylim(0, 65)
    ax.set_xlim(-.7, 11.7)
    ax.set_yticks(np.arange(0, 61, 10), [f"{v}%" for v in range(0, 61, 10)])
    ax.tick_params(axis="y", length=0, labelsize=9, colors=MUTED, pad=9)
    ax.set_xticks([])
    ax.grid(axis="y", color=LINE, linewidth=.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color="#BFC9DA", linewidth=1)
    for i, model in enumerate(MODELS):
        height = model["mean"]
        patch = FancyBboxPatch((i - .295, 0), .59, height,
                     boxstyle="round,pad=0,rounding_size=0.17",
                     facecolor=model["color"], edgecolor="none", zorder=3)
        ax.add_patch(patch)
        ax.errorbar(i, height, yerr=model["sd"], color="#4A566D", capsize=3.2,
                    capthick=1.1, elinewidth=1.1, fmt="none", zorder=4)
        ax.text(i, height + model["sd"] + 1.65, f"{height:.2f}",
                ha="center", va="bottom", fontsize=10.1, weight="bold")
        ax.text(i, -3.9, model["short_label"], ha="center", va="top",
                fontsize=9.1, linespacing=1.55, clip_on=False)
        icon = OffsetImage(logo(model, 128), zoom=33/128)
        ax.add_artist(AnnotationBbox(icon, (i, -18.3), xycoords="data",
                                     frameon=False, annotation_clip=False, pad=0))
    fig.savefig(MEDIA / "success_rates.svg")
    fig.savefig(MEDIA / "success_rates.png", dpi=180)
    plt.close(fig)


def task_matrix():
    matrix = np.array([[m["tasks"][t]["mean"] for m in MODELS] for t in range(14)])
    fig = plt.figure(figsize=(13.7, 8.1), dpi=120)
    card(fig)
    fig.text(.032, .944, "Every task. Every primary model. Success (%)", fontsize=23, weight="bold")
    ax = fig.add_axes([.244, .12, .703, .745])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("va", ["#F5F6FC", "#DDD7F3", "#AEA0DF", "#7863BA", "#4F428B"])
    ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    ax.set_xticks(range(12), [m["short_label"] for m in MODELS], fontsize=8.5)
    ax.set_yticks(range(14), [MODELS[0]["tasks"][t]["task"].replace("_", " ") for t in range(14)], fontsize=8.6)
    ax.tick_params(length=0, pad=10)
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 14, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.axhline(10.5, color="#AEBDD0", linewidth=2)
    for s in ax.spines.values():
        s.set_visible(False)
    for r in range(14):
        for c in range(12):
            ax.text(c, r, f"{matrix[r,c]:.1f}", ha="center", va="center", fontsize=8.6,
                    color="white" if matrix[r,c] >= 61 else "#36425A")
    fig.savefig(MEDIA / "success_by_task.svg")
    plt.close(fig)


def arc_label(ax, text, radius, center, color, size=12.5, tracking=.7):
    """Place upright glyphs along an arc, reversing direction on the lower half."""
    font = FontProperties(family="DejaVu Sans", weight="bold", size=size)
    metrics = TextToPath()
    widths = np.array([metrics.get_text_width_height_descent(c, font, False)[0]
                       for c in text])
    total = widths.sum() + tracking * (len(text) - 1)
    centers = np.cumsum(widths) - widths / 2 + np.arange(len(text)) * tracking - total / 2
    ax.apply_aspect()
    pixels_per_unit = np.linalg.norm(ax.transData.transform((1, 0)) - ax.transData.transform((0, 0)))
    points_to_units = ax.figure.dpi / 72 / pixels_per_unit
    cap_midline = TextPath((0, 0), "H", prop=font).get_extents().y1 / 2
    direction = -1 if np.sin(np.deg2rad(center)) >= 0 else 1
    for character, width, offset in zip(text, widths, centers):
        if character.isspace():
            continue
        angle = np.deg2rad(center) + direction * offset * points_to_units / radius
        glyph = TextPath((0, 0), character, prop=font)
        transform = (Affine2D().translate(-width / 2, -cap_midline)
                     .scale(points_to_units).rotate(angle + direction * np.pi / 2)
                     .translate(radius * np.cos(angle), radius * np.sin(angle)))
        ax.add_patch(PathPatch(glyph, transform=transform + ax.transData,
                              facecolor=color, edgecolor="none", zorder=5))


class Radar:
    def __init__(self):
        self.fig = plt.figure(figsize=(12.8, 8.6), dpi=150)
        card(self.fig)
        header(self.fig, "02", "TOP 10 · BEHAVIORAL PROFILES", "Nine dimensions of embodied intelligence")
        self.ax = self.fig.add_axes([.01, .075, .675, .72])
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-183, 183)
        self.ax.set_ylim(-183, 183)
        self.ax.axis("off")
        self.theta = np.pi / 2 - np.arange(9) * 2 * np.pi / 9
        self.unit = np.column_stack([np.cos(self.theta), np.sin(self.theta)])
        outline = Polygon(self.unit * 100, closed=True, facecolor="#FAFBFE",
                          edgecolor="none", zorder=-3)
        self.ax.add_patch(outline)
        for j, name in enumerate(GROUP_NAMES):
            center = 50 - j * 120
            start, end = center - 60, center + 60
            sector = Wedge((0, 0), 100, start, end, facecolor=GROUPS[j],
                           edgecolor="none", alpha=.075, zorder=-2)
            sector.set_clip_path(outline)
            self.ax.add_patch(sector)
            self.ax.add_patch(Wedge((0, 0), 158, start + 2, end - 2, width=46,
                                    facecolor=GROUPS[j], edgecolor="none", alpha=.045, zorder=-2))
            angles = np.deg2rad(np.linspace(start + 3, end - 3, 120))
            self.ax.plot(158 * np.cos(angles), 158 * np.sin(angles), color=GROUPS[j],
                         linewidth=2.2, alpha=.75, solid_capstyle="round", zorder=1)
            boundary = np.deg2rad(start)
            self.ax.plot(np.array([103, 158]) * np.cos(boundary),
                         np.array([103, 158]) * np.sin(boundary), color=LINE,
                         linewidth=.85, linestyle=(0, (2, 3)), zorder=0)
            arc_label(self.ax, name.upper(), 172, center, GROUP_INK[j])
        for radius in (20, 40, 60, 80, 100):
            self.ax.add_patch(Polygon(self.unit * radius, closed=True,
                  fill=False, edgecolor=LINE, linewidth=.85, zorder=0))
        for i, unit in enumerate(self.unit):
            self.ax.plot([0, unit[0]*100], [0, unit[1]*100], color=LINE, linewidth=.7, zorder=0)
        labels = ["Target\nlocalization", "Active\nexploration", "Spatial\nrelations",
                  "Manipulation\nsemantics", "Manipulation\nplanning", "Fine-grained\nanalysis",
                  "Error\ndetection", "Online\ncorrection", "Post-failure\nadjustment"]
        for i, (label, dim) in enumerate(zip(labels, DATA["dimensions"])):
            center = np.rad2deg(self.theta[i])
            radii = (141, 126) if np.sin(self.theta[i]) >= 0 else (126, 141)
            for line, radius in zip(label.split("\n"), radii):
                arc_label(self.ax, line, radius, center, GROUP_INK[dim["group"]], size=11.2, tracking=.2)
        self.polygon = Polygon(self.unit * 50, closed=True, linewidth=2.4,
                               edgecolor=MODELS[0]["color"], facecolor=(*to_rgb(MODELS[0]["color"]), .16), zorder=3)
        self.ax.add_patch(self.polygon)
        self.points = self.ax.scatter([], [], s=24, zorder=4, edgecolors="white", linewidths=1)
        self.fig.add_artist(FancyBboxPatch((.69, .194), .27, .56,
                           boxstyle="round,pad=.006,rounding_size=.018", transform=self.fig.transFigure,
                           edgecolor=LINE, linewidth=.9, facecolor="#F8FAFD", zorder=-1))
        self.index_label = self.fig.text(.716, .712, "", fontsize=10.5, color=INK, weight="bold")
        self.icon_ax = self.fig.add_axes([.716, .625, .035, .057])
        self.icon_ax.axis("off")
        self.icon_artist = self.icon_ax.imshow(logo(MODELS[0], 60))
        self.model_label = self.fig.text(.763, .650, "", fontsize=16.5, weight="bold", va="center")
        self.fig.text(.716, .567, "OVERALL TASK SUCCESS", fontsize=9.5, color=INK, weight="bold")
        self.success_label = self.fig.text(.715, .513, "", fontsize=31, weight="bold", va="center")
        self.group_values = []
        for j, name in enumerate(GROUP_NAMES):
            y = .433 - j*.081
            self.fig.text(.716, y, name, fontsize=11, weight="bold", color=GROUP_INK[j])
            self.group_values.append(self.fig.text(.716, y-.032, "", fontsize=10.5, weight="bold", color=INK))
        self.progress = []
        for i in range(10):
            x = .044 + i*.0913
            patch = FancyBboxPatch((x, .05), .078, .006,
                     boxstyle="round,pad=0,rounding_size=.0025", transform=self.fig.transFigure,
                     facecolor=LINE, edgecolor="none")
            self.fig.add_artist(patch)
            self.progress.append(patch)

    def update(self, index, mix=0.0):
        current, nxt = MODELS[index], MODELS[(index+1) % 10]
        a, b = np.array(current["capabilities"]), np.array(nxt["capabilities"])
        smooth = mix*mix*(3-2*mix)
        values = a*(1-smooth) + b*smooth
        color = np.array(to_rgb(current["color"]))*(1-smooth) + np.array(to_rgb(nxt["color"]))*smooth
        self.polygon.set_xy(self.unit * values[:,None])
        self.polygon.set_edgecolor(color)
        self.polygon.set_facecolor((*color, .17))
        self.points.set_offsets(self.unit * values[:,None])
        self.points.set_facecolors([color])
        # Numeric labels stay at real endpoints; only the shape morphs between models.
        selected = index if mix < .5 else (index+1)%10
        model = MODELS[selected]
        self.index_label.set_text(f"MODEL {selected+1:02d} / 10")
        self.icon_artist.set_data(logo(model, 60))
        self.model_label.set_text(model["name"].replace("Doubao-seed-", "Doubao ").replace("Gemini-", "Gemini "))
        self.model_label.set_fontsize(14.5 if len(model["name"]) > 18 else 16.5)
        self.success_label.set_text(f"{model['mean']:.2f}%")
        self.success_label.set_color(np.array(to_rgb(model["color"])) * .7 + np.array(to_rgb(INK)) * .3)
        for j, text in enumerate(self.group_values):
            text.set_text("  ·  ".join(f"{DATA['dimensions'][k]['code']} {model['capabilities'][k]:.1f}" for k in range(j*3,j*3+3)))
        for i, patch in enumerate(self.progress):
            patch.set_facecolor(model["color"] if i == selected else LINE)

    def save(self, animate, gif_only=False):
        self.update(0)
        if not gif_only:
            self.fig.savefig(MEDIA / "capabilities_poster.png", dpi=150)
            self.fig.savefig(MEDIA / "capabilities_poster.svg")
        if not animate:
            plt.close(self.fig)
            return
        fps, hold, transition = 20, 28, 18
        width, height = self.fig.canvas.get_width_height()
        master = MEDIA / ".radar-master.mkv"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
                   "-c:v", "libx264rgb", "-preset", "fast", "-crf", "0", "-threads", "4",
                   str(master)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for i in range(10):
                self.update(i)
                self.fig.canvas.draw()
                frame = np.asarray(self.fig.canvas.buffer_rgba())[:,:,:3].tobytes()
                for _ in range(hold):
                    process.stdin.write(frame)
                for step in range(1, transition+1):
                    self.update(i, step/transition)
                    self.fig.canvas.draw()
                    process.stdin.write(np.asarray(self.fig.canvas.buffer_rgba())[:,:,:3].tobytes())
                print(f"Animation: {i+1}/10 models", flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Video rendering failed")
        palette = MEDIA / ".radar-palette.png"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(master),
                        "-vf", "palettegen=stats_mode=full", "-frames:v", "1", str(palette)], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(master),
                        "-i", str(palette), "-lavfi", "paletteuse=dither=none:diff_mode=rectangle",
                        "-loop", "0", str(MEDIA/"capabilities_top10.gif")], check=True)
        if not gif_only:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(master),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-threads", "4", str(MEDIA / "capabilities_top10.mp4")], check=True)
        palette.unlink()
        master.unlink()
        plt.close(self.fig)


def build_readme():
    names = []
    for author in DATA["authors"]:
        marker = str(author["affiliation"]) + (",*" if author.get("equal") else "")
        names.append(f"{author['name']}<sup>{marker}</sup>")
    authors = " · ".join(names[:4]) + "<br>\n" + " · ".join(names[4:])
    table = "| Model | Success | Run-to-run SD |\n|:--|--:|--:|\n" + "".join(
        f"| {m['name']} | {m['mean']:.2f}% | {m['sd']:.2f} pp |\n" for m in MODELS)
    readme = f'''<p align="center">
  <img src="docs/media/title.svg" alt="VA-Bench: Measuring Embodied Spatial Intelligence through Visual Demonstrations, Active Perception, and Metric Control" width="100%">
</p>

<p align="center">
{authors}
</p>

<p align="center">
<sup>1</sup> Dalian University of Technology &nbsp;&nbsp; <sup>2</sup> Nanyang Technological University<br>
<sub>* Equal contribution</sub>
</p>

<p align="center">
  <a href="docs/paper/VA-Bench.pdf" title="Read the paper (40 pages)">
    <img src="docs/media/paper-button.svg" alt="PDF" width="118" height="46">
  </a>
  &nbsp; / &nbsp;
  <a href="https://arxiv.org/abs/2609.19554" title="Read VA-Bench on arXiv">
    <img src="docs/media/arxiv-button.svg" alt="arXiv" width="128" height="46">
  </a>
</p>

<p align="center">
<a href="#task-success">Results</a> &nbsp; / &nbsp;
<a href="#behavioral-profiles">Capability profiles</a> &nbsp; / &nbsp;
<a href="#quick-start">Quick start</a> &nbsp; / &nbsp;
<a href="https://github.com/zhangzhongbo2213/VABench/releases/tag/assets-v1">Download assets</a>
</p>

**Embodied spatial intelligence, tested through action.** VA-Bench evaluates the
complete observe–reason–act–revise loop. From RGB-only demonstrations, multimodal
models learn task context, actively choose camera viewpoints, issue metric Cartesian
commands, and revise their actions using execution feedback. We evaluate 12 models
on 14 manipulation tasks, combining strict task success with nine behavioral
diagnostics of perception, manipulation, and recovery.

## Task success

<p align="center">
  <img src="docs/media/success_rates.svg" alt="Overall success on all 14 tasks for 12 models, with model logos and sample-standard-deviation error bars" width="100%">
</p>

<details>
<summary><strong>View exact scores and all 14 task results</strong></summary>

Each run covers **11 single-arm and 3 dual-arm tasks**, with 20 physically verified
seeds per task. Bars show the mean of the three run-level task macro-averages;
error bars show sample standard deviation. The top three means are close and their
run-level ranges overlap.

{table}

<img src="docs/media/success_by_task.svg" alt="Complete 14-task by 12-model success-rate matrix" width="100%">

[Per-task CSV](docs/data/success_by_task.csv) · [Figure data](docs/data/results.json) · [Sources and aggregation](docs/data/SOURCES.md)

</details>

## Behavioral profiles

<p align="center">
  <img src="docs/media/capabilities_top10.gif" alt="Animated radar profiles of the top 10 models, showing nine behavioral dimensions on a fixed 0–100 scale" width="100%">
</p>

<details>
<summary><strong>Read the nine dimensions</strong></summary>

The tour follows the **top 10 models by overall task success**. Each profile pools
applicable single- and dual-arm episodes from one annotated run; the three recovery
scores exclude error-free episodes. Smooth transitions connect the recorded profiles.

| Spatial perception | Robot manipulation | Error recovery |
|:--|:--|:--|
| **TL** Target localization | **MS** Manipulation semantics | **ED** Error detection |
| **AE** Active exploration | **MP** Manipulation planning | **OC** Online correction |
| **SR** Spatial relations | **FG** Fine-grained pre-contact analysis | **PF** Post-failure adjustment |

Sonnet-5 uses its second annotated run; the other primary models use their first.
The radar describes observed behaviors, while the success chart measures completed
tasks. [Static profile](docs/media/capabilities_poster.png) · [MP4 animation](docs/media/capabilities_top10.mp4)

</details>

## Quick start

**All assets required by the 14 main tasks are provided with this VA-Bench release.**
No separate asset download from RoboTwin is needed. If `assets/` is already present,
start below; otherwise, extract the release's [asset bundle](https://github.com/zhangzhongbo2213/VABench/releases/tag/assets-v1)
into the repository root as described in the [setup guide](docs/SETUP.md).

From the repository root, create the benchmark environment and install dependencies:

```bash
conda create -n va-bench python=3.10 -y
conda activate va-bench
bash script/install.sh
```

See the [setup guide](docs/SETUP.md) for system prerequisites, environment checks,
and reusing an existing RoboTwin environment.

Connect your vision-language model and start with one episode:

```bash
export AGENT_BASE_URL="http://localhost:8000/v1"
export AGENT_MODEL="your-model-name"
export AGENT_API_KEY="EMPTY"

python script/run_benchmark.py --run-id smoke \\
  --tasks grasp_single_cube --num-seeds 1 --gpus 0 --parallel 1
```

The default wire protocol is **OpenAI Chat Completions** (`chat_completions`).
To use Anthropic's native Messages API, switch explicitly before launching:

```bash
export AGENT_WIRE_API="anthropic_messages"
# Or add: --wire-api anthropic_messages
```

When Claude is served through an OpenAI-compatible endpoint, keep the default
`chat_completions` protocol.

Run all 14 tasks and monitor progress:

```bash
python script/run_benchmark.py --run-id full --gpus 0 1 --parallel 3
python script/monitor.py --run-id full
```

[Environment & asset setup](docs/SETUP.md) · [Tasks & validated seeds](configs/main_tasks.json) · [Asset release](https://github.com/zhangzhongbo2213/VABench/releases/tag/assets-v1)

---

Built on RoboTwin. [MIT License](LICENSE) · [Model logo credits](docs/media/logos/SOURCES.md)
'''
    (ROOT / "README.md").write_text(readme)


def preview():
    from markdown_it import MarkdownIt
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    html = parser.render((ROOT / "README.md").read_text())
    import re
    html = re.sub(r'<h2>(.*?)</h2>', lambda m: '<h2 id="'+re.sub('[^a-z0-9 -]','',m[1].lower()).replace(' ','-')+'">'+m[1]+'</h2>', html)
    template = (ROOT / "docs/design/preview.html").read_text()
    (ROOT / "README.preview.html").write_text(template.replace("<!-- README -->", html))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--animate", action="store_true")
    mode.add_argument("--gif-only", action="store_true", help="Regenerate only the radar GIF")
    args = parser.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    if args.gif_only:
        Radar().save(True, gif_only=True)
        print("Radar GIF generated", flush=True)
        return
    title_art()
    success_chart()
    task_matrix()
    Radar().save(args.animate)
    build_readme()
    preview()
    for svg in MEDIA.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print("README, figures and local preview generated", flush=True)


if __name__ == "__main__":
    main()
