from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


COLORS = {
    "gripper": "#11A8C4",
    "object": "#D64A9B",
    "object_axis": "#F28E2B",
    "support": "#39A96B",
    "true": "#20A464",
    "false": "#D64545",
    "unknown": "#D6A21C",
    "grid": "#D9DEE5",
    "text": "#17202A",
}

NODE_STYLE = {
    "gripper.jaw_center": ("gripper", "D", 72, "jaw center"),
    "gripper.finger_a_inner": ("gripper", "o", 62, "contact A"),
    "gripper.finger_b_inner": ("gripper", "o", 62, "contact B"),
    "object.center": ("object", "o", 82, "object center"),
    "object.axis_start": ("object_axis", "^", 54, "axis start"),
    "object.axis_end": ("object_axis", "^", 54, "axis end"),
    "support.plane_anchor": ("support", "s", 58, "support"),
}
SHORT_NODE_LABELS = {
    "gripper.jaw_center": "jaw",
    "gripper.finger_a_inner": "A",
    "gripper.finger_b_inner": "B",
    "object.center": "object",
    "object.axis_start": "axis-",
    "object.axis_end": "axis+",
    "support.plane_anchor": "table",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a sparse spatial graph without an RGB background.")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perspective-output", type=Path)
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_sheet(graph, args.output)
    if args.perspective_output is not None:
        args.perspective_output.parent.mkdir(parents=True, exist_ok=True)
        render_perspective(graph, args.perspective_output)
    print(f"spatial_graph: {args.output.resolve()}")
    if args.perspective_output is not None:
        print(f"perspective_graph: {args.perspective_output.resolve()}")


def graph_geometry(graph: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    world = {
        node["id"]: np.asarray(node["position_mean_world_m"], dtype=np.float64)
        for node in graph["nodes"]
    }
    origin = world["object.center"].copy()
    local_mm = {node_id: (position - origin) * 1000.0 for node_id, position in world.items()}
    return local_mm, origin


def render_sheet(graph: dict, output: Path) -> None:
    nodes, origin = graph_geometry(graph)
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 0.84], height_ratios=[1, 1])
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 1])
    ax_yz = fig.add_subplot(grid[0, 2])
    ax_info = fig.add_subplot(grid[1, 2])

    draw_3d(ax3d, graph, nodes)
    draw_projection(ax_xy, graph, nodes, axes=(0, 1), labels=("World X", "World Y"), title="Top / XY")
    draw_projection(ax_xz, graph, nodes, axes=(0, 2), labels=("World X", "World Z"), title="Front / XZ")
    draw_projection(ax_yz, graph, nodes, axes=(1, 2), labels=("World Y", "World Z"), title="Side / YZ")
    draw_info(ax_info, graph, origin)

    context = graph.get("diagnostic_context", {})
    phase = str(context.get("phase", "state snapshot")).replace("_", " ")
    object_label = str(context.get("object", "object"))
    fig.suptitle(
        f"Sparse 3D Operational Graph | verify_grasp | {object_label} | {phase}",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=0.975,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.055, top=0.925, wspace=0.28, hspace=0.28)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_perspective(graph: dict, output: Path) -> None:
    nodes, _ = graph_geometry(graph)
    fig = plt.figure(figsize=(10, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    draw_3d(ax, graph, nodes)
    fig.suptitle("Sparse 3D Operational Graph", fontsize=19, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.92)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


def draw_3d(ax, graph: dict, nodes: dict[str, np.ndarray]) -> None:
    draw_support_plane_3d(ax, nodes)
    draw_structural_axes_3d(ax, nodes)
    draw_relation_edges_3d(ax, graph, nodes)
    draw_nodes_3d(ax, nodes)

    all_points = np.stack([position for node_id, position in nodes.items() if node_id in NODE_STYLE])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    spans = np.maximum(maxs - mins, [90.0, 260.0, 95.0])
    half = spans * np.array([0.72, 0.58, 0.75])
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])
    ax.set_box_aspect((half[0], half[1], half[2]))
    ax.view_init(elev=24, azim=-52)
    ax.set_xlabel("World X relative to object (mm)", labelpad=10)
    ax.set_ylabel("World Y relative to object (mm)", labelpad=10)
    ax.set_zlabel("World Z relative to object (mm)", labelpad=10)
    ax.set_title("3D perspective", fontsize=14, pad=14, fontweight="bold")
    style_3d_axis(ax)
    ax.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=True, fontsize=9)


def draw_support_plane_3d(ax, nodes: dict[str, np.ndarray]) -> None:
    anchor = nodes["support.plane_anchor"]
    object_axis_points = np.stack([nodes["object.axis_start"], nodes["object.axis_end"]])
    x_center = float(object_axis_points[:, 0].mean())
    y_center = float(object_axis_points[:, 1].mean())
    x = np.linspace(x_center - 70, x_center + 70, 2)
    y = np.linspace(y_center - 150, y_center + 150, 2)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, anchor[2])
    ax.plot_surface(xx, yy, zz, color=COLORS["support"], alpha=0.10, linewidth=0)
    ax.plot_wireframe(xx, yy, zz, color=COLORS["support"], alpha=0.30, linewidth=0.7)


def draw_structural_axes_3d(ax, nodes: dict[str, np.ndarray]) -> None:
    axis_start = nodes["object.axis_start"]
    axis_end = nodes["object.axis_end"]
    ax.plot(
        [axis_start[0], axis_end[0]],
        [axis_start[1], axis_end[1]],
        [axis_start[2], axis_end[2]],
        color=COLORS["object_axis"],
        linewidth=5,
        alpha=0.85,
        label="object main axis",
    )
    finger_a = nodes["gripper.finger_a_inner"]
    finger_b = nodes["gripper.finger_b_inner"]
    ax.plot(
        [finger_a[0], finger_b[0]],
        [finger_a[1], finger_b[1]],
        [finger_a[2], finger_b[2]],
        color=COLORS["gripper"],
        linewidth=4,
        linestyle="--",
        alpha=0.85,
        label="gripper closing axis",
    )


def draw_relation_edges_3d(ax, graph: dict, nodes: dict[str, np.ndarray]) -> None:
    duplicate_offsets = {
        "supported_by_table": np.array([-4.0, 0.0, 0.0]),
        "lifted_from_support": np.array([4.0, 0.0, 0.0]),
        "object_between_fingers": np.array([-3.0, 0.0, 0.0]),
        "moves_with_gripper": np.array([3.0, 0.0, 0.0]),
    }
    for edge in graph["edges"]:
        edge_id = edge["id"]
        if edge["relation"] == "contact":
            draw_contact_ring_3d(ax, nodes[edge["source"]], edge_id)
            continue
        source = nodes[edge["source"]] + duplicate_offsets.get(edge_id, 0.0)
        target = nodes[edge["target"]] + duplicate_offsets.get(edge_id, 0.0)
        color = COLORS[edge["state"]]
        linestyle = "--" if edge["state"] == "unknown" else (":" if edge["state"] == "false" else "-")
        ax.plot(
            [source[0], target[0]],
            [source[1], target[1]],
            [source[2], target[2]],
            color=color,
            linewidth=2.8,
            linestyle=linestyle,
            alpha=0.95,
        )
        midpoint = (source + target) / 2.0
        ax.text(midpoint[0], midpoint[1], midpoint[2], relation_label(edge_id), color=color, fontsize=8)


def draw_contact_ring_3d(ax, point: np.ndarray, edge_id: str) -> None:
    theta = np.linspace(0, 2 * np.pi, 60)
    radius = 7.0
    ax.plot(
        point[0] + radius * np.cos(theta),
        point[1] + radius * np.sin(theta),
        np.full_like(theta, point[2]),
        color=COLORS["true"],
        linewidth=2.8,
    )
    ax.text(point[0] + 8, point[1], point[2] + 3, relation_label(edge_id), color=COLORS["true"], fontsize=8)


def draw_nodes_3d(ax, nodes: dict[str, np.ndarray]) -> None:
    for node_id, (color_key, marker, size, label) in NODE_STYLE.items():
        point = nodes[node_id]
        ax.scatter(
            point[0],
            point[1],
            point[2],
            s=size,
            marker=marker,
            color=COLORS[color_key],
            edgecolor="white",
            linewidth=1.2,
            depthshade=False,
            zorder=10,
        )
        offset = np.array([4.0, 4.0, 4.0])
        ax.text(*(point + offset), label, fontsize=8.5, color=COLORS["text"])


def draw_projection(ax, graph: dict, nodes: dict[str, np.ndarray], *, axes, labels, title: str) -> None:
    first, second = axes
    support = nodes["support.plane_anchor"]
    if second == 2:
        ax.axhspan(support[2] - 2, support[2] + 2, color=COLORS["support"], alpha=0.18)
    draw_projection_line(ax, nodes, "object.axis_start", "object.axis_end", axes, COLORS["object_axis"], 4.0, "-")
    draw_projection_line(
        ax,
        nodes,
        "gripper.finger_a_inner",
        "gripper.finger_b_inner",
        axes,
        COLORS["gripper"],
        3.2,
        "--",
    )
    for edge in graph["edges"]:
        if edge["relation"] == "contact":
            point = nodes[edge["source"]]
            ax.add_patch(Circle((point[first], point[second]), radius=4.2, fill=False, color=COLORS["true"], linewidth=2.2))
            continue
        offset = -2.5 if edge["id"] in {"supported_by_table", "object_between_fingers"} else 2.5
        source = nodes[edge["source"]].copy()
        target = nodes[edge["target"]].copy()
        source[first] += offset
        target[first] += offset
        linestyle = "--" if edge["state"] == "unknown" else (":" if edge["state"] == "false" else "-")
        ax.plot(
            [source[first], target[first]],
            [source[second], target[second]],
            color=COLORS[edge["state"]],
            linewidth=2.0,
            linestyle=linestyle,
            alpha=0.9,
        )
    for node_id, (color_key, marker, size, label) in NODE_STYLE.items():
        point = nodes[node_id]
        ax.scatter(
            point[first],
            point[second],
            s=size * 0.72,
            marker=marker,
            color=COLORS[color_key],
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        ax.annotate(
            SHORT_NODE_LABELS[node_id],
            (point[first], point[second]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xlabel(f"{labels[0]} relative to object (mm)")
    ax.set_ylabel(f"{labels[1]} relative to object (mm)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, color=COLORS["grid"], linewidth=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_color("#B8C0CA")


def draw_projection_line(ax, nodes, source_id, target_id, axes, color, width, linestyle) -> None:
    source = nodes[source_id]
    target = nodes[target_id]
    ax.plot(
        [source[axes[0]], target[axes[0]]],
        [source[axes[1]], target[axes[1]]],
        color=color,
        linewidth=width,
        linestyle=linestyle,
        alpha=0.85,
    )


def draw_info(ax, graph: dict, origin_world: np.ndarray) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.96, "Graph verdict", fontsize=13, fontweight="bold", color=COLORS["text"])
    ax.text(
        0.0,
        0.86,
        f"{graph['verdict'].upper()}  confidence={graph['confidence']:.2f}",
        fontsize=13,
        fontweight="bold",
        color=COLORS["unknown"],
    )
    ax.text(0.0, 0.75, "Relations", fontsize=12, fontweight="bold", color=COLORS["text"])
    y = 0.68
    for edge in graph["edges"]:
        state = edge["state"]
        symbol = {"true": "TRUE", "false": "FALSE", "unknown": "UNKNOWN"}[state]
        ax.text(0.0, y, relation_label(edge["id"]), fontsize=9.5, color=COLORS["text"])
        ax.text(0.98, y, symbol, fontsize=9.5, color=COLORS[state], ha="right", fontweight="bold")
        y -= 0.075
    ax.text(0.0, y - 0.01, "Missing evidence", fontsize=12, fontweight="bold", color=COLORS["text"])
    y -= 0.09
    for item in graph["missing_evidence"]:
        ax.text(0.02, y, f"- {item}", fontsize=9.5, color=COLORS["unknown"])
        y -= 0.065
    ax.text(
        0.0,
        0.005,
        "Object-centered coordinates in mm | no RGB or dense point cloud",
        fontsize=7.5,
        color="#56616D",
        va="bottom",
    )


def style_3d_axis(ax) -> None:
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update({"color": COLORS["grid"], "linewidth": 0.7})
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
        axis.pane.set_edgecolor("#C8CFD8")


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLORS["gripper"], marker="o", linewidth=2, label="gripper/contact points"),
        Line2D([0], [0], color=COLORS["object"], marker="o", linewidth=0, label="object center"),
        Line2D([0], [0], color=COLORS["object_axis"], linewidth=4, label="object main axis"),
        Line2D([0], [0], color=COLORS["true"], linewidth=2, label="true relation"),
        Line2D([0], [0], color=COLORS["false"], linewidth=2, linestyle=":", label="false relation"),
        Line2D([0], [0], color=COLORS["unknown"], linewidth=2, linestyle="--", label="unknown relation"),
    ]


def relation_label(edge_id: str) -> str:
    return {
        "object_between_fingers": "between fingers",
        "finger_a_contact": "contact A",
        "finger_b_contact": "contact B",
        "supported_by_table": "supported by",
        "lifted_from_support": "lifted",
        "moves_with_gripper": "moves with",
    }[edge_id]


if __name__ == "__main__":
    main()
