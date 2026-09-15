from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "gripper": "#0891B2",
    "object": "#D946A0",
    "object_axis": "#F59E0B",
    "support": "#22A06B",
    "true": "#1F9D61",
    "false": "#D14343",
    "unknown": "#C58A00",
    "grid": "#D8DEE6",
    "text": "#17202A",
}

NODE_STYLES = {
    "gripper.jaw_center": ("gripper", "D", 60, "jaw base"),
    "gripper.finger_a_base": ("gripper", "s", 48, "finger A base"),
    "gripper.finger_b_base": ("gripper", "s", 48, "finger B base"),
    "gripper.finger_a_inner_tip": ("gripper", "o", 62, "inner tip A"),
    "gripper.finger_b_inner_tip": ("gripper", "o", 62, "inner tip B"),
    "gripper.grasp_center": ("gripper", "*", 115, "grasp center"),
    "object.center": ("object", "o", 85, "pen center"),
    "object.axis_start": ("object_axis", "^", 55, "pen axis -"),
    "object.axis_end": ("object_axis", "^", 55, "pen axis +"),
    "support.plane_anchor": ("support", "s", 52, "support plane"),
}

RELATION_NAMES = {
    "object_between_fingers": "Target enclosed by finger interval",
    "grasp_region_along_object_axis": "Grasp center within pen-axis region",
    "grasp_height_aligned": "Grasp center height aligned",
    "closing_axis_perpendicular_to_object_axis": "Closing axis perpendicular to pen axis",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a verify_pregrasp sparse graph without RGB.")
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(graph, args.output)
    print(f"pregrasp_spatial_graph: {args.output.resolve()}")


def render(graph: dict[str, Any], output: Path) -> None:
    nodes = node_positions_mm(graph)
    fig = plt.figure(figsize=(17, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.35, 1.0, 0.95), height_ratios=(1.0, 1.0))
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 1])
    ax_yz = fig.add_subplot(grid[0, 2])
    ax_info = fig.add_subplot(grid[1, 2])

    draw_3d(ax_3d, graph, nodes)
    draw_projection(ax_xy, graph, nodes, (0, 1), "Top view: X / Y")
    draw_projection(ax_xz, graph, nodes, (0, 2), "Front view: X / Z")
    draw_projection(ax_yz, graph, nodes, (1, 2), "Side view: Y / Z")
    draw_relation_table(ax_info, graph)

    context = graph.get("diagnostic_context", {})
    query = graph.get("query", context.get("query", "verify_pregrasp"))
    access = graph.get("access", "unknown")
    fig.suptitle(
        f"Sparse 3D Pre-grasp Graph | {query} | {access}",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
        y=0.975,
    )
    fig.text(
        0.5,
        0.018,
        "Object-centered coordinates in millimeters. Lines are operational axes and measured offsets, not a dense point cloud.",
        ha="center",
        fontsize=9,
        color="#52606D",
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.065, top=0.92, wspace=0.27, hspace=0.28)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def node_positions_mm(graph: dict[str, Any]) -> dict[str, np.ndarray]:
    world = {
        str(node["id"]): np.asarray(node["position_mean_world_m"], dtype=np.float64)
        for node in graph["nodes"]
    }
    origin = world.get("object.center", world.get("gripper.grasp_center", np.zeros(3, dtype=np.float64)))
    return {node_id: (position - origin) * 1000.0 for node_id, position in world.items()}


def draw_3d(ax, graph: dict[str, Any], nodes: dict[str, np.ndarray]) -> None:
    support = nodes.get("support.plane_anchor")
    if support is not None:
        x = np.linspace(-110, 110, 2)
        y = np.linspace(-150, 150, 2)
        xx, yy = np.meshgrid(x, y)
        zz = np.full_like(xx, support[2])
        ax.plot_surface(xx, yy, zz, color=COLORS["support"], alpha=0.09, linewidth=0)
        ax.plot_wireframe(xx, yy, zz, color=COLORS["support"], alpha=0.25, linewidth=0.7)
    draw_skeleton_3d(ax, nodes)
    draw_offset_3d(ax, graph, nodes)
    draw_nodes_3d(ax, nodes)
    styled_points = [point for node_id, point in nodes.items() if node_id in NODE_STYLES]
    points = np.stack(styled_points) if styled_points else np.zeros((1, 3), dtype=np.float64)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    span = np.maximum(points.max(axis=0) - points.min(axis=0), [150.0, 260.0, 110.0])
    half = span * np.array([0.65, 0.60, 0.78])
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    ax.set_zlim(center[2] - half[2], center[2] + half[2])
    ax.set_box_aspect(half)
    ax.view_init(elev=25, azim=-52)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Operational 3D skeleton", fontsize=14, fontweight="bold", pad=14)
    style_3d(ax)
    ax.legend(handles=legend_handles(), loc="upper left", fontsize=8.5, frameon=True)


def draw_skeleton_3d(ax, nodes: dict[str, np.ndarray]) -> None:
    line_3d(ax, nodes, "object.axis_start", "object.axis_end", COLORS["object_axis"], 5.0, "-")
    line_3d(
        ax,
        nodes,
        "gripper.finger_a_inner_tip",
        "gripper.finger_b_inner_tip",
        COLORS["gripper"],
        4.2,
        "--",
    )
    for base, tip in (
        ("gripper.finger_a_base", "gripper.finger_a_inner_tip"),
        ("gripper.finger_b_base", "gripper.finger_b_inner_tip"),
    ):
        line_3d(ax, nodes, base, tip, COLORS["gripper"], 3.0, "-")


def draw_offset_3d(ax, graph: dict[str, Any], nodes: dict[str, np.ndarray]) -> None:
    edge = edge_map(graph).get("grasp_region_along_object_axis")
    if edge is None:
        return
    if "object.center" not in nodes or "gripper.grasp_center" not in nodes:
        return
    source = nodes["object.center"]
    target = nodes["gripper.grasp_center"]
    color = COLORS[edge_state(edge)]
    ax.plot(
        [source[0], target[0]],
        [source[1], target[1]],
        [source[2], target[2]],
        color=color,
        linewidth=2.7,
        linestyle=":" if edge_state(edge) == "false" else "-",
    )
    midpoint = (source + target) / 2.0
    ax.text(*(midpoint + np.array([3.0, 3.0, 3.0])), "center offset", color=color, fontsize=8)


def draw_nodes_3d(ax, nodes: dict[str, np.ndarray]) -> None:
    for node_id, (color_key, marker, size, label) in NODE_STYLES.items():
        if node_id not in nodes:
            continue
        point = nodes[node_id]
        ax.scatter(*point, s=size, marker=marker, color=COLORS[color_key], edgecolor="white", linewidth=1.0)
        if node_id in {
            "gripper.grasp_center",
            "gripper.finger_a_inner_tip",
            "gripper.finger_b_inner_tip",
            "object.center",
        }:
            ax.text(*(point + np.array([4.0, 4.0, 4.0])), label, fontsize=8, color=COLORS["text"])


def draw_projection(
    ax,
    graph: dict[str, Any],
    nodes: dict[str, np.ndarray],
    axes: tuple[int, int],
    title: str,
) -> None:
    first, second = axes
    support = nodes.get("support.plane_anchor")
    if support is not None and second == 2:
        ax.axhspan(support[2] - 1.5, support[2] + 1.5, color=COLORS["support"], alpha=0.18)
    line_2d(ax, nodes, "object.axis_start", "object.axis_end", axes, COLORS["object_axis"], 4.2, "-")
    line_2d(
        ax,
        nodes,
        "gripper.finger_a_inner_tip",
        "gripper.finger_b_inner_tip",
        axes,
        COLORS["gripper"],
        3.6,
        "--",
    )
    for base, tip in (
        ("gripper.finger_a_base", "gripper.finger_a_inner_tip"),
        ("gripper.finger_b_base", "gripper.finger_b_inner_tip"),
    ):
        line_2d(ax, nodes, base, tip, axes, COLORS["gripper"], 2.5, "-")
    along = edge_map(graph).get("grasp_region_along_object_axis")
    if along is not None and "object.center" in nodes and "gripper.grasp_center" in nodes:
        source = nodes["object.center"]
        target = nodes["gripper.grasp_center"]
        ax.plot(
            [source[first], target[first]],
            [source[second], target[second]],
            color=COLORS[edge_state(along)],
            linewidth=2.2,
            linestyle=":" if edge_state(along) == "false" else "-",
        )
    for node_id in (
        "gripper.finger_a_inner_tip",
        "gripper.finger_b_inner_tip",
        "gripper.grasp_center",
        "object.center",
        "object.axis_start",
        "object.axis_end",
    ):
        if node_id not in nodes:
            continue
        color_key, marker, size, label = NODE_STYLES[node_id]
        point = nodes[node_id]
        ax.scatter(
            point[first],
            point[second],
            s=size * 0.7,
            marker=marker,
            color=COLORS[color_key],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        if node_id in {"gripper.grasp_center", "object.center"}:
            ax.annotate(label, (point[first], point[second]), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel(("X", "Y", "Z")[first] + " relative to pen (mm)")
    ax.set_ylabel(("X", "Y", "Z")[second] + " relative to pen (mm)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, color=COLORS["grid"], linewidth=0.7)
    ax.set_aspect("equal", adjustable="datalim")


def draw_relation_table(ax, graph: dict[str, Any]) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    verdict = str(graph.get("verdict", inferred_verdict(graph)))
    confidence = float(graph.get("confidence", inferred_confidence(graph)))
    verdict_color = COLORS.get(verdict, COLORS["unknown"])
    if verdict == "execute":
        verdict_color = COLORS["true"]
    elif verdict in {"adjust", "reject"}:
        verdict_color = COLORS["false"]
    ax.text(0.0, 0.96, "Decision", fontsize=13, fontweight="bold", color=COLORS["text"])
    ax.text(0.0, 0.87, f"{verdict.upper()}  confidence={confidence:.3f}", fontsize=12, fontweight="bold", color=verdict_color)
    ax.text(0.0, 0.76, "Query relations", fontsize=12, fontweight="bold", color=COLORS["text"])
    y = 0.68
    for edge_id in RELATION_NAMES:
        edge = edge_map(graph).get(edge_id)
        if edge is None:
            continue
        state = edge_state(edge)
        probability = float(edge.get("probability", 0.5))
        ax.text(0.0, y, RELATION_NAMES[edge_id], fontsize=8.8, color=COLORS["text"])
        ax.text(0.98, y, f"{state.upper()}  p={probability:.3f}", fontsize=8.8, color=COLORS[state], ha="right", fontweight="bold")
        y -= 0.055
        measurement = measurement_text(edge_id, edge.get("measurement", {}))
        if measurement:
            ax.text(0.03, y, measurement, fontsize=7.8, color="#5A6773")
            y -= 0.073
        else:
            y -= 0.035
    ax.text(0.0, max(0.02, y), "Green=true, red=false, amber=uncertain", fontsize=8, color="#52606D")


def measurement_text(edge_id: str, value: dict[str, Any]) -> str:
    enclosure_margin = value.get("enclosure_margin_m", value.get("predicted_enclosure_margin_m"))
    if edge_id == "object_between_fingers" and enclosure_margin is not None:
        return f"enclosure margin = {1000.0 * float(enclosure_margin):+.1f} mm"
    axis_offset = value.get("offset_along_object_axis_m", value.get("predicted_axis_offset_m"))
    axis_limit = value.get("allowed_abs_offset_m", value.get("predicted_axis_limit_m"))
    if edge_id == "grasp_region_along_object_axis" and axis_offset is not None:
        return (
            f"axis offset = {1000.0 * float(axis_offset):+.1f} mm; "
            f"limit = {1000.0 * float(axis_limit):.1f} mm"
        )
    height_offset = value.get("vertical_offset_m", value.get("predicted_vertical_offset_m"))
    if edge_id == "grasp_height_aligned" and height_offset is not None:
        height_limit = float(value.get("allowed_abs_offset_m", 0.018))
        return (
            f"height offset = {1000.0 * float(height_offset):+.1f} mm; "
            f"limit = {1000.0 * height_limit:.1f} mm"
        )
    axis_angle = value.get("axis_angle_deg", value.get("predicted_axis_angle_deg"))
    if edge_id == "closing_axis_perpendicular_to_object_axis" and axis_angle is not None:
        return f"axis angle = {float(axis_angle):.1f} deg; target = 90 deg"
    evidence = value.get("view_observability")
    if evidence is not None:
        return f"latest-view observability = {float(evidence):.3f}"
    return ""


def edge_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(edge["id"]): edge for edge in graph.get("edges", [])}


def edge_state(edge: dict[str, Any]) -> str:
    decision_state = str(edge.get("decision_state", ""))
    if decision_state in {"true", "false", "unknown"}:
        return decision_state
    state = str(edge.get("state", "unknown"))
    if state in {"true", "false", "unknown"}:
        return state
    probability = float(edge.get("probability", 0.5))
    return "true" if probability >= 0.8 else "false" if probability <= 0.2 else "unknown"


def inferred_verdict(graph: dict[str, Any]) -> str:
    states = [edge_state(edge) for edge in graph.get("edges", [])]
    if any(state == "false" for state in states):
        return "adjust"
    if states and all(state == "true" for state in states):
        return "execute"
    return "uncertain"


def inferred_confidence(graph: dict[str, Any]) -> float:
    probabilities = [float(edge.get("probability", 0.5)) for edge in graph.get("edges", [])]
    return min(probabilities) if probabilities else 0.0


def line_3d(ax, nodes, source_id, target_id, color, width, linestyle) -> None:
    if source_id not in nodes or target_id not in nodes:
        return
    source, target = nodes[source_id], nodes[target_id]
    ax.plot(
        [source[0], target[0]],
        [source[1], target[1]],
        [source[2], target[2]],
        color=color,
        linewidth=width,
        linestyle=linestyle,
        alpha=0.9,
    )


def line_2d(ax, nodes, source_id, target_id, axes, color, width, linestyle) -> None:
    if source_id not in nodes or target_id not in nodes:
        return
    source, target = nodes[source_id], nodes[target_id]
    ax.plot(
        [source[axes[0]], target[axes[0]]],
        [source[axes[1]], target[axes[1]]],
        color=color,
        linewidth=width,
        linestyle=linestyle,
        alpha=0.9,
    )


def style_3d(ax) -> None:
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update({"color": COLORS["grid"], "linewidth": 0.7})
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
        axis.pane.set_edgecolor("#C8CFD8")


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLORS["gripper"], linewidth=4, linestyle="--", label="gripper closing axis"),
        Line2D([0], [0], color=COLORS["object_axis"], linewidth=5, label="pen main axis"),
        Line2D([0], [0], color=COLORS["object"], marker="o", linewidth=0, label="pen center"),
        Line2D([0], [0], color=COLORS["gripper"], marker="*", linewidth=0, label="grasp center"),
    ]


if __name__ == "__main__":
    main()
