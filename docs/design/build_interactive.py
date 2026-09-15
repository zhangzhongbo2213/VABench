"""Build the standalone radar player: python docs/design/build_interactive.py."""
from __future__ import annotations

import base64
import io
import json
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from build_readme import DATA, INK, MODELS, ROOT, Radar, logo


def main():
    radar = Radar()
    radar.update(0)
    for artist in [radar.polygon, radar.points, radar.index_label, radar.model_label,
                   radar.success_label, *radar.group_values, *radar.progress]:
        artist.set_visible(False)
    radar.icon_ax.set_visible(False)
    stream = io.StringIO()
    radar.fig.savefig(stream, format="svg", metadata={"Date": None})
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    svg = ET.fromstring(stream.getvalue())
    _, _, width, height = map(float, svg.get("viewBox").split())
    svg.attrib.update({"id": "radar", "role": "img", "aria-labelledby": "chart-title chart-description"})
    svg.attrib.pop("width")
    svg.attrib.pop("height")
    title = ET.Element(f"{{{ns}}}title", {"id": "chart-title"})
    title.text = "VA-Bench: nine capability dimensions across the top ten models"
    description = ET.Element(f"{{{ns}}}desc", {"id": "chart-description"})
    description.text = "Select a model below to inspect its capability profile. Scores use a fixed 0–100 scale."
    svg.insert(0, description)
    svg.insert(0, title)
    dynamic = ET.SubElement(svg, f"{{{ns}}}g", {"id": "dynamic-profile"})
    ET.SubElement(dynamic, f"{{{ns}}}polygon", {"id": "profile-polygon", "fill-opacity": ".17",
                  "stroke-width": "2.4", "stroke-linejoin": "round"})
    for i in range(9):
        ET.SubElement(dynamic, f"{{{ns}}}circle", {"id": f"point-{i}", "r": "2.45",
                      "stroke": "white", "stroke-width": "1"})

    def text(name, x, y, size, anchor=None):
        attrs = {"id": name, "x": str(x * width), "y": str((1 - y) * height),
                 "font-family": "DejaVu Sans, Arial, sans-serif", "font-size": str(size),
                 "font-weight": "700", "fill": INK}
        if anchor:
            attrs["dominant-baseline"] = anchor
        ET.SubElement(dynamic, f"{{{ns}}}text", attrs)

    text("model-index", .716, .712, 10.5)
    text("model-name", .763, .650, 16.5, "central")
    text("model-success", .715, .513, 31, "central")
    for i in range(3):
        text(f"group-values-{i}", .716, .433 - i * .081 - .032, 10.5)
    ET.SubElement(dynamic, f"{{{ns}}}image", {"id": "model-logo", "x": str(.716 * width),
                  "y": str((1 - .682) * height), "width": str(.035 * width),
                  "height": str(.057 * height), "preserveAspectRatio": "xMidYMid meet"})
    models = []
    radar.ax.apply_aspect()
    for model in MODELS[:10]:
        points = radar.ax.transData.transform(radar.unit * np.array(model["capabilities"])[:, None])
        points *= 72 / radar.fig.dpi
        points[:, 1] = height - points[:, 1]
        icon = io.BytesIO()
        Image.fromarray(logo(model, 160)).save(icon, format="PNG")
        models.append({"name": model["name"], "label": model["name"].replace("Doubao-seed-", "Doubao ").replace("Gemini-", "Gemini "),
                       "mean": model["mean"], "color": model["color"], "capabilities": model["capabilities"],
                       "points": points.round(5).tolist(), "logo": "data:image/png;base64," + base64.b64encode(icon.getvalue()).decode()})
    plt.close(radar.fig)
    payload = json.dumps({"models": models, "codes": [d["code"] for d in DATA["dimensions"]]}, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "docs/design/radar.html").read_text()
    html = template.replace("<!-- RADAR -->", ET.tostring(svg, encoding="unicode")).replace("/* RADAR_DATA */", payload)
    output = ROOT / "docs/interactive/radar.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n")
    print(f"Standalone interactive radar generated: {output}")


if __name__ == "__main__":
    main()
