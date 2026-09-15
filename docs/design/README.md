# README visuals

Open `../../README.preview.html` in a browser to review the README locally.
The README itself uses GitHub-compatible Markdown, simple HTML, local SVGs and
a looping GIF. The local preview embeds the interactive radar player, which
autoplays, pauses at the current frame, and lets visitors select a model using
the ten gray bars. Selecting a model pauses the player; Play resumes from there.

To regenerate figures and the preview:

```bash
python docs/design/build_readme.py --animate
python docs/design/build_interactive.py
```

Run the command from the repository root with `matplotlib`, `numpy`, `Pillow`,
`markdown-it-py` and the `ffmpeg` executable installed. Without `--animate`, the
script updates the static figures, README and preview while keeping the existing
animation. No model endpoint or simulator is used.

`docs/data/results.json` contains the plotted values and author metadata.
Success values are recomputed from the three source runs; capabilities use the
manuscript's pooled nine-dimension table. See `docs/data/SOURCES.md` for the exact
aggregation and annotation conditions. Logo sources and their license are in
`docs/media/logos/`.

The animation holds each recorded profile for 1.4 seconds and transitions to
the next in 0.9 seconds at 20 frames per second. Radar coordinates ease between
profiles on a fixed scale; numeric results remain at the recorded endpoints,
and switch at the transition midpoint. The last profile transitions back to the first.
Three colored sectors group the nine axes; each group name follows the outer arc
around its three capability labels.

`docs/interactive/radar.html` is a standalone page with inline SVG, model data,
logos and JavaScript. It works offline when opened directly in a browser, and
the labels remain sharp when enlarged. The generator reuses the chart geometry
and result data from `build_readme.py`. Gray bars support clicks and keyboard
navigation with arrow keys, Home and End; the pause control supports the keyboard.

GitHub README pages do not execute the player JavaScript. The README therefore
keeps its looping GIF and links to the downloadable HTML player. Hosting this
file on a static website would allow the link to open the interactive page online.
