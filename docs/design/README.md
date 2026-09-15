# README visuals

Open `../../README.preview.html` in a browser to review the README locally.
The README itself uses GitHub-compatible Markdown, simple HTML, local SVGs and
a looping GIF. The preview adds local pause/restart controls.

To regenerate figures and the preview:

```bash
python docs/design/build_readme.py --animate
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
with a fade between model cards. The last profile transitions back to the first.
