# MapifyPython

MapifyPython statically analyzes Python source and turns it into an interactive SVG execution map. It does **not** run the inspected code.

The map shows top-to-bottom statement order, branches, loops, exception paths, class-body execution, and deferred function bodies. It also emits a JSON catalog of imports, functions, classes, methods, assignments, arguments, source locations, and calls.

## Web interface (GitHub Pages)

The root `index.html` is a complete browser interface. Visitors can paste Python or open a `.py` file and generate a map without sending their source to a server. The analyzer runs locally through Pyodide.

To publish it, open the repository's **Settings → Pages**, select **Deploy from a branch**, then choose `main` and `/(root)`. The project URL will be:

```text
https://datawzard.github.io/MapifyPython/
```

The page requires internet access on first load to download the version-pinned Pyodide runtime.

## Quick start

Python 3.10 or newer is required. No third-party packages are needed.

```bash
python mapify.py example.py --out example-map.html
```

Open `example-map.html` in a browser. To analyze a package recursively:

```bash
python mapify.py path/to/package --out package-map.html
```

To produce machine-readable output:

```bash
python mapify.py example.py --format json --out example-map.json
```

## What the diagram means

- Solid nodes execute as their containing scope is evaluated.
- Dashed nodes are deferred, such as a function body that runs only when called.
- Branches describe possible paths, not a recorded runtime trace.

Static analysis cannot know values from user input, network responses, dynamic imports, reflection, or `eval`. A future runtime-trace layer could supplement the static map for those cases.

## Tests

```bash
python -m unittest discover -s tests -v
```
