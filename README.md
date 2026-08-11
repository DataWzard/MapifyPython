# MapifyPython

MapifyPython visualizes Python structure and possible execution flow without running the inspected code.

## Public browser app

The root `index.html` contains one browser interface with three selectable views:

- **Node graph** for modules, classes, functions, imports, containment, and calls.
- **Execution tree** for statement order, conditions, loops, exception paths, and deferred function bodies.
- **Raw JSON** for the complete analysis result.

The analyzer runs locally in the visitor's browser through Pyodide. Python source is not uploaded to a server.

To publish with GitHub Pages, open **Settings → Pages**, choose **Deploy from a branch**, then select `main` and `/(root)`. The default project URL is:

```text
https://datawzard.github.io/MapifyPython/
```

The page downloads version-pinned Pyodide and D3 assets from jsDelivr when it loads.

## Local command-line use

```bash
python mapify.py path/to/script.py --out map.json
```
