# MapifyPython

MapifyPython visualizes Python structure and possible execution flow without running the inspected code.

## Public browser app

- **Node graph** for modules, classes, functions, imports, containment, and calls.
- **Execution tree** for statement order, conditions, loops, exception paths, and deferred function bodies.
- **Raw JSON** for the complete analysis result.

Both visual views center automatically when opened. Page scrolling does not change graph zoom; use the explicit **Zoom −**, **Zoom +**, and **Center** controls instead. The execution tree initially rolls subprocesses beneath their top-level process: select any `+` node to drill down, select `−` to roll it back up, or use **Roll up** to return to the overview. Either graph can be shown full screen or exported as SVG or high-resolution PNG.

The analyzer runs locally in the visitor's browser through Pyodide. Python source is not uploaded to a server.

The default project URL is:

```text
https://datawzard.github.io/MapifyPython/
```

The page downloads version-pinned Pyodide and D3 assets from jsDelivr when it loads.

## Local command-line use

```bash
python mapify.py path/to/script.py --out map.json
```
