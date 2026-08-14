# PyMap

PyMap visualizes Python structure and possible execution flow without running the inspected code.

## Public browser app

- **Node graph** for modules, classes, functions, imports, containment, and calls.
- **Execution flow** as a vertically scrolling, collapsible outline for statement order, conditions, loops, exception paths, and deferred function bodies.
- **Raw JSON** for the complete result analysis.

The node graph centers automatically when opened. The execution flow keeps every step at a fixed readable size and scrolls vertically, so large files never shrink the entire diagram to fit. Page scrolling does not change graph zoom; use the explicit **Zoom −**, **Zoom +**, and **Center** controls instead. Subprocesses initially roll beneath their top-level process: select any `+` step to drill down, select `−` to roll it back up, or use **Roll up** to return to the module overview. Either view can be shown full screen or exported as SVG or high-resolution PNG.

The analyzer runs locally in the visitor's browser through Pyodide. Python source is not uploaded to a server.

The public app URL is:

```text
https://stacksanalytics.us/PyMap/
```

The page downloads version-pinned Pyodide and D3 assets from jsDelivr when it loads.

## Security and privacy

PyMap is a static browser-only application. It has no accounts, backend API, database, analytics, or cookies, and inspected source code remains in the browser. CDN requests still expose ordinary network metadata such as an IP address and user agent to jsDelivr.

Run the local security suite with:

```bash
python scripts/security_audit.py
python -m unittest discover -s tests -v
```

See [SECURITY.md](SECURITY.md) for vulnerability reporting and [PRIVACY.md](PRIVACY.md) for the privacy summary.

## Local command-line use

```bash
python pymap.py path/to/script.py --out map.json
```
