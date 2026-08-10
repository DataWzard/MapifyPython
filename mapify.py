#!/usr/bin/env python3
"""Map Python source into a JSON model and an interactive HTML/SVG diagram."""

from __future__ import annotations

import argparse
import ast
import html
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class FlowNode:
    id: str
    kind: str
    label: str
    lineno: int | None = None
    end_lineno: int | None = None
    detail: str | None = None
    execution: str = "immediate"
    children: list["FlowNode"] = field(default_factory=list)


def _text(node: ast.AST | None, limit: int = 100) -> str:
    if node is None:
        return ""
    try:
        value = ast.unparse(node).replace("\n", " ")
    except Exception:
        value = node.__class__.__name__
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _location(node: ast.AST) -> dict[str, int | None]:
    return {
        "lineno": getattr(node, "lineno", None),
        "end_lineno": getattr(node, "end_lineno", None),
        "col": getattr(node, "col_offset", None),
        "end_col": getattr(node, "end_col_offset", None),
    }


class FlowBuilder:
    """Turn statements into an ordered tree without executing user code."""

    def __init__(self) -> None:
        self._next_id = 0

    def node(
        self,
        kind: str,
        label: str,
        source: ast.AST | None = None,
        *,
        detail: str | None = None,
        execution: str = "immediate",
        children: Iterable[FlowNode] = (),
    ) -> FlowNode:
        result = FlowNode(
            id=f"node-{self._next_id}",
            kind=kind,
            label=label,
            lineno=getattr(source, "lineno", None),
            end_lineno=getattr(source, "end_lineno", None),
            detail=detail,
            execution=execution,
            children=list(children),
        )
        self._next_id += 1
        return result

    def block(self, statements: list[ast.stmt], execution: str = "immediate") -> list[FlowNode]:
        return [self.statement(statement, execution) for statement in statements]

    def branch(self, label: str, statements: list[ast.stmt], source: ast.AST, execution: str) -> FlowNode:
        return self.node("branch", label, source, execution=execution, children=self.block(statements, execution))

    def statement(self, stmt: ast.stmt, execution: str = "immediate") -> FlowNode:
        deferred = "deferred"
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async function" if isinstance(stmt, ast.AsyncFunctionDef) else "function"
            body = self.branch("body (when called)", stmt.body, stmt, deferred)
            return self.node("function", f"Define {prefix} {stmt.name}", stmt, execution=execution, children=[body])
        if isinstance(stmt, ast.ClassDef):
            body = self.branch("class body", stmt.body, stmt, execution)
            return self.node("class", f"Define class {stmt.name}", stmt, execution=execution, children=[body])
        if isinstance(stmt, ast.If):
            children = [self.branch("true", stmt.body, stmt, execution)]
            if stmt.orelse:
                children.append(self.branch("false / elif", stmt.orelse, stmt, execution))
            return self.node("condition", f"If {_text(stmt.test)}", stmt, execution=execution, children=children)
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            children = [self.branch("each iteration", stmt.body, stmt, execution)]
            if stmt.orelse:
                children.append(self.branch("loop completed", stmt.orelse, stmt, execution))
            return self.node("loop", f"For {_text(stmt.target)} in {_text(stmt.iter)}", stmt, execution=execution, children=children)
        if isinstance(stmt, ast.While):
            children = [self.branch("while true", stmt.body, stmt, execution)]
            if stmt.orelse:
                children.append(self.branch("loop completed", stmt.orelse, stmt, execution))
            return self.node("loop", f"While {_text(stmt.test)}", stmt, execution=execution, children=children)
        if isinstance(stmt, ast.Try):
            children = [self.branch("try", stmt.body, stmt, execution)]
            for handler in stmt.handlers:
                name = _text(handler.type) or "Exception"
                children.append(self.branch(f"except {name}", handler.body, handler, execution))
            if stmt.orelse:
                children.append(self.branch("else", stmt.orelse, stmt, execution))
            if stmt.finalbody:
                children.append(self.branch("finally", stmt.finalbody, stmt, execution))
            return self.node("try", "Try / except", stmt, execution=execution, children=children)
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            contexts = ", ".join(_text(item.context_expr) for item in stmt.items)
            return self.node("with", f"With {contexts}", stmt, execution=execution, children=self.block(stmt.body, execution))
        if isinstance(stmt, ast.Match):
            cases = []
            for case in stmt.cases:
                label = f"case {_text(case.pattern)}"
                if case.guard:
                    label += f" if {_text(case.guard)}"
                cases.append(self.branch(label, case.body, case.pattern, execution))
            return self.node("condition", f"Match {_text(stmt.subject)}", stmt, execution=execution, children=cases)

        labels: list[tuple[type[ast.AST], str, str]] = [
            (ast.Import, "import", "Import"), (ast.ImportFrom, "import", "Import"),
            (ast.Assign, "assignment", "Set"), (ast.AnnAssign, "assignment", "Set"),
            (ast.AugAssign, "assignment", "Update"), (ast.Return, "return", "Return"),
            (ast.Raise, "raise", "Raise"), (ast.Assert, "assert", "Assert"),
            (ast.Break, "jump", "Break"), (ast.Continue, "jump", "Continue"),
            (ast.Pass, "pass", "Pass"), (ast.Delete, "statement", "Delete"),
            (ast.Global, "scope", "Global"), (ast.Nonlocal, "scope", "Nonlocal"),
        ]
        for cls, kind, prefix in labels:
            if isinstance(stmt, cls):
                return self.node(kind, f"{prefix}: {_text(stmt)}", stmt, execution=execution)
        return self.node("call" if isinstance(stmt, ast.Expr) else "statement", _text(stmt), stmt, execution=execution)


def _arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, str | None]]:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    output = []
    posonly_count = len(args.posonlyargs)
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        output.append({
            "name": argument.arg,
            "kind": "posonly" if index < posonly_count else "positional",
            "annotation": _text(argument.annotation) or None,
            "default": _text(default) or None,
        })
    if args.vararg:
        output.append({"name": args.vararg.arg, "kind": "vararg", "annotation": _text(args.vararg.annotation) or None, "default": None})
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        output.append({"name": argument.arg, "kind": "kwonly", "annotation": _text(argument.annotation) or None, "default": _text(default) or None})
    if args.kwarg:
        output.append({"name": args.kwarg.arg, "kind": "kwarg", "annotation": _text(args.kwarg.annotation) or None, "default": None})
    return output


def _calls(node: ast.AST) -> list[str]:
    found: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _text(child.func)
            if name and name not in found:
                found.append(name)
    return found


def _catalog(tree: ast.Module) -> dict[str, Any]:
    imports, functions, classes, assignments = [], [], [], []
    for node in tree.body:  # only true module-level declarations
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"kind": "import", "name": alias.name, "asname": alias.asname, "location": _location(node)})
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({"kind": "from", "module": node.module, "level": node.level, "name": alias.name, "asname": alias.asname, "location": _location(node)})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name, "async": isinstance(node, ast.AsyncFunctionDef),
                "arguments": _arguments(node), "returns": _text(node.returns) or None,
                "decorators": [_text(item) for item in node.decorator_list],
                "docstring": ast.get_docstring(node), "calls": _calls(node), "location": _location(node),
            })
        elif isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({"name": child.name, "async": isinstance(child, ast.AsyncFunctionDef), "arguments": _arguments(child), "returns": _text(child.returns) or None, "calls": _calls(child), "location": _location(child)})
            classes.append({"name": node.name, "bases": [_text(base) for base in node.bases], "decorators": [_text(item) for item in node.decorator_list], "docstring": ast.get_docstring(node), "methods": methods, "location": _location(node)})
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append({"source": _text(node), "location": _location(node)})
    return {"imports": imports, "functions": functions, "classes": classes, "assignments": assignments}


def map_python_source(source: str, filename: str = "<string>") -> dict[str, Any]:
    tree = ast.parse(source, filename=filename, type_comments=True)
    builder = FlowBuilder()
    root = builder.node("module", f"Module {Path(filename).name}", children=builder.block(tree.body))
    return {
        "schema_version": 1,
        "source": filename,
        "docstring": ast.get_docstring(tree),
        "catalog": _catalog(tree),
        "flow": asdict(root),
    }


def map_python_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return map_python_source(path.read_text(encoding="utf-8-sig"), str(path))


def map_python_package(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    modules = []
    for path in sorted(root.rglob("*.py")):
        if any(part.startswith(".") or part in {"__pycache__", ".venv", "venv"} for part in path.parts):
            continue
        try:
            modules.append(map_python_file(path))
        except (SyntaxError, UnicodeError) as error:
            modules.append({"source": str(path), "error": f"{type(error).__name__}: {error}"})
    return {"schema_version": 1, "root": str(root), "modules": modules}


def map_to_json(mapped: dict[str, Any], *, indent: int = 2) -> str:
    return json.dumps(mapped, indent=indent, ensure_ascii=False)


def map_to_html(mapped: dict[str, Any], *, title: str = "MapifyPython") -> str:
    payload = json.dumps(mapped, ensure_ascii=False).replace("<", "\\u003c")
    safe_title = html.escape(title)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{safe_title}</title><style>
:root{{--bg:#0b1020;--panel:#11182b;--text:#e8eefc;--muted:#94a3b8;--edge:#52627d;--accent:#65d1ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;padding:12px 18px;background:#0b1020e8;border-bottom:1px solid #26324a;display:flex;gap:16px;align-items:center}}
header strong{{font-size:16px}} header span{{color:var(--muted)}} button{{background:#1d2942;color:var(--text);border:1px solid #3a4b6c;border-radius:6px;padding:6px 10px;cursor:pointer}}
#canvas{{overflow:auto;height:calc(100vh - 54px)}} svg{{min-width:100%;min-height:100%}} .edge{{stroke:var(--edge);stroke-width:1.5;fill:none}}
.node rect{{fill:var(--panel);stroke:#415373;stroke-width:1.5;rx:8}} .node.immediate rect{{stroke:var(--accent)}} .node.deferred rect{{stroke-dasharray:5 4}}
.node text{{fill:var(--text);font:13px ui-monospace,monospace}} .node .meta{{fill:var(--muted);font-size:11px}} .node:hover rect{{fill:#18233b}}
</style></head><body><header><strong>{safe_title}</strong><span id="source"></span><button id="fit">Fit</button><span>solid = immediate · dashed = deferred</span></header>
<div id="canvas"><svg id="svg"><g id="scene"></g></svg></div><script>
const DATA={payload}; const root=DATA.flow || {{kind:'package',label:'Package',children:(DATA.modules||[]).filter(x=>x.flow).map(x=>x.flow)}};
document.getElementById('source').textContent=DATA.source||DATA.root||'';
const NS='http://www.w3.org/2000/svg', scene=document.getElementById('scene'), svg=document.getElementById('svg');
let row=0, nodes=[]; function walk(n,depth=0,parent=null){{let item={{n,depth,parent,x:40+depth*285,y:35+row++*86}};nodes.push(item);(n.children||[]).forEach(c=>walk(c,depth+1,item));}} walk(root);
for(const item of nodes){{if(!item.parent)continue;let p=item.parent;let path=document.createElementNS(NS,'path');path.setAttribute('class','edge');path.setAttribute('d',`M${{p.x+230}},${{p.y+28}} C${{p.x+258}},${{p.y+28}} ${{item.x-28}},${{item.y+28}} ${{item.x}},${{item.y+28}}`);scene.appendChild(path);}}
for(const item of nodes){{let g=document.createElementNS(NS,'g');g.setAttribute('class',`node ${{item.n.execution||'immediate'}}`);g.setAttribute('transform',`translate(${{item.x}} ${{item.y}})`);let rect=document.createElementNS(NS,'rect');rect.setAttribute('width',230);rect.setAttribute('height',56);g.appendChild(rect);let label=document.createElementNS(NS,'text');label.setAttribute('x',12);label.setAttribute('y',23);label.textContent=(item.n.label||'').slice(0,31);g.appendChild(label);let meta=document.createElementNS(NS,'text');meta.setAttribute('class','meta');meta.setAttribute('x',12);meta.setAttribute('y',43);meta.textContent=`${{item.n.kind}}${{item.n.lineno?' · line '+item.n.lineno:''}}`;g.appendChild(meta);let tip=document.createElementNS(NS,'title');tip.textContent=item.n.label+(item.n.detail?'\n'+item.n.detail:'');g.appendChild(tip);scene.appendChild(g);}}
const width=Math.max(700,...nodes.map(x=>x.x+270)),height=Math.max(500,row*86+40);svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`);svg.setAttribute('width',width);svg.setAttribute('height',height);
document.getElementById('fit').onclick=()=>{{svg.setAttribute('width','100%');svg.setAttribute('height','100%')}};
</script></body></html>'''


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize Python structure and execution order")
    parser.add_argument("path", help="Python file or directory")
    parser.add_argument("--format", choices=("json", "html"), default="html")
    parser.add_argument("--out", help="Output path (default: stdout for JSON, mapify.html for HTML)")
    args = parser.parse_args(argv)
    target = Path(args.path)
    if not target.exists():
        parser.error(f"Path not found: {target}")
    try:
        mapped = map_python_package(target) if target.is_dir() else map_python_file(target)
    except (SyntaxError, UnicodeError) as error:
        print(f"mapify: {error}", file=sys.stderr)
        return 2
    content = map_to_json(mapped) if args.format == "json" else map_to_html(mapped, title=f"MapifyPython — {target.name}")
    output = Path(args.out) if args.out else (Path("mapify.html") if args.format == "html" else None)
    if output:
        output.write_text(content, encoding="utf-8")
        print(output.resolve())
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
