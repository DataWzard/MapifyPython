#!/usr/bin/env python3
"""
PyMap — a Python AST and execution-flow visualizer inspired by MapifyJSON.

Goals (assumptions, since the original repo context isn't available):
- Parse a Python module or package into a rich, JSON-serializable map.
- Capture modules, imports, classes, methods, functions, arguments, returns,
  decorators, attributes, assignments, constants, docstrings, and a light call graph.
- Preserve source locations (lineno/col), file paths, and nesting relationships.
- Offer a simple CLI: point at a file or directory, get a JSON map out.

This file is fully self-contained and has zero third‑party deps.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

# ---------------------------
# Utility helpers
# ---------------------------

def _safe_get_id(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_safe_get_id(node.value)}.{node.attr}" if _safe_get_id(node.value) else node.attr
    if isinstance(node, ast.alias):
        return node.asname or node.name
    return None


def _expr_to_str(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)  # Python 3.9+
    except Exception:
        # Fallback repr for older / corner nodes
        return node.__class__.__name__


def _annotation(node: Optional[ast.AST]) -> Optional[str]:
    return _expr_to_str(node)


def _default_value(node: Optional[ast.AST]) -> Optional[str]:
    return _expr_to_str(node)


def _docstring(node: Union[ast.AST, List[ast.stmt], None]) -> Optional[str]:
    try:
        return ast.get_docstring(node) if node else None
    except Exception:
        return None


def _loc(node: ast.AST) -> Dict[str, Optional[int]]:
    lineno = getattr(node, 'lineno', None)
    end_lineno = getattr(node, 'end_lineno', None)
    col = getattr(node, 'col_offset', None)
    end_col = getattr(node, 'end_col_offset', None)
    return {
        "lineno": int(lineno) if lineno is not None else None,
        "end_lineno": int(end_lineno) if end_lineno is not None else None,
        "col": int(col) if col is not None else None,
        "end_col": int(end_col) if end_col is not None else None,
    }


# ---------------------------
# Data containers (JSON‑friendly)
# ---------------------------

@dataclass
class ImportMap:
    kind: str  # "import" or "from"
    module: Optional[str]  # for from‑imports
    name: str
    asname: Optional[str]
    location: Dict[str, Optional[int]]


@dataclass
class ArgumentMap:
    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None
    kind: str = "positional"  # positional, vararg, kwonly, kwarg, posonly


@dataclass
class FunctionMap:
    name: str
    qualname: str
    decorators: List[str]
    returns: Optional[str]
    args: List[ArgumentMap]
    docstring: Optional[str]
    location: Dict[str, Optional[int]]
    calls: List[str] = field(default_factory=list)  # simple call graph (callee names)
    is_async: bool = False
    is_method: bool = False
    visibility: str = "public"  # public/protected/private via naming convention


@dataclass
class AssignmentMap:
    targets: List[str]
    value: Optional[str]
    annotation: Optional[str]
    is_class_attr: bool
    is_module_const: bool
    location: Dict[str, Optional[int]]


@dataclass
class ClassMap:
    name: str
    qualname: str
    bases: List[str]
    decorators: List[str]
    docstring: Optional[str]
    location: Dict[str, Optional[int]]
    methods: List[FunctionMap] = field(default_factory=list)
    attributes: List[AssignmentMap] = field(default_factory=list)
    visibility: str = "public"


@dataclass
class ModuleMap:
    path: str
    module: str
    docstring: Optional[str]
    imports: List[ImportMap] = field(default_factory=list)
    classes: List[ClassMap] = field(default_factory=list)
    functions: List[FunctionMap] = field(default_factory=list)
    assignments: List[AssignmentMap] = field(default_factory=list)


@dataclass
class FlowNode:
    """A statement or branch in the static execution-flow tree."""
    id: str
    kind: str
    label: str
    location: Dict[str, Optional[int]]
    execution: str = "immediate"
    children: List["FlowNode"] = field(default_factory=list)


class FlowBuilder:
    """Build an ordered control-flow tree without executing the source."""

    def __init__(self) -> None:
        self._counter = 0

    def _node(
        self,
        kind: str,
        label: str,
        source: ast.AST,
        *,
        execution: str = "immediate",
        children: Iterable[FlowNode] = (),
    ) -> FlowNode:
        node = FlowNode(
            id=f"flow-{self._counter}",
            kind=kind,
            label=label,
            location=_loc(source),
            execution=execution,
            children=list(children),
        )
        self._counter += 1
        return node

    def _block(self, statements: List[ast.stmt], execution: str) -> List[FlowNode]:
        return [self._statement(statement, execution) for statement in statements]

    def _branch(self, label: str, statements: List[ast.stmt], source: ast.AST, execution: str) -> FlowNode:
        return self._node("branch", label, source, execution=execution, children=self._block(statements, execution))

    def build(self, tree: ast.Module, filename: str) -> FlowNode:
        return self._node("module", f"Module {Path(filename).name}", tree, children=self._block(tree.body, "immediate"))

    def _statement(self, statement: ast.stmt, execution: str) -> FlowNode:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async function" if isinstance(statement, ast.AsyncFunctionDef) else "function"
            body = self._branch("body (when called)", statement.body, statement, "deferred")
            return self._node("function", f"Define {prefix} {statement.name}", statement, execution=execution, children=[body])
        if isinstance(statement, ast.ClassDef):
            body = self._branch("class body", statement.body, statement, execution)
            return self._node("class", f"Define class {statement.name}", statement, execution=execution, children=[body])
        if isinstance(statement, ast.If):
            children = [self._branch("true", statement.body, statement, execution)]
            if statement.orelse:
                children.append(self._branch("false / elif", statement.orelse, statement, execution))
            return self._node("condition", f"If {_expr_to_str(statement.test)}", statement, execution=execution, children=children)
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            children = [self._branch("each iteration", statement.body, statement, execution)]
            if statement.orelse:
                children.append(self._branch("loop completed", statement.orelse, statement, execution))
            return self._node("loop", f"For {_expr_to_str(statement.target)} in {_expr_to_str(statement.iter)}", statement, execution=execution, children=children)
        if isinstance(statement, ast.While):
            children = [self._branch("while true", statement.body, statement, execution)]
            if statement.orelse:
                children.append(self._branch("loop completed", statement.orelse, statement, execution))
            return self._node("loop", f"While {_expr_to_str(statement.test)}", statement, execution=execution, children=children)
        if isinstance(statement, ast.Try):
            children = [self._branch("try", statement.body, statement, execution)]
            for handler in statement.handlers:
                children.append(self._branch(f"except {_expr_to_str(handler.type) or 'Exception'}", handler.body, handler, execution))
            if statement.orelse:
                children.append(self._branch("else", statement.orelse, statement, execution))
            if statement.finalbody:
                children.append(self._branch("finally", statement.finalbody, statement, execution))
            return self._node("try", "Try / except", statement, execution=execution, children=children)
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            contexts = ", ".join(_expr_to_str(item.context_expr) or "context" for item in statement.items)
            return self._node("with", f"With {contexts}", statement, execution=execution, children=self._block(statement.body, execution))
        if isinstance(statement, ast.Match):
            cases = []
            for case in statement.cases:
                label = f"case {_expr_to_str(case.pattern)}"
                if case.guard:
                    label += f" if {_expr_to_str(case.guard)}"
                cases.append(self._branch(label, case.body, case.pattern, execution))
            return self._node("condition", f"Match {_expr_to_str(statement.subject)}", statement, execution=execution, children=cases)

        kind = "statement"
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            kind = "import"
        elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            kind = "assignment"
        elif isinstance(statement, ast.Expr):
            kind = "call" if isinstance(statement.value, ast.Call) else "expression"
        elif isinstance(statement, ast.Return):
            kind = "return"
        elif isinstance(statement, ast.Raise):
            kind = "raise"
        elif isinstance(statement, (ast.Break, ast.Continue)):
            kind = "jump"
        label = (_expr_to_str(statement) or statement.__class__.__name__).replace("\n", " ")
        if len(label) > 110:
            label = label[:109] + "…"
        return self._node(kind, label, statement, execution=execution)


# ---------------------------
# Core visitor
# ---------------------------

class _CallCollector(ast.NodeVisitor):
    """Collect simple names of called functions/methods within a function body."""
    def __init__(self) -> None:
        self.calls: List[str] = []

    def visit_Call(self, node: ast.Call) -> Any:
        name = _safe_get_id(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)


def _visibility(name: str) -> str:
    if name.startswith('__') and not name.endswith('__'):
        return "private"
    if name.startswith('_'):
        return "protected"
    return "public"


class Mapify(ast.NodeVisitor):
    """AST -> JSON‑serializable Python objects."""

    def __init__(self, file_path: Union[str, Path], module_name: Optional[str] = None):
        self.file_path = str(file_path)
        self.module_name = module_name or self._infer_module_name(self.file_path)
        self.module_map = ModuleMap(path=self.file_path, module=self.module_name, docstring=None)
        self._class_stack: List[ClassMap] = []

    # ------------- module level -------------
    def build(self, tree: ast.AST) -> ModuleMap:
        self.module_map.docstring = _docstring(tree)
        self.visit(tree)
        return self.module_map

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.module_map.imports.append(
                ImportMap(kind="import", module=None, name=alias.name, asname=alias.asname, location=_loc(node))
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        mod = node.module or ""
        for alias in node.names:
            self.module_map.imports.append(
                ImportMap(kind="from", module=mod, name=alias.name, asname=alias.asname, location=_loc(node))
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # module‑level function
        self._handle_function(node, is_method=False, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._handle_function(node, is_method=False, is_async=True)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        cls = ClassMap(
            name=node.name,
            qualname=self._qualname(node.name),
            bases=[_expr_to_str(b) for b in node.bases],
            decorators=[_expr_to_str(d) for d in node.decorator_list],
            docstring=_docstring(node),
            location=_loc(node),
            visibility=_visibility(node.name),
        )
        self._class_stack.append(cls)
        self.generic_visit(node)
        self._class_stack.pop()
        self.module_map.classes.append(cls)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        target = _expr_to_str(node.target)
        if target is None:
            return
        asg = AssignmentMap(
            targets=[target],
            value=_default_value(node.value),
            annotation=_annotation(node.annotation),
            is_class_attr=bool(self._class_stack),
            is_module_const=not self._class_stack and target.isupper(),
            location=_loc(node),
        )
        self._add_assignment(asg)

    def visit_Assign(self, node: ast.Assign) -> Any:
        targets = [t for t in node.targets]
        asg = AssignmentMap(
            targets=[_expr_to_str(t) or "<unknown>" for t in targets],
            value=_default_value(node.value),
            annotation=None,
            is_class_attr=bool(self._class_stack),
            is_module_const=not self._class_stack and all((getattr(t, 'id', '') or '').isupper() for t in targets if isinstance(t, ast.Name)),
            location=_loc(node),
        )
        self._add_assignment(asg)

    # ------------- helpers -------------
    def _handle_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], *, is_method: bool, is_async: bool) -> None:
        # If inside a class, this is a method
        if self._class_stack:
            is_method = True
        args = self._map_args(node.args)
        deco = [_expr_to_str(d) for d in node.decorator_list]
        fc = _CallCollector()
        for stmt in node.body:
            fc.visit(stmt)
        fn = FunctionMap(
            name=node.name,
            qualname=self._qualname(node.name),
            decorators=deco,
            returns=_annotation(node.returns),
            args=args,
            docstring=_docstring(node),
            location=_loc(node),
            calls=fc.calls,
            is_async=is_async,
            is_method=is_method,
            visibility=_visibility(node.name),
        )
        if self._class_stack:
            self._class_stack[-1].methods.append(fn)
        else:
            self.module_map.functions.append(fn)

    def _map_args(self, a: ast.arguments) -> List[ArgumentMap]:
        out: List[ArgumentMap] = []
        # posonly (3.8+)
        for i, arg in enumerate(getattr(a, 'posonlyargs', [])):
            out.append(ArgumentMap(name=arg.arg, annotation=_annotation(arg.annotation), kind="posonly"))
        # regular args
        defaults_iter = iter([None] * (len(a.args) - len(a.defaults)) + list(a.defaults))
        for arg in a.args:
            out.append(ArgumentMap(name=arg.arg, annotation=_annotation(arg.annotation), default=_default_value(next(defaults_iter)), kind="positional"))
        # vararg *args
        if a.vararg:
            out.append(ArgumentMap(name=a.vararg.arg, annotation=_annotation(a.vararg.annotation), kind="vararg"))
        # kwonly args
        kw_defaults = a.kw_defaults or []
        for i, arg in enumerate(a.kwonlyargs):
            default = _default_value(kw_defaults[i]) if i < len(kw_defaults) else None
            out.append(ArgumentMap(name=arg.arg, annotation=_annotation(arg.annotation), default=default, kind="kwonly"))
        # kwarg **kwargs
        if a.kwarg:
            out.append(ArgumentMap(name=a.kwarg.arg, annotation=_annotation(a.kwarg.annotation), kind="kwarg"))
        return out

    def _add_assignment(self, asg: AssignmentMap) -> None:
        if self._class_stack:
            self._class_stack[-1].attributes.append(asg)
        else:
            self.module_map.assignments.append(asg)

    def _qualname(self, name: str) -> str:
        if self._class_stack:
            return f"{self.module_name}.{self._class_stack[-1].name}.{name}"
        return f"{self.module_name}.{name}"

    def _infer_module_name(self, path: str) -> str:
        p = Path(path)
        stem = p.stem
        # crude, yet practical: package/module dotted name if under a package root
        return stem


# ---------------------------
# Public API
# ---------------------------

def analyze_python_source(source: str, filename: str = "snippet.py") -> Dict[str, Any]:
    """Return both the structure catalog and hierarchical execution flow."""
    tree = ast.parse(source, filename=filename, type_comments=True)
    module_map = Mapify(filename, module_name=Path(filename).stem).build(tree)
    flow = FlowBuilder().build(tree, filename)
    return {
        "schema_version": 1,
        "source": filename,
        "catalog": asdict(module_map),
        "flow": asdict(flow),
    }


def map_analysis_to_json(analysis: Dict[str, Any], *, indent: Optional[int] = None) -> str:
    return json.dumps(analysis, indent=indent, ensure_ascii=False)

def map_python_file(path: Union[str, Path]) -> ModuleMap:
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path), type_comments=True)
    visitor = Mapify(path)
    return visitor.build(tree)


def map_python_package(root: Union[str, Path]) -> Dict[str, Any]:
    root = Path(root)
    results: Dict[str, Any] = {
        "root": str(root),
        "modules": [],
    }
    for py in root.rglob("*.py"):
        if py.name.startswith("."):
            continue
        try:
            mod_map = map_python_file(py)
            results["modules"].append(asdict(mod_map))
        except SyntaxError as e:
            results["modules"].append({
                "path": str(py),
                "error": f"SyntaxError: {e}",
            })
    return results


def map_to_json(obj: Union[ModuleMap, Dict[str, Any]], *, indent: int = 2) -> str:
    if isinstance(obj, ModuleMap):
        return json.dumps(asdict(obj), indent=indent, ensure_ascii=False)
    return json.dumps(obj, indent=indent, ensure_ascii=False)


# ---------------------------
# CLI
# ---------------------------

def _cli(argv: List[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description="PyMap: Python AST -> JSON map")
    p.add_argument("path", help="Python file or package directory to map")
    p.add_argument("--out", dest="out", help="Output JSON file (default: print to stdout)")
    p.add_argument("--package", action="store_true", help="Treat path as a package/directory and recurse")
    args = p.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        p.error(f"Path not found: {target}")

    if args.package or target.is_dir():
        mapped = map_python_package(target)
        js = map_to_json(mapped)
    else:
        mapped = map_python_file(target)
        js = map_to_json(mapped)

    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
    else:
        print(js)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
