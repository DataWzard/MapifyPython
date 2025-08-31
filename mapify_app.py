#!/usr/bin/env python3
"""
Streamlit app for MapifyPython — paste code → visual map (Graph + JSON Tree ala MapifyJSON).

Usage
-----
1) pip install streamlit networkx pyvis
2) Place this file somewhere
3) streamlit run mapify_app.py
4) Paste one or more Python files (optionally name them), choose edges, and view the graph or JSON tree.

Notes
-----
- AST parsing only; no runtime execution.
- Graph = call & containment edges (pyvis). JSON Tree = hierarchical view using jsontr.ee.js (if present).
- No external network calls; jsontr.ee.js is loaded from local file if present.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
import networkx as nx
from pyvis.network import Network

# ------------------------------------
# Minimal AST -> Map (embedded from mapify)
# ------------------------------------

def _safe_get_id(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _safe_get_id(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.alias):
        return node.asname or node.name
    return None


def _expr_to_str(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _docstring(node: Union[ast.AST, List[ast.stmt], None]) -> Optional[str]:
    try:
        return ast.get_docstring(node) if node else None
    except Exception:
        return None


def _loc(node: ast.AST) -> Dict[str, Optional[int]]:
    return {
        "lineno": getattr(node, 'lineno', None),
        "end_lineno": getattr(node, 'end_lineno', None),
        "col": getattr(node, 'col_offset', None),
        "end_col": getattr(node, 'end_col_offset', None),
    }

@dataclass
class ImportMap:
    kind: str
    module: Optional[str]
    name: str
    asname: Optional[str]
    location: Dict[str, Optional[int]]

@dataclass
class ArgumentMap:
    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None
    kind: str = "positional"

@dataclass
class FunctionMap:
    name: str
    qualname: str
    decorators: List[str]
    returns: Optional[str]
    args: List[ArgumentMap]
    docstring: Optional[str]
    location: Dict[str, Optional[int]]
    calls: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    visibility: str = "public"

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

class _CallCollector(ast.NodeVisitor):
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
    def __init__(self, file_path: str, module_name: Optional[str] = None):
        self.file_path = file_path
        self.module_name = module_name or file_path
        self.module_map = ModuleMap(path=file_path, module=self.module_name, docstring=None)
        self._class_stack: List[ClassMap] = []
    def build(self, tree: ast.AST) -> ModuleMap:
        self.module_map.docstring = _docstring(tree)
        self.visit(tree)
        return self.module_map
    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.module_map.imports.append(ImportMap(kind="import", module=None, name=alias.name, asname=alias.asname, location=_loc(node)))
    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        mod = node.module or ""
        for alias in node.names:
            self.module_map.imports.append(ImportMap(kind="from", module=mod, name=alias.name, asname=alias.asname, location=_loc(node)))
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._handle_function(node, is_async=False)
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._handle_function(node, is_async=True)
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
            value=_expr_to_str(node.value),
            annotation=_expr_to_str(node.annotation),
            is_class_attr=bool(self._class_stack),
            is_module_const=not self._class_stack and target.isupper(),
            location=_loc(node),
        )
        self._add_assignment(asg)
    def visit_Assign(self, node: ast.Assign) -> Any:
        targets = [t for t in node.targets]
        asg = AssignmentMap(
            targets=[_expr_to_str(t) or "<unknown>" for t in targets],
            value=_expr_to_str(node.value),
            annotation=None,
            is_class_attr=bool(self._class_stack),
            is_module_const=not self._class_stack and all((getattr(t, 'id', '') or '').isupper() for t in targets if isinstance(t, ast.Name)),
            location=_loc(node),
        )
        self._add_assignment(asg)
    def _handle_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], *, is_async: bool) -> None:
        is_method = bool(self._class_stack)
        a = node.args
        args: List[ArgumentMap] = []
        for pa in getattr(a, 'posonlyargs', []):
            args.append(ArgumentMap(name=pa.arg, annotation=_expr_to_str(pa.annotation), kind="posonly"))
        defaults_iter = iter([None] * (len(a.args) - len(a.defaults)) + list(a.defaults))
        for arg in a.args:
            args.append(ArgumentMap(name=arg.arg, annotation=_expr_to_str(arg.annotation), default=_expr_to_str(next(defaults_iter)), kind="positional"))
        if a.vararg:
            args.append(ArgumentMap(name=a.vararg.arg, annotation=_expr_to_str(a.vararg.annotation), kind="vararg"))
        for i, kw in enumerate(a.kwonlyargs):
            default = _expr_to_str((a.kw_defaults or [None]*len(a.kwonlyargs))[i])
            args.append(ArgumentMap(name=kw.arg, annotation=_expr_to_str(kw.annotation), default=default, kind="kwonly"))
        if a.kwarg:
            args.append(ArgumentMap(name=a.kwarg.arg, annotation=_expr_to_str(a.kwarg.annotation), kind="kwarg"))
        fc = _CallCollector()
        for stmt in node.body:
            fc.visit(stmt)
        fn = FunctionMap(
            name=node.name,
            qualname=self._qualname(node.name),
            decorators=[_expr_to_str(d) for d in node.decorator_list],
            returns=_expr_to_str(node.returns),
            args=args,
            docstring=_docstring(node),
            location=_loc(node),
            calls=fc.calls,
            is_async=is_async,
            is_method=is_method,
            visibility=("private" if node.name.startswith('__') and not node.name.endswith('__') else ("protected" if node.name.startswith('_') else "public")),
        )
        if self._class_stack:
            self._class_stack[-1].methods.append(fn)
        else:
            self.module_map.functions.append(fn)
    def _add_assignment(self, asg: AssignmentMap) -> None:
        if self._class_stack:
            self._class_stack[-1].attributes.append(asg)
        else:
            self.module_map.assignments.append(asg)
    def _qualname(self, name: str) -> str:
        if self._class_stack:
            return f"{self.module_name}.{self._class_stack[-1].name}.{name}"
        return f"{self.module_name}.{name}"

# ------------------------------------
# Graph builder
# ------------------------------------

def build_graph(module_maps: List[ModuleMap], *, show_contains: bool, show_calls: bool, show_imports: bool) -> nx.DiGraph:
    G = nx.DiGraph()

    def add_node(node_id: str, label: str, ntype: str, title: str = ""):
        G.add_node(node_id, label=label, ntype=ntype, title=title)

    def add_edge(src: str, dst: str, etype: str):
        if src and dst and src != dst:
            G.add_edge(src, dst, etype=etype)

    # Index for resolving simple call names → best guess fully-qualified
    index_names: Dict[str, str] = {}

    for m in module_maps:
        m_id = f"mod:{m.module}"
        add_node(m_id, m.module, "module", title=(m.docstring or ""))
        # imports
        if show_imports:
            for im in m.imports:
                imp_name = im.module if im.kind == "from" else im.name
                if imp_name:
                    imp_id = f"ext:{imp_name}"
                    add_node(imp_id, imp_name, "external")
                    add_edge(m_id, imp_id, "imports")
        # classes
        for c in m.classes:
            c_id = f"class:{c.qualname}"
            add_node(c_id, c.name, "class", title=(c.docstring or ""))
            index_names.setdefault(c.name, c_id)
            if show_contains:
                add_edge(m_id, c_id, "contains")
            # methods
            for fn in c.methods:
                f_id = f"func:{fn.qualname}"
                add_node(f_id, fn.name + ("()"), "function", title=(fn.docstring or ""))
                index_names.setdefault(fn.name, f_id)
                if show_contains:
                    add_edge(c_id, f_id, "contains")
                if show_calls:
                    for callee in fn.calls:
                        dst = index_names.get(callee)
                        if dst:
                            add_edge(f_id, dst, "calls")
        # functions
        for fn in m.functions:
            f_id = f"func:{fn.qualname}"
            add_node(f_id, fn.name + ("()"), "function", title=(fn.docstring or ""))
            index_names.setdefault(fn.name, f_id)
            if show_contains:
                add_edge(m_id, f_id, "contains")
            if show_calls:
                for callee in fn.calls:
                    dst = index_names.get(callee)
                    if dst:
                        add_edge(f_id, dst, "calls")
    return G

# ------------------------------------
# UI
# ------------------------------------

st.set_page_config(page_title="MapifyPython — Visual Mapper", layout="wide")
st.title("MapifyPython — Paste code → Visual map")

with st.sidebar:
    st.markdown("### Paste Python files")
    st.caption("Optionally give each paste a file name for better labels.")

    file_entries: List[Tuple[str, str]] = []
    num_files = st.number_input("How many files?", min_value=1, max_value=20, value=1, step=1)
    for i in range(int(num_files)):
        name = st.text_input(f"File {i+1} name", value=f"snippet_{i+1}.py")
        code = st.text_area(f"Code for {name}", height=200, key=f"code_{i}")
        if code.strip():
            file_entries.append((name, code))

    st.markdown("### Edges to show (Graph)")
    show_contains = st.checkbox("Containment (module→class/function)", value=True)
    show_calls = st.checkbox("Calls (function→function)", value=True)
    show_imports = st.checkbox("Imports (module→external)", value=True)

    st.markdown("### Layout")
    physics = st.checkbox("Enable physics layout", value=True)
    height = st.text_input("Graph/Tree height", value="700px")

parse_btn = st.button("Map it ")

# Tabs: Graph view and JSON Tree view (MapifyJSON style)
tab_graph, tab_tree, tab_raw = st.tabs(["Graph", "JSON Tree", "Raw JSON"])

module_maps: List[ModuleMap] = []
errors: List[str] = []

if parse_btn and file_entries:
    for fname, code in file_entries:
        try:
            tree = ast.parse(code, filename=fname, type_comments=True)
            m = Mapify(file_path=fname, module_name=fname.rsplit('.', 1)[0]).build(tree)
            module_maps.append(m)
        except SyntaxError as e:
            errors.append(f"{fname}: SyntaxError: {e}")

    if errors:
        st.error("
".join(errors))

with tab_graph:
    st.subheader("Graph")
    if module_maps:
        G = build_graph(module_maps, show_contains=show_contains, show_calls=show_calls, show_imports=show_imports)
        net = Network(height=height, width="100%", directed=True, notebook=False)
        net.barnes_hut() if physics else net.hrepulsion()
        # style by node type
        for n, data in G.nodes(data=True):
            ntype = data.get("ntype", "other")
            title = data.get("title") or ""
            label = data.get("label", n)
            if ntype == "module":
                color = "#89b4fa"; shape = "box"
            elif ntype == "class":
                color = "#f38ba8"; shape = "database"
            elif ntype == "function":
                color = "#a6e3a1"; shape = "ellipse"
            elif ntype == "external":
                color = "#cba6f7"; shape = "box"
            else:
                color = "#cdd6f4"; shape = "dot"
            net.add_node(n, label=label, title=title, color=color, shape=shape)
        for u, v, data in G.edges(data=True):
            et = data.get("etype", "edge")
            if et == "contains":
                dashes = True; arrows = "to"
            elif et == "calls":
                dashes = False; arrows = "to"
            elif et == "imports":
                dashes = True; arrows = "to"
            else:
                dashes = False; arrows = "to"
            net.add_edge(u, v, arrows=arrows, dashes=dashes)
        html = net.generate_html(notebook=False)
        st.components.v1.html(html, height=int(height.replace("px", "").strip() or 700), scrolling=True)
    else:
        st.info("Paste code on the left and click **Map it**.")

with tab_tree:
    st.subheader("JSON Tree (MapifyJSON style)")
    if module_maps:
        # build a hierarchical object similar to MapifyJSON input (arbitrary JSON is fine)
        tree_obj = {
            "files": [asdict(m) for m in module_maps]
        }
        # try to load local jsontr.ee.js
        js_src = None
        js_path = Path("jsontr.ee.js")
        if js_path.exists():
            try:
                js_src = js_path.read_text(encoding="utf-8")
            except Exception:
                js_src = None
        if js_src is None:
            st.warning(
                "jsontr.ee.js not found next to this file. Download it from the MapifyJSON repo and place it alongside this app."
            )
            st.json(tree_obj)
        else:
            # embed JS + data
            container_id = "json-tree"
            html = f"""
            <div id='{container_id}'></div>
            <script>
            {js_src}
            const data = {json.dumps(tree_obj)};
            const el = document.getElementById('{container_id}');
            el.innerHTML = generateJSONTree(data);
            </script>
            """
            st.components.v1.html(html, height=int(height.replace("px", "").strip() or 700), scrolling=True)
    else:
        st.info("Paste code and map it to see the JSON tree.")

with tab_raw:
    st.subheader("Raw module maps (JSON)")
    if module_maps:
        for m in module_maps:
            st.json(asdict(m))
    else:
        st.caption("Nothing yet.")
