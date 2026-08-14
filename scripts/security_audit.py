#!/usr/bin/env python3
"""Deterministic local and live security checks for the PyMap static site."""

from __future__ import annotations

import argparse
import json
import re
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "", ".css", ".html", ".ini", ".js", ".json", ".md", ".py", ".toml",
    ".txt", ".yaml", ".yml",
}
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
ALLOWED_SCRIPT_HOSTS = {"cdn.jsdelivr.net"}
SENSITIVE_LIVE_PATHS = (".env", ".env.production", ".git/config")
COMMON_NONPRODUCTION_LABELS = ("staging", "stage", "dev", "test", "beta", "api", "admin")

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\bA" + r"KIA[0-9A-Z]{16}\b"),
    "AWS temporary access key": re.compile(r"\bA" + r"SIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAI" + r"za[0-9A-Za-z_-]{35}\b"),
    "OpenAI-style secret": re.compile(r"\bsk" + r"-[0-9A-Za-z_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh" + r"[oprsu]_[0-9A-Za-z]{20,}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_" + r"pat_[0-9A-Za-z_]{20,}\b"),
    "Supabase management token": re.compile(r"\bsbp" + r"_[0-9A-Za-z]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


@dataclass
class AuditResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    def merge(self, other: "AuditResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.passed.extend(other.passed)


class DocumentInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: list[dict[str, str]] = []
        self.scripts: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta":
            self.meta.append(values)
        elif tag.lower() == "script":
            self.scripts.append(values)


def iter_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 2_000_000:
            yield path


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def audit_git_history(root: Path) -> AuditResult:
    result = AuditResult()
    try:
        names = subprocess.run(
            ["git", "log", "--all", "--name-only", "--pretty=format:"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        result.warnings.append("Git history was unavailable; historical filename checks were skipped")
        return result

    leaked_names = sorted({name for name in names if Path(name).name == ".env" or Path(name).name.startswith(".env.")})
    if leaked_names:
        result.errors.append("Environment files occur in Git history: " + ", ".join(leaked_names))
    else:
        result.passed.append("No environment files occur in Git history")
    return result


def audit_local(root: Path = ROOT) -> AuditResult:
    result = AuditResult()
    files = list(iter_text_files(root))

    env_files = [relative(path, root) for path in files if path.name == ".env" or path.name.startswith(".env.")]
    if env_files:
        result.errors.append("Environment files are present: " + ", ".join(sorted(env_files)))
    else:
        result.passed.append("No environment files are present")

    secret_hits: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                secret_hits.append(f"{relative(path, root)} ({label})")
    if secret_hits:
        result.errors.append("Potential secrets detected in: " + ", ".join(sorted(secret_hits)))
    else:
        result.passed.append("No known API-key or private-key patterns were detected")

    gitignore = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    for required in (".env", ".env.*"):
        if required not in gitignore:
            result.errors.append(f".gitignore is missing {required}")
    if all(required in gitignore for required in (".env", ".env.*")):
        result.passed.append("Environment files are blocked by .gitignore")

    index = (root / "index.html").read_text(encoding="utf-8")
    inspector = DocumentInspector()
    inspector.feed(index)

    csp_values = [
        meta.get("content", "")
        for meta in inspector.meta
        if meta.get("http-equiv", "").lower() == "content-security-policy"
    ]
    required_csp = ("default-src 'none'", "object-src 'none'", "base-uri 'none'", "form-action 'none'")
    if not csp_values or not all(directive in csp_values[0] for directive in required_csp):
        result.errors.append("index.html is missing the required restrictive Content Security Policy")
    else:
        result.passed.append("A restrictive Content Security Policy is present")

    referrer_values = [
        meta.get("content", "").lower()
        for meta in inspector.meta
        if meta.get("name", "").lower() == "referrer"
    ]
    if "no-referrer" not in referrer_values:
        result.errors.append("index.html is missing a no-referrer policy")
    else:
        result.passed.append("Referrer data is disabled")

    remote_scripts = [script for script in inspector.scripts if script.get("src")]
    for script in remote_scripts:
        src = script["src"]
        parsed = urllib.parse.urlparse(src)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SCRIPT_HOSTS:
            result.errors.append(f"Unapproved remote script: {src}")
        if not re.search(r"(?:@|/v)\d+(?:\.\d+){1,2}", src):
            result.errors.append(f"Remote script is not version-pinned: {src}")
        if not re.fullmatch(r"sha384-[A-Za-z0-9+/]{64}", script.get("integrity", "")):
            result.errors.append(f"Remote script lacks SHA-384 integrity protection: {src}")
        if script.get("crossorigin", "").lower() != "anonymous":
            result.errors.append(f"Remote script lacks anonymous CORS mode: {src}")
    if remote_scripts and not any(error.startswith("Unapproved remote script") or "Remote script" in error for error in result.errors):
        result.passed.append("Remote scripts are allowlisted, pinned, and integrity-protected")

    forbidden_browser_features = {
        "cookies": r"document\s*\.\s*cookie",
        "local storage": r"\blocalStorage\b",
        "session storage": r"\bsessionStorage\b",
        "dynamic eval": r"\beval\s*\(",
    }
    for label, pattern in forbidden_browser_features.items():
        if re.search(pattern, index):
            result.errors.append(f"index.html uses forbidden browser feature: {label}")
    if not any("forbidden browser feature" in error for error in result.errors):
        result.passed.append("No cookies, persistent browser storage, or dynamic eval are used")

    if re.search(r"\bsupabase\b|supabase\.co", index, re.IGNORECASE):
        result.errors.append("Supabase usage was detected; verify RLS and key scope before deployment")
    else:
        result.passed.append("No Supabase client or backend dependency is present")

    insecure_urls = {
        url for url in re.findall(r"http://[^\s\"'`<>]+", index, re.IGNORECASE)
        if url.rstrip(");") != "http://www.w3.org/2000/svg"
    }
    if insecure_urls:
        result.errors.append("Insecure HTTP URL detected in index.html")
    else:
        result.passed.append("All application URLs use HTTPS or same-origin paths")

    result.merge(audit_git_history(root))
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def request(url: str) -> tuple[int, object, bytes]:
    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "PyMap-Security-Audit/1.0"})
    try:
        response = opener.open(req, timeout=20)
    except urllib.error.HTTPError as exc:
        response = exc
    return response.status, response.headers, response.read()


def audit_live(base_url: str) -> AuditResult:
    result = AuditResult()
    if not base_url.startswith("https://"):
        result.errors.append("Live audit URL must use HTTPS")
        return result
    if not base_url.endswith("/"):
        base_url += "/"

    try:
        status, headers, body = request(base_url)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Live site could not be loaded: {type(exc).__name__}")
        return result

    if status != 200:
        result.errors.append(f"Live site returned HTTP {status}")
    else:
        result.passed.append("Live site returns HTTP 200 over HTTPS")
    if headers.get_all("Set-Cookie", []):
        result.errors.append("Live site sets cookies")
    else:
        result.passed.append("Live site does not set cookies")

    live_html = body.decode("utf-8", errors="ignore")
    inspector = DocumentInspector()
    inspector.feed(live_html)
    if not any(meta.get("http-equiv", "").lower() == "content-security-policy" for meta in inspector.meta):
        result.errors.append("Live page does not contain the Content Security Policy meta tag")
    else:
        result.passed.append("Live page contains the Content Security Policy")

    for path in SENSITIVE_LIVE_PATHS:
        target = urllib.parse.urljoin(base_url, path)
        try:
            sensitive_status, _, _ = request(target)
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Could not check {target}: {type(exc).__name__}")
            continue
        if sensitive_status == 200:
            result.errors.append(f"Sensitive path is publicly accessible: {target}")
        else:
            result.passed.append(f"Sensitive path is blocked: {path} (HTTP {sensitive_status})")

    origin = urllib.parse.urlunparse((*urllib.parse.urlparse(base_url)[:2], "/", "", "", ""))
    try:
        _, origin_headers, _ = request(origin)
        if not origin_headers.get("Strict-Transport-Security"):
            result.warnings.append("Origin does not advertise HSTS")
        else:
            result.passed.append("Origin advertises HSTS")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"Could not check origin HSTS: {type(exc).__name__}")

    return result


def audit_subdomains(domain: str, allowed: set[str]) -> AuditResult:
    result = AuditResult()
    candidates = {domain, f"www.{domain}"}
    candidates.update(f"{label}.{domain}" for label in COMMON_NONPRODUCTION_LABELS)
    try:
        ct_url = "https://crt.sh/?q=" + urllib.parse.quote(f"%.{domain}") + "&output=json"
        req = urllib.request.Request(ct_url, headers={"User-Agent": "PyMap-Security-Audit/1.0"})
        records = json.load(urllib.request.urlopen(req, timeout=30))
        for record in records:
            for name in record.get("name_value", "").splitlines():
                normalized = name.lower().lstrip("*.")
                if normalized == domain or normalized.endswith("." + domain):
                    candidates.add(normalized)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"Certificate-transparency lookup failed: {type(exc).__name__}")

    resolving: set[str] = set()
    for host in sorted(candidates):
        try:
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            resolving.add(host)
        except socket.gaierror:
            continue

    unexpected = sorted(resolving - allowed)
    if unexpected:
        result.errors.append("Unexpected subdomains resolve: " + ", ".join(unexpected))
    else:
        result.passed.append("No unexpected production, staging, development, API, or admin subdomains resolve")
    result.passed.append("Resolving approved hosts: " + ", ".join(sorted(resolving & allowed)))
    return result


def print_result(result: AuditResult) -> None:
    for message in result.passed:
        print(f"PASS: {message}")
    for message in result.warnings:
        print(f"WARN: {message}")
    for message in result.errors:
        print(f"FAIL: {message}")
    print(f"SUMMARY: {len(result.passed)} passed, {len(result.warnings)} warnings, {len(result.errors)} failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a static site for common deployment and repository risks")
    parser.add_argument("--live-url", help="Optional deployed application URL")
    parser.add_argument("--domain", help="Optional apex domain for DNS and certificate checks")
    parser.add_argument("--allowed-subdomain", action="append", default=[], help="Approved resolving host; repeat as needed")
    args = parser.parse_args(argv)

    result = audit_local()
    if args.live_url:
        result.merge(audit_live(args.live_url))
    if args.domain:
        allowed = set(args.allowed_subdomain) or {args.domain, f"www.{args.domain}"}
        result.merge(audit_subdomains(args.domain, allowed))
    print_result(result)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
