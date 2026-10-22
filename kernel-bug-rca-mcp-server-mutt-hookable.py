#!/usr/bin/env python3
"""
kernel-bug-rca MCP server: exposes the kernel-bug-rca agent pipeline as
Model Context Protocol tools over stdio, so any MCP client can run a
root-cause analysis on a kernel bug report.

Two run modes:

  1. MCP SERVER (for MCP clients: Claude Desktop, Devin Desktop / Windsurf,
     Cursor, or any MCP-capable client):

        python3 kernel_bug_rca_mcp.py serve

     The client launches it with `mcpServers` config, e.g.:

        "kernel-bug-rca": {
          "command": "python3",
          "args": ["/home/you/bin/kernel_bug_rca_mcp.py", "serve"],
          "env": {
            "KERNEL_RCA_SCRIPT": "/home/you/bin/kernel_bug_rca.py",
            "KERNEL_RCA_REPO":     "/home/you/src/linux",
            "KERNEL_RCA_BACKEND":  "vllm",
            "KERNEL_RCA_MODEL":    "Qwen3-32B"
          }
        }

  2. MUTT PIPE (no MCP client needed; mutt pipes the current message in):

     muttrc:
        macro index,pager <tab>B \\
          "<pipe-message>python3 ~/.local/bin/kernel_bug_rca_mcp.py pipe<enter>" \\
          "run kernel bug RCA on this message"

     mutt pipes the raw message (headers + body, mbox-ish) on stdin; the
     tool saves it to a temp file, runs the pipeline, and prints the reply
     draft (plus where the full markdown report was written).

  Extra: `python3 kernel_bug_rca_mcp.py call <report> [--repo R]` invokes
  a single analysis directly, speaking the same subprocess interface the
  MCP tool uses - useful for cron/CI on new mail.

Protocol: MCP over stdio (JSON-RPC 2.0, newline-delimited).
Tools exposed:
  - kernel_bug_rca          : full analysis of a report file
  - kernel_bug_rca_search   : search lore + local git for a query (cheap)
  - kernel_bug_rca_draft    : triage-only (fast, no tool loop)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

PROTOCOL_VERSION = "2024-11-05"
PROG = "kernel-bug-rca"

RCA_SCRIPT = os.environ.get(
    "KERNEL_RCA_SCRIPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_bug_rca.py"))
DEFAULT_REPO = os.environ.get("KERNEL_RCA_REPO", os.path.expanduser("~/src/linux"))
DEFAULT_BRANCH = os.environ.get("KERNEL_RCA_BRANCH", "")
DEFAULT_BACKEND = os.environ.get("KERNEL_RCA_BACKEND", "ollama")
DEFAULT_MODEL = os.environ.get("KERNEL_RCA_MODEL", "qwen3:32b")
DEFAULT_SKILLS = os.environ.get("KERNEL_RCA_SKILLS", "")
MAX_TURNS = os.environ.get("KERNEL_RCA_MAX_TURNS", "10")
TIMEOUT_S = int(os.environ.get("KERNEL_RCA_TIMEOUT", "1800"))

TOOLS = [
    {
        "name": "kernel_bug_rca",
        "description":
            "Run a full multi-agent root cause analysis (triage, evidence "
            "gathering with git/lore, RCA + reply draft) on a Linux kernel "
            "bug report. Input: path to an .mbox or .txt report file. "
            "Returns the RCA markdown and the reply draft. Can take several "
            "minutes with a local model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "report_path": {
                    "type": "string",
                    "description": "Path to the bug report (.mbox or .txt)."},
                "report_text": {
                    "type": "string",
                    "description":
                        "Raw report text instead of a path (a temp file is "
                        "created automatically)."},
                "repo": {"type": "string",
                         "description": "Kernel git tree (default: env "
                                        "KERNEL_RCA_REPO)."},
                "branch": {"type": "string",
                           "description": "Git ref/branch to analyze, e.g. "
                                          "'for-next' or 'v6.12' (default: "
                                          "env KERNEL_RCA_BRANCH or the "
                                          "checked-out HEAD)."},
                "max_tokens": {"type": "integer",
                               "description": "max output tokens "
                                              "(claude backend)."},
                "skills": {"type": "string",
                           "description": "Directory with skills "
                                          "(SKILL.md folders or .md files); "
                                          "default: env KERNEL_RCA_SKILLS."},
                "backend": {"type": "string",
                            "enum": ["ollama", "vllm", "mistral", "claude"],
                            "description": "LLM backend. mistral/claude are "
                                           "hosted; keys come from the "
                                           "server env (MISTRAL_API_KEY / "
                                           "ANTHROPIC_API_KEY) or the "
                                           "api_key argument."},
                "api_key": {"type": "string",
                            "description": "API key for mistral/claude "
                                           "(prefer env vars)."},
                "model": {"type": "string"},
                "max_turns": {"type": "integer"},
                "out": {"type": "string",
                        "description": "Where to write the RCA markdown "
                                       "(default: next to the report)."},
            },
            "anyOf": [{"required": ["report_path"]},
                      {"required": ["report_text"]}],
        },
    },
    {
        "name": "kernel_bug_rca_draft",
        "description":
            "Fast triage-only pass on a kernel bug report (subsystem, "
            "severity, crash signature, stack frames, search hints). "
            "No git/lore tool loop; a single LLM call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "report_path": {"type": "string"},
                "report_text": {"type": "string"},
                "model": {"type": "string"},
                "backend": {"type": "string"},
            },
            "anyOf": [{"required": ["report_path"]},
                      {"required": ["report_text"]}],
        },
    },
    {
        "name": "kernel_bug_rca_search",
        "description":
            "Cheap evidence lookup without the LLM: full-text search of "
            "lore.kernel.org and/or git log --grep in the local kernel tree. "
            "Returns matches to feed into a bug analysis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "repo": {"type": "string"},
                "search_git": {"type": "boolean", "default": True},
                "search_lore": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Analysis backend (subprocess calls into kernel_bug_rca.py)
# ---------------------------------------------------------------------------


def _ensure_script() -> None:
    if not os.path.isfile(RCA_SCRIPT):
        raise FileNotFoundError(
            f"kernel_bug_rca.py not found at {RCA_SCRIPT}; set "
            f"KERNEL_RCA_SCRIPT to its path.")


def _materialize(args: dict) -> str:
    """Return report path, materializing report_text if needed."""
    path = args.get("report_path")
    if path:
        return os.path.abspath(os.path.expanduser(path))
    text = args.get("report_text") or ""
    fd, tmp = tempfile.mkstemp(suffix=".mbox", prefix="rca-")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return tmp


def run_full_rca(args: dict) -> str:
    _ensure_script()
    report = _materialize(args)
    if not os.path.isfile(report):
        raise FileNotFoundError(f"report not found: {report}")
    out = args.get("out") or os.path.splitext(report)[0] + "-rca.md"
    json_out = out.rsplit(".", 1)[0] + ".json"
    cmd = [sys.executable, RCA_SCRIPT, report,
           "--repo", args.get("repo", DEFAULT_REPO),
           "--backend", args.get("backend", DEFAULT_BACKEND),
           "--model", args.get("model", DEFAULT_MODEL),
           "--max-turns", str(args.get("max_turns", MAX_TURNS)),
           "--out", out, "--json-out", json_out, "-q"]
    skills_dir = args.get("skills", DEFAULT_SKILLS)
    if skills_dir:
        cmd += ["--skills", skills_dir]
    branch = args.get("branch", DEFAULT_BRANCH)
    if branch:
        cmd += ["--branch", branch]
    api_key = args.get("api_key", "")
    if api_key:
        cmd += ["--api-key", api_key]
    max_tokens = args.get("max_tokens")
    if max_tokens:
        cmd += ["--max-tokens", str(max_tokens)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline failed (rc={proc.returncode}): "
                           f"{proc.stderr.strip()[:1500]}")
    try:
        with open(out) as f:
            md = f.read()
    except OSError as e:
        raise RuntimeError(f"no RCA output: {e}")
    draft = ""
    m = re.search(r"## 5\. Reply draft\s+```text\n(.*?)\n```", md, re.DOTALL)
    if m:
        draft = m.group(1).strip()
    return (f"RCA written to: {out}\nJSON: {json_out}\n\n"
            f"=== REPLY DRAFT ===\n{draft}\n\n=== FULL REPORT ===\n{md[:15000]}"
            + ("\n... (truncated; open the markdown file for the full report)"
               if len(md) > 15000 else ""))


def run_triage_only(args: dict) -> str:
    _ensure_script()
    sys.path.insert(0, os.path.dirname(RCA_SCRIPT))
    import importlib.util
    spec = importlib.util.spec_from_file_location("kernel_bug_rca", RCA_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kernel_bug_rca"] = mod
    spec.loader.exec_module(mod)
    cfg = mod.Config(backend=args.get("backend", DEFAULT_BACKEND),
                     model=args.get("model", DEFAULT_MODEL))
    cfg.resolve()
    fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="rca-triage-")
    with os.fdopen(fd, "w") as f:
        f.write(args.get("report_text") or "")
    path = args.get("report_path") or tmp
    rep = mod.ingest(path)
    result = mod.run_triage(mod.LLM(cfg), rep, lambda _m: None)
    return json.dumps(result, indent=2)


def run_search(args: dict) -> str:
    query = args["query"]
    repo = os.path.abspath(os.path.expanduser(args.get("repo", DEFAULT_REPO)))
    results = []
    if args.get("search_git", True) and os.path.isdir(os.path.join(repo, ".git")):
        try:
            p = subprocess.run(["git", "-C", repo, "log", "--all", "-i",
                                f"--grep={query}", "--max-count=15",
                                "--oneline"], capture_output=True,
                               text=True, timeout=60)
            if p.stdout.strip():
                results.append("== git log matches ==\n" + p.stdout.strip())
        except Exception as e:  # noqa: BLE001
            results.append(f"git search failed: {e}")
    if args.get("search_lore", True):
        import urllib.parse
        import urllib.request
        url = "https://lore.kernel.org/all/?q=" + urllib.parse.quote(query)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": PROG})
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
            hits = re.findall(r'href="(/all/[^"]+)">([^<]{15,})<', html)[:15]
            if hits:
                results.append("== lore matches ==\n" + "\n".join(
                    f"- https://lore.kernel.org{u} | "
                    f"{re.sub('<[^>]+>', '', s).strip()}" for u, s in hits))
        except Exception as e:  # noqa: BLE001
            results.append(f"lore search failed: {e}")
    return "\n\n".join(results) or "(no results)"


def dispatch(name: str, args: dict) -> str:
    if name == "kernel_bug_rca":
        return run_full_rca(args)
    if name == "kernel_bug_rca_draft":
        return run_triage_only(args)
    if name == "kernel_bug_rca_search":
        return run_search(args)
    raise ValueError(f"unknown tool: {name}")


# ---------------------------------------------------------------------------
# MCP stdio server (JSON-RPC 2.0, newline-delimited)
# ---------------------------------------------------------------------------


def serve() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _send({"jsonrpc": "2.0", "id": None,
                   "error": {"code": -32700, "message": "parse error"}})
            continue
        method = req.get("method", "")
        rid = req.get("id")
        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": PROG, "version": "1.0.0"}}})
        elif method in ("notifications/initialized", "notifications/cancelled"):
            pass  # notification: no response
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": rid, "result": {}})
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = req.get("params", {})
            tname = params.get("name", "")
            targs = params.get("arguments", {}) or {}
            try:
                output = dispatch(tname, targs)
                _send({"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": output}],
                    "isError": False}})
            except Exception as e:  # noqa: BLE001
                _send({"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text",
                                 "text": f"error: {e}"}],
                    "isError": True}})
        elif method == "tools/list" or method == "resources/list":
            _send({"jsonrpc": "2.0", "id": rid, "result": {}})
        else:
            if rid is not None:
                _send({"jsonrpc": "2.0", "id": rid,
                       "error": {"code": -32601,
                                 "message": f"method not found: {method}"}})


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# mutt pipe mode: read message from stdin, run RCA, print draft
# ---------------------------------------------------------------------------


def pipe() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print("(empty message from mutt)", file=sys.stderr)
        return 1
    out_dir = os.environ.get("KERNEL_RCA_OUT_DIR", os.path.expanduser("~/rca"))
    os.makedirs(out_dir, exist_ok=True)
    stamp = os.path.basename(tempfile.mkstemp()[1])
    report = os.path.join(out_dir, f"report-{stamp}.mbox")
    with open(report, "w") as f:
        f.write(raw)
    args = {"report_path": report,
            "repo": DEFAULT_REPO, "backend": DEFAULT_BACKEND,
            "model": DEFAULT_MODEL,
            "out": os.path.join(out_dir, f"rca-{stamp}.md")}
    try:
        result = run_full_rca(args)
    except Exception as e:  # noqa: BLE001
        print(f"RCA failed: {e}", file=sys.stderr)
        return 2
    print(result)
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if mode == "serve":
        serve()
        return 0
    if mode == "pipe":
        return pipe()
    if mode == "call":
        # call <report> [--repo R] [--backend B] [--model M] [--max-turns N]
        rest = sys.argv[2:]
        if not rest:
            print("usage: call <report> [--repo R] [--backend B] "
                  "[--model M] [--max-turns N]", file=sys.stderr)
            return 1
        a: dict = {"report_path": rest[0]}
        i = 1
        while i + 1 < len(rest) + 1:
            if i + 1 < len(rest) and rest[i] in ("--repo", "--backend",
                                                 "--model", "--max-turns"):
                a[rest[i].lstrip("-")] = rest[i + 1]
                i += 2
            else:
                i += 1
        try:
            print(dispatch("kernel_bug_rca", a))
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"error: {e}", file=sys.stderr)
            return 2
    print(f"unknown mode: {mode} (use serve|pipe|call)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())