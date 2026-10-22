# # SPDX-License-Identifier: GPL-2.0-or-later
#!/usr/bin/env python3
"""
kernel-bug-rca: Agentic root-cause-analysis for Linux kernel bug reports.

Pipeline (multi-agent, sequential with shared state):

  1. INGEST   - parse an mbox file or plain-text bug report (email headers,
                body, attached dmesg/OOPS traces, test-system metadata).
  2. TRIAGE   - LLM agent classifies the report: subsystem, suspect area,
                severity, crash type, extracts the crash signature, picks
                applicable skills, and the questions that need answering.
  3. CONTEXT  - tool-driven agent gathers evidence from the local kernel
                git tree: git log / git show / git blame / grep, plus
                lore.kernel.org full-text search for prior discussions.
  4. RCA      - LLM agent synthesizes evidence into a root cause analysis:
                ranked hypothesis, failure chain, suspected commit(s),
                proposed fix, follow-up questions, and a ready-to-send
                mailing-list reply draft.

Skills: pass --skills DIR to load reusable instruction packs (Claude-style
<SKILL NAME>/SKILL.md folders or flat .md prompt snippets). The triage agent
sees a menu of loaded skills and selects the applicable ones; their full
bodies are then injected into the RCA agent's prompt.

LLM backends: Ollama (default, http://localhost:11434/v1), vLLM
(http://localhost:8000/v1), Mistral API (api.mistral.ai, OpenAI-compatible),
or Anthropic Claude (api.anthropic.com, native Messages API). API keys come
from the environment: MISTRAL_API_KEY or ANTHROPIC_API_KEY (or --api-key).

Only stdlib is used (urllib, json, subprocess, mailbox, argparse) - no pip deps.

Usage:
    python3 kernel_bug_rca.py report.mbox
    python3 kernel_bug_rca.py bug.txt --repo ~/src/linux --model qwen3:32b \
        --backend vllm --max-turns 12 --out rca.md --json-out rca.json
    python3 kernel_bug_rca.py bug.txt --skills ~/.kernel-skills
    python3 kernel_bug_rca.py bug.txt --backend mistral --model mistral-large-latest
    python3 kernel_bug_rca.py bug.txt --backend claude --model claude-sonnet-4-5
"""

from __future__ import annotations

import argparse
import json
import mailbox
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
MISTRAL_URL = os.environ.get("MISTRAL_URL", "https://api.mistral.ai/v1")
ANTHROPIC_URL = os.environ.get("ANTHROPIC_URL", "https://api.anthropic.com")
LORE_SEARCH_URL = "https://lore.kernel.org/all/?q="
HTTP_TIMEOUT = 240

# Fallback model per backend when --model is not given.
DEFAULT_MODELS = {
    "ollama": "qwen3:32b",
    "vllm": "Qwen3-32B",
    "mistral": "mistral-large-latest",
    "claude": "claude-sonnet-4-5",
}

SYSTEM_HINTS = (
    "You are analyzing Linux kernel bug reports, focused on the filesystem and "
    "storage stack (btrfs, xfs, zoned block devices / zonefs, blk-mq, dm). "
    "You know the kernel development process, mailing-list etiquette, and "
    "common failure patterns of ZNS devices (zone write pointer mismanagement, "
    "unfinished zone resets, log-tree corruption on zoned btrfs, active zone "
    "exhaustion, etc.)."
)


@dataclass
class Config:
    backend: str = "ollama"          # ollama | vllm | mistral | claude
    base_url: str = ""
    model: str = ""                 # resolved per backend if left empty
    repo: str = "."                  # path to local kernel git tree
    branch: str = ""                 # git ref to analyze (default: checked-out HEAD)
    skills_dir: str = ""             # directory of SKILL.md folders or .md files
    api_key: str = ""                # mistral/claude key (env: *_API_KEY)
    max_tokens: int = 8192           # required by the Anthropic API
    max_turns: int = 10              # tool-call loop limit for the context agent
    max_tool_output: int = 8000      # chars of tool output fed back to the LLM
    temperature: float = 0.2
    json_mode: bool = True

    def resolve(self) -> None:
        if not self.base_url:
            self.base_url = {
                "vllm": VLLM_URL,
                "mistral": MISTRAL_URL,
                "claude": ANTHROPIC_URL,
            }.get(self.backend, OLLAMA_URL)
        self.base_url = self.base_url.rstrip("/")
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.backend, DEFAULT_MODELS["ollama"])
        if not self.api_key:
            env_var = {
                "mistral": "MISTRAL_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
                "vllm": "VLLM_API_KEY",
            }.get(self.backend, "LLM_API_KEY")
            self.api_key = os.environ.get(env_var, "")


# ---------------------------------------------------------------------------
# Skills: reusable instruction packs that tune agent behaviour
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    name: str
    description: str
    body: str

    def menu_line(self) -> str:
        return f"- {self.name}: {self.description}"


def load_skills(skills_dir: str) -> list[Skill]:
    """Load skills from a directory.

    Accepted layouts (both are scanned):
      1. Claude-style:  <dir>/<skill-name>/SKILL.md   (name from frontmatter
         or folder name; description from frontmatter 'description:')
      2. Flat snippets: <dir>/<anything>.md           (name = filename stem)

    Frontmatter is parsed leniently (no yaml dependency): a leading '---'
    block with 'name:'/'description:' lines is stripped from the body.
    """
    skills: list[Skill] = []
    if not skills_dir or not os.path.isdir(skills_dir):
        return skills
    candidates: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(skills_dir)):
        p = os.path.join(skills_dir, entry)
        if os.path.isdir(p):
            sk = os.path.join(p, "SKILL.md")
            if os.path.isfile(sk):
                candidates.append((entry, sk))
        elif entry.lower().endswith(".md"):
            candidates.append((entry[:-3], p))
    for default_name, path in candidates:
        try:
            raw = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        name, desc, body = default_name, "", raw.strip()
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.DOTALL)
        if m:
            fm, body = m.group(1), m.group(2)
            for line in fm.splitlines():
                kv = re.match(r"^(name|description):\s*(.*)$", line)
                if kv:
                    val = kv.group(2).strip().strip('"\'')
                    if kv.group(1) == "name":
                        name = val or default_name
                    else:
                        desc = val
        if not desc:
            first = body.strip().splitlines()
            desc = first[0][:120] if first else ""
        skills.append(Skill(name, desc, body.strip()))
    return skills


def skills_menu(skills: list[Skill]) -> str:
    if not skills:
        return ""
    return ("\n\nAVAILABLE SKILLS (specialised review/analysis instruction "
            "packs; reference them by name in applicable_skills):\n"
            + "\n".join(s.menu_line() for s in skills))


def skill_bodies(skills: list[Skill], names: list[str],
                 cap: int = 12000) -> str:
    """Render the full bodies of the named skills, capped at `cap` chars."""
    if not skills or not names:
        return ""
    by_name = {s.name: s for s in skills}
    out = []
    for n in names:
        s = by_name.get(str(n).strip())
        if not s:
            # Fuzzy match: skill name contained in the requested string.
            s = next((x for x in skills if x.name.lower() in str(n).lower()
                      or str(n).lower() in x.name.lower()), None)
        if s:
            out.append(f"### SKILL: {s.name}\n{s.body}\n### END SKILL\n")
    text = "\n".join(out)
    if len(text) > cap:
        text = text[:cap] + "\n(skill text truncated)"
    return ("\n\nACTIVE SKILLS (follow these instructions where they "
            "conflict with your defaults):\n" + text) if text else ""


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible chat completions, stdlib only)
# ---------------------------------------------------------------------------


class LLM:
    """Chat client for the configured backend.

    ollama / vllm / mistral speak the OpenAI chat-completions protocol.
    claude uses the Anthropic Messages API; the OpenAI-style messages and
    tool schemas used internally by the agents are converted transparently.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.calls = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """One model call. Returns {"content": str, "tool_calls": [...]}"""
        if self.cfg.backend == "claude":
            return self._chat_anthropic(messages, tools)
        return self._chat_openai(messages, tools)

    # -- OpenAI-compatible: ollama, vLLM, Mistral ----------------------------

    def _chat_openai(self, messages: list[dict],
                     tools: list[dict] | None) -> dict:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        if tools:
            payload["tools"] = tools
        if self.cfg.json_mode and not tools:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        req = urllib.request.Request(
            self.cfg.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        self.calls += 1
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        msg = data["choices"][0]["message"]
        return {
            "content": msg.get("content") or "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args": _safe_json(tc["function"]["arguments"]),
                }
                for tc in (msg.get("tool_calls") or [])
            ],
        }

    # -- Anthropic Messages API (claude) -------------------------------------

    def _chat_anthropic(self, messages: list[dict],
                        tools: list[dict] | None) -> dict:
        system_texts: list[str] = []
        msgs: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_texts.append(msg.get("content") or "")
            elif role == "tool":
                # OpenAI tool result -> Anthropic user/tool_result block.
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result",
                     "tool_use_id": msg.get("tool_call_id"),
                     "content": msg.get("content") or ""}]})
            elif role == "assistant" and msg.get("tool_calls"):
                blocks = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "input": _safe_json(fn.get("arguments") or "{}"),
                    })
                msgs.append({"role": "assistant", "content": blocks})
            else:
                msgs.append({"role": role, "content": msg.get("content") or ""})

        # The Messages API requires strictly alternating roles; merge runs of
        # consecutive same-role messages (e.g. text assistant + tool_use
        # assistant, or tool_result user + plain user).
        merged: list[dict] = []
        for msg in msgs:
            if merged and merged[-1]["role"] == msg["role"]:
                prev = merged[-1]["content"]
                cur = msg["content"]
                prev = prev if isinstance(prev, list) else \
                    [{"type": "text", "text": prev}]
                cur = cur if isinstance(cur, list) else \
                    [{"type": "text", "text": cur}]
                merged[-1]["content"] = prev + cur
            else:
                merged.append(msg)

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": merged,
        }
        if system_texts:
            payload["system"] = "\n\n".join(system_texts)
        if tools:
            payload["tools"] = [
                {"name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "input_schema": t["function"].get(
                     "parameters", {"type": "object", "properties": {}})}
                for t in tools]
        req = urllib.request.Request(
            self.cfg.base_url + "/v1/messages",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "x-api-key": self.cfg.api_key,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        self.calls += 1
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        blocks = data.get("content", [])
        content = "".join(b.get("text", "") for b in blocks
                          if b.get("type") == "text")
        tool_calls = [{"id": b.get("id"), "name": b.get("name"),
                       "args": b.get("input", {})}
                      for b in blocks if b.get("type") == "tool_use"]
        return {"content": content, "tool_calls": tool_calls}


def _safe_json(s: str) -> dict:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def _json_block(text: str) -> dict:
    """Extract the first JSON object from model text (handles json fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Ingestion: mbox / plain text -> BugReport
# ---------------------------------------------------------------------------

CRASH_PATTERNS = [
    r"general protection fault", r"bug:", r"warning:", r"oops:",
    r"kernel panic", r"call trace:", r"rip:", r"unable to handle",
    r"list_del corruption", r"refcount_t", r"workqueue:.*hung",
    r"task .* blocked for more than", r"hung_task", r"i/o error",
    r"btrfs error", r"btrfs warning", r"btrfs:.*assert", r"xfs.*error",
    r"blk_zone", r"zone append", r"write pointer",
]  # lower-case; matched against lower-cased log lines


@dataclass
class BugReport:
    subject: str = ""
    from_addr: str = ""
    date: str = ""
    body: str = ""
    trace: str = ""            # extracted OOPS/panic/hung-task excerpt
    source_file: str = ""
    is_mbox: bool = False
    metadata: dict = field(default_factory=dict)  # test-system tags etc.

    @property
    def crash_type(self) -> str:
        t = (self.trace or self.body).lower()
        for pat, name in [
            (r"kernel panic", "panic"),
            (r"general protection fault|oops:|rip:", "oops/gpf"),
            (r"warning:|list_del corruption|refcount", "warning"),
            (r"hung_task|blocked for more than", "hung task"),
            (r"btrfs error|xfs.*error|i/o error", "fs/io error"),
        ]:
            if re.search(pat, t, re.IGNORECASE):
                return name
        return "unknown"


def ingest(path: str) -> BugReport:
    raw = open(path, "rb").read()
    rep = BugReport(source_file=path)

    if path.endswith(".mbox") or raw.startswith(b"From "):
        rep.is_mbox = True
        mbox = mailbox.mbox(path)
        msgs = list(mbox)
        if not msgs:
            raise ValueError(f"empty mbox: {path}")
        # Prefer the most technical-looking message (the one with the
        # longest extracted trace).
        if len(msgs) > 1:
            msg = max(msgs, key=lambda m: len(_extract_trace(_body_of(m))))
        else:
            msg = msgs[0]
        rep.subject = str(msg.get("Subject") or msgs[0].get("Subject") or "")
        rep.from_addr = str(msg.get("From") or msgs[0].get("From") or "")
        rep.date = str(msg.get("Date") or msgs[0].get("Date") or "")
        rep.body = _body_of(msg)
        rep.metadata["n_messages"] = len(msgs)
        # Fold in short replies so the agent sees the whole thread.
        if len(msgs) > 1:
            extras = "\n\n".join(
                f"--- reply from {m.get('From', '?')} ---\n"
                + _body_of(m)[:3000]
                for m in msgs[1:5]
            )
            rep.body += "\n\n===== OTHER MESSAGES IN THREAD =====\n" + extras
    else:
        # Plain text: may still carry mail headers.
        import email
        import email.policy
        text = raw.decode("utf-8", errors="replace")
        hdr = email.message_from_string(text, policy=email.policy.default)
        if hdr.get("Subject") or hdr.get("From"):
            rep.subject = str(hdr.get("Subject") or "")
            rep.from_addr = str(hdr.get("From") or "")
            rep.date = str(hdr.get("Date") or "")
            body = hdr.get_content()
            rep.body = body if isinstance(body, str) else text
        else:
            rep.body = text
            m = re.search(r"^Subject:\s*(.+)$", text, re.MULTILINE)
            rep.subject = m.group(1) if m else os.path.basename(path)

    rep.trace = _extract_trace(rep.body)
    return rep


def _body_of(msg) -> str:
    if msg.is_multipart():
        parts = []
        for p in msg.walk():
            if p.get_content_type() == "text/plain":
                parts.append(_get_text_payload(p))
        return "\n".join(parts)
    return _get_text_payload(msg)


def _get_text_payload(msg) -> str:
    # Works for both the new (EmailMessage.get_content) and the legacy
    # compat32 policy (mailbox.mbox default), where get_content is absent.
    get_content = getattr(msg, "get_content", None)
    if get_content is not None:
        try:
            body = get_content()
            return body if isinstance(body, str) else ""
        except Exception:
            pass
    payload = msg.get_payload(decode=True)
    if payload is None:
        return str(msg.get_payload() or "")
    charset = msg.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _extract_trace(body: str) -> str:
    """Grab the OOPS/WARNING/hung-task block from the report body."""
    lines = body.splitlines()
    keep: list[str] = []
    collecting = False
    for i, ln in enumerate(lines):
        low = ln.lower()
        if not collecting and any(re.search(p, low) for p in CRASH_PATTERNS):
            collecting = True
            # Include a couple of preceding lines for context.
            keep.extend(lines[max(0, i - 2):i])
        if collecting:
            keep.append(ln)
            if len(keep) > 80:
                break
            # A blank line well into the block ends it.
            if ln.strip() == "" and len(keep) > 5:
                break
    return "\n".join(keep)[:20000]


# ---------------------------------------------------------------------------
# Tools for the context agent
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    spec: dict                                   # OpenAI tool schema
    fn: Callable[[dict], str]


def _git(cfg: Config, *args: str, max_out: int = 8000) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", cfg.repo] + list(args),
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        return out[:max_out] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "git timed out"
    except subprocess.CalledProcessError as e:
        return (e.stderr or e.stdout or "git error")[:2000]


def build_tools(cfg: Config) -> list[Tool]:
    rev = cfg.branch.strip() or "HEAD"

    def git_log(a):
        args = ["log", f"--max-count={int(a.get('n', 20))}",
                "--oneline", "--no-merges", rev]
        paths = a.get("paths") or []
        if paths:
            args.append("--")
            args.extend(paths)
        return _git(cfg, *args)

    def git_show(a):
        commit = a["commit"]
        # Allow short SHAs / subjects to resolve against the chosen ref.
        return _git(cfg, "show", "--stat", "--patch", commit,
                    max_out=cfg.max_tool_output)

    def git_blame(a):
        start = int(a.get("line", 1))
        span = int(a.get("span", 30))
        return _git(cfg, "blame", "-L", f"{start},{start + span}",
                    "--date=short", rev, "--", a["file"],
                    max_out=cfg.max_tool_output)

    def grep_source(a):
        pat = a["pattern"]
        paths = a.get("paths") or []
        if cfg.branch:
            # Search the branch content, not the working tree.
            args = ["grep", "-n", "-E", pat, cfg.branch, "--"] + \
                   (paths or [":/"])
            out = _git(cfg, *args, max_out=cfg.max_tool_output)
        else:
            include = "--include=" + (a.get("include") or "*.[ch]")
            cmd = ["grep", "-rn", "-E", pat, include] + (paths or ["."])
            try:
                out = subprocess.run(cmd, cwd=cfg.repo, capture_output=True,
                                     text=True, timeout=60).stdout
            except Exception as e:
                return f"grep error: {e}"
        out = out or ""
        if out == "(no output)":
            out = ""
        inc = (a.get("include") or "").strip()
        if inc:
            # git grep lines are file:line:content; filter on the filename.
            out = "\n".join(l for l in out.splitlines()
                            if l.split(":", 1)[0].endswith(
                                inc.replace("*", "")))
        return out[:cfg.max_tool_output] or "(no matches)"

    def git_log_search(a):
        scope = ["--all"] if not cfg.branch else [cfg.branch]
        return _git(cfg, "log", *scope, f"--grep={a['pattern']}", "-i",
                    f"--max-count={int(a.get('n', 15))}", "--oneline")

    def read_file(a):
        start = max(0, int(a.get("start", 1)) - 1)
        end = start + int(a.get("count", 80))
        if cfg.branch:
            # Read file content from the branch, not the working tree.
            out = _git(cfg, "show", f"{cfg.branch}:{a['path']}",
                       max_out=cfg.max_tool_output * 4)
            if out.startswith("fatal:"):
                return out[:2000]
            lines = out.splitlines()
            return "".join(f"{start + i + 1:5d}: {l}\n"
                           for i, l in enumerate(
                               lines[start:end]))[:cfg.max_tool_output]
        p = os.path.normpath(os.path.join(cfg.repo, a["path"]))
        repo_abs = os.path.abspath(cfg.repo)
        if not os.path.abspath(p).startswith(repo_abs):
            return "refusing path outside repo"
        try:
            with open(p, errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            return f"error: {e}"
        return "".join(f"{start + i + 1:5d}: {l}"
                       for i, l in enumerate(lines[start:end]))[:cfg.max_tool_output]

    def lore_search(a):
        q = urllib.parse.quote(a["query"])
        try:
            req = urllib.request.Request(
                LORE_SEARCH_URL + q,
                headers={"User-Agent": "kernel-bug-rca/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"lore search failed: {e}"
        hits = re.findall(r'<a[^>]+href="(/all/[^"]+)"[^>]*>(.*?)</a>', html)
        if not hits:
            hits = re.findall(r'href="(/all/[^"]+)">([^<]{15,})<', html)
        out = [f"- {re.sub('<[^>]+>', '', s).strip()} | https://lore.kernel.org{u}"
               for u, s in hits[:15]]
        return "\n".join(out) or "(no results)"

    def kernel_version(_a):
        return _git(cfg, "describe", "--tags", cfg.branch.strip() or "HEAD")

    return [
        Tool({
            "type": "function",
            "function": {
                "name": "git_log",
                "description": "git log --oneline for given paths "
                               "(e.g. fs/btrfs/, drivers/block/, fs/xfs/)",
                "parameters": {"type": "object", "properties": {
                    "n": {"type": "integer", "default": 20},
                    "paths": {"type": "array", "items": {"type": "string"}}},
                    "required": []},
            }}, git_log),
        Tool({
            "type": "function",
            "function": {
                "name": "git_log_search",
                "description": "Search all commit messages for a pattern "
                               "(e.g. 'zone append btrfs', 'write pointer')",
                "parameters": {"type": "object", "properties": {
                    "pattern": {"type": "string"},
                    "n": {"type": "integer", "default": 15}},
                    "required": ["pattern"]},
            }}, git_log_search),
        Tool({
            "type": "function",
            "function": {
                "name": "git_show",
                "description": "Show a commit's message, stat and full patch",
                "parameters": {"type": "object", "properties": {
                    "commit": {"type": "string"}}, "required": ["commit"]},
            }}, git_show),
        Tool({
            "type": "function",
            "function": {
                "name": "git_blame",
                "description": "git blame a region of a file to find the "
                               "commits that introduced suspect code",
                "parameters": {"type": "object", "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "span": {"type": "integer", "default": 30}},
                    "required": ["file", "line"]},
            }}, git_blame),
        Tool({
            "type": "function",
            "function": {
                "name": "grep_source",
                "description": "grep -rn -E a pattern in the kernel tree "
                               "(optionally restricted to paths)",
                "parameters": {"type": "object", "properties": {
                    "pattern": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "include": {"type": "string"}},
                    "required": ["pattern"]},
            }}, grep_source),
        Tool({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a source file region with line numbers",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "default": 1},
                    "count": {"type": "integer", "default": 80}},
                    "required": ["path"]},
            }}, read_file),
        Tool({
            "type": "function",
            "function": {
                "name": "lore_search",
                "description": "Full-text search of lore.kernel.org "
                               "mailing list archives",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"}}, "required": ["query"]},
            }}, lore_search),
        Tool({
            "type": "function",
            "function": {
                "name": "kernel_version",
                "description": "git describe of the kernel tree "
                               "under analysis",
                "parameters": {"type": "object", "properties": {}},
            }}, kernel_version),
    ]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class ToolAgent:
    """LLM agent with a tool-call loop (used by the CONTEXT stage)."""

    def __init__(self, llm: LLM, cfg: Config, tools: list[Tool],
                 log: Callable[[str], None]):
        self.llm = llm
        self.cfg = cfg
        self.tools = {t.spec["function"]["name"]: t for t in tools}
        self.specs = [t.spec for t in tools]
        self.log = log

    def run(self, system: str, user: str) -> tuple[str, list[dict]]:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        evidence: list[dict] = []
        final = ""
        for _turn in range(self.cfg.max_turns):
            res = self.llm.chat(messages, tools=self.specs)
            if not res["tool_calls"]:
                final = res["content"]
                break
            if res["content"]:
                messages.append({"role": "assistant", "content": res["content"]})
            for tc in res["tool_calls"]:
                tool = self.tools.get(tc["name"])
                if not tool:
                    out = f"unknown tool {tc['name']}"
                else:
                    self.log(f"      [{tc['name']}] "
                             f"{json.dumps(tc['args'])[:120]}")
                    try:
                        out = tool.fn(tc["args"])
                    except Exception as e:
                        out = f"tool error: {e}"
                evidence.append({"tool": tc["name"], "args": tc["args"],
                                 "output": out})
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": tc["id"], "type": "function",
                        "function": {"name": tc["name"],
                                     "arguments": json.dumps(tc["args"])}}]})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": out[:self.cfg.max_tool_output]})
        else:
            messages.append({"role": "user", "content":
                             "Tool budget exhausted. Give your final evidence "
                             "summary as JSON now: "
                             '{"evidence": [...], "suspect_commits": [...]}'})
            final = self.llm.chat(messages)["content"]
        return final, evidence


def run_triage(llm: LLM, rep: BugReport, log,
               skills: list[Skill] | None = None) -> dict:
    log("[1/3] TRIAGE agent")
    if skills:
        log(f"      {len(skills)} skills loaded: "
            + ", ".join(s.name for s in skills))
    body = rep.body[:24000]
    prompt = f"""Analyze this Linux kernel bug report and return ONLY a JSON object.

Subject: {rep.subject}
From: {rep.from_addr}
Crash type detected by pre-parser: {rep.crash_type}
{skills_menu(skills or [])}
--- BODY (truncated) ---
{body}
--- END BODY ---

--- EXTRACTED TRACE ---
{rep.trace or "(none found)"}
---

Return JSON with keys:
  "summary": one-paragraph summary,
  "subsystem": primary subsystem (e.g. "btrfs / zoned"),
  "components": [suspect files or areas],
  "severity": "critical|high|medium|low",
  "crash_type": "panic|oops|warning|hung_task|data_corruption|io_error|behavioral",
  "reproducible": true/false/"unknown",
  "first_bad_version": version string or null if not mentioned,
  "crash_signature": the single most identifying line/function of the trace,
  "stack_frames": [key functions from the call trace],
  "applicable_skills": [names of skills from the list above that are relevant
                        to this bug, empty list if none apply],
  "questions": [questions that must be answered to find the root cause],
  "search_hints": [grep patterns / lore queries worth trying, 4-8 items]"""
    out = llm.chat([{"role": "system", "content": SYSTEM_HINTS +
                     " Answer strictly in JSON."},
                    {"role": "user", "content": prompt}])
    result = _json_block(out["content"]) or {"summary": out["content"]}
    log(f"      -> {result.get('subsystem', '?')} / "
        f"{result.get('crash_type', '?')} / sev={result.get('severity', '?')}"
        + (f" / skills={result.get('applicable_skills')}"
           if result.get("applicable_skills") else ""))
    return result


def run_context(llm: LLM, cfg: Config, rep: BugReport, triage: dict, log,
                skills: list[Skill] | None = None) -> dict:
    log("[2/3] CONTEXT agent (tools: git, grep, lore)")
    agent = ToolAgent(llm, cfg, build_tools(cfg), log)
    hints = "\n".join(f"- {h}" for h in triage.get("search_hints", []))
    questions = "\n".join(f"- {q}" for q in triage.get("questions", []))
    frames = ", ".join(str(f) for f in triage.get("stack_frames", []))
    system = SYSTEM_HINTS + skills_menu(skills or []) + f"""

You are the EVIDENCE-GATHERING agent. Investigate the local kernel git tree
({cfg.repo}) and lore.kernel.org to collect facts about this bug.
Use the tools; do not guess. Prioritize:
1. Recent commits (last ~6 months) touching the suspect files.
2. git blame on functions that appear in the crash signature / stack frames.
3. Whether the crash signature matches a known upstream discussion or fix.
4. For each suspect commit, check whether a later fix already exists
   (git log --grep, git show).

Finish with a JSON object:
{{"evidence": [{{"fact": "...", "source": "tool + query"}}],
  "suspect_commits": [{{"commit": "...", "subject": "...", "why_suspect": "..."}}]}}"""
    user = f"""Bug: {rep.subject}
Analyzing branch/ref: {cfg.branch or 'HEAD (checked-out working tree)'}
Crash signature: {triage.get('crash_signature')}
Stack frames: {frames}
Suspect components: {triage.get('components')}
Search hints:
{hints}
Questions to answer:
{questions}
Pre-parsed crash type: {rep.crash_type}
First bad version (if mentioned): {triage.get('first_bad_version')}

Check the tree version with the kernel_version tool, then gather evidence."""
    content, evidence = agent.run(system, user)
    result = _json_block(content)
    if not result:
        result = {"evidence": [{"fact": content, "source": "agent summary"}]}
    result["_tool_calls"] = evidence
    log(f"      -> {len(result.get('evidence', []))} evidence items, "
        f"{len(result.get('suspect_commits', []))} suspect commits, "
        f"{len(evidence)} tool calls")
    return result


def run_rca(llm: LLM, rep: BugReport, triage: dict, context: dict, log,
            skills: list[Skill] | None = None) -> dict:
    log("[3/3] RCA agent")
    active = skill_bodies(skills or [], triage.get("applicable_skills") or [])
    if active:
        log(f"      active skills: {triage.get('applicable_skills')}")
    evidence = json.dumps(context.get("evidence", []), indent=1)[:20000]
    suspects = json.dumps(context.get("suspect_commits", []), indent=1)[:6000]
    keys = ("summary", "subsystem", "severity", "crash_type",
            "crash_signature", "stack_frames")
    triage_brief = json.dumps({k: triage.get(k) for k in keys})
    prompt = f"""You are the ROOT CAUSE ANALYSIS agent for a Linux kernel bug.
{active}
BUG REPORT
Subject: {rep.subject}
Triage: {triage_brief}
First bad version (if any): {triage.get('first_bad_version')}

EVIDENCE GATHERED
{evidence}

SUSPECT COMMITS
{suspects}

CRASH TRACE
{rep.trace[:8000] or "(none)"}

Produce a rigorous root cause analysis as JSON:
  "confidence": "high|medium|low",
  "root_cause": clear statement of the root cause (or best hypothesis chain),
  "failure_chain": [ordered causal steps: trigger, intermediate steps, crash],
  "code_path": the code path involved (functions, files),
  "culprit_commits": [{{"commit": "...", "reason": "...",
                        "fix_status": "fixed upstream by <sha> | unfixed | unknown"}}],
  "is_regression": true/false/"unknown",
  "known_issue": true if this matches a known upstream issue; include references,
  "proposed_fix": concrete description of what the fix should change,
  "proposed_patch_sketch": minimal code-change sketch or null,
  "workarounds": [user-actionable mitigations (mount options, config, etc.)],
  "missing_info": [what to ask the reporter],
  "reply_draft": a ready-to-send mailing-list reply (polite, technical, asks
                 for missing info if needed, proposes diagnosis and next steps).

Base every claim on the evidence. If uncertain, say so explicitly."""
    out = llm.chat([{"role": "system", "content": SYSTEM_HINTS +
                     " Answer strictly in JSON."},
                    {"role": "user", "content": prompt}])
    result = _json_block(out["content"])
    if not result:
        result = {"root_cause": out["content"], "confidence": "unknown"}
    log(f"      -> confidence={result.get('confidence', '?')}")
    return result


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_markdown(rep: BugReport, triage: dict, context: dict, rca: dict,
                    model: str, repo: str) -> str:
    def md_list(items):
        return "\n".join(f"- {i}" for i in (items or [])) or "- (none)"

    cb = chr(96) * 3  # markdown code fence, built dynamically

    culprit = "\n".join(
        f"- `{c.get('commit')}` - {c.get('reason')} "
        f"(fix status: {c.get('fix_status', 'unknown')})"
        for c in rca.get("culprit_commits", [])) or "- (none identified)"
    chain = " ->\n".join(str(s) for s in rca.get("failure_chain", [])) or "n/a"
    evidence = "\n".join(
        f"- {e.get('fact')} _(source: {e.get('source')})_"
        for e in context.get("evidence", [])) or "- (none)"
    sketch = rca.get("proposed_patch_sketch") or ""
    reply = rca.get("reply_draft") or ""
    used_skills = md_list(triage.get("applicable_skills"))

    return f"""# Kernel Bug RCA: {rep.subject}

*Reported by:* {rep.from_addr or 'unknown'} | *Date:* {rep.date or 'unknown'}
*Source:* `{os.path.basename(rep.source_file)}` | *Crash type:* {rep.crash_type}
*Analysis model:* `{model}` | *Tree:* `{repo}`

## 1. Triage

**Summary.** {triage.get('summary')}

| Field | Value |
|---|---|
| Subsystem | {triage.get('subsystem')} |
| Severity | {triage.get('severity')} |
| Crash type | {triage.get('crash_type')} |
| Crash signature | `{triage.get('crash_signature')}` |
| Reproducible | {triage.get('reproducible')} |
| First bad version | {triage.get('first_bad_version')} |

**Stack frames.** {', '.join(str(f) for f in triage.get('stack_frames', []))}

**Skills applied.** {used_skills}

## 2. Evidence ({len(context.get('_tool_calls', []))} tool calls)

{evidence}

## 3. Root Cause Analysis

**Confidence:** {rca.get('confidence', 'unknown')}

{rca.get('root_cause')}

**Failure chain**

{chain}

**Code path.** {rca.get('code_path')}

**Culprit commits**

{culprit}

**Regression:** {rca.get('is_regression', 'unknown')} |
**Known issue:** {rca.get('known_issue', 'unknown')}

## 4. Proposed fix

{rca.get('proposed_fix')}

{cb}text
{sketch}
{cb}

**Workarounds**

{md_list(rca.get('workarounds'))}

**Missing info to request**

{md_list(rca.get('missing_info'))}

## 5. Reply draft

{cb}text
{reply}
{cb}

---
*Generated by kernel-bug-rca; verify culprit commits before acting.*
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Agentic RCA for Linux kernel bug reports (mbox or txt).")
    ap.add_argument("report", help="bug report file (.mbox or .txt)")
    ap.add_argument("--repo", default=".", help="local kernel git tree path")
    ap.add_argument("--branch", default=os.environ.get("KERNEL_RCA_BRANCH", ""),
                    help="git ref/branch to analyze, e.g. 'for-next' or "
                         "'v6.12' (default: checked-out HEAD; grep and file "
                         "reads use the branch content, not the working "
                         "tree, without touching your checkout)")
    ap.add_argument("--skills", default=os.environ.get("KERNEL_RCA_SKILLS", ""),
                    help="directory with skills (SKILL.md folders or .md files)")
    ap.add_argument("--backend",
                    choices=["ollama", "vllm", "mistral", "claude"],
                    default="ollama")
    ap.add_argument("--base-url", default="", help="override endpoint base URL")
    ap.add_argument("--model", default="",
                    help="model name (default per backend: ollama=qwen3:32b, "
                         "vllm=Qwen3-32B, mistral=mistral-large-latest, "
                         "claude=claude-sonnet-4-5)")
    ap.add_argument("--api-key", default="",
                    help="API key for mistral/claude (prefer env vars "
                         "MISTRAL_API_KEY / ANTHROPIC_API_KEY)")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="max output tokens (required by the claude backend)")
    ap.add_argument("--max-turns", type=int, default=10,
                    help="max tool-call rounds for the context agent")
    ap.add_argument("--out", default="", help="output markdown file")
    ap.add_argument("--json-out", default="", help="output JSON file")
    ap.add_argument("--no-json-mode", action="store_true",
                    help="disable response_format json_object "
                         "(for models that don't support it)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config(backend=args.backend, base_url=args.base_url,
                 model=args.model, repo=args.repo, max_turns=args.max_turns,
                 skills_dir=args.skills, branch=args.branch,
                 api_key=args.api_key, max_tokens=args.max_tokens,
                 json_mode=not args.no_json_mode)
    cfg.resolve()

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    skills = load_skills(cfg.skills_dir)
    if cfg.skills_dir and not skills:
        log(f"warning: no skills found in {cfg.skills_dir}")

    if not os.path.isdir(os.path.join(cfg.repo, ".git")):
        log(f"warning: {cfg.repo} does not look like a git tree")
    if cfg.backend in ("mistral", "claude") and not cfg.api_key:
        log(f"error: {cfg.backend} backend needs an API key; set "
            f"{'MISTRAL_API_KEY' if cfg.backend == 'mistral' else 'ANTHROPIC_API_KEY'} "
            f"or use --api-key")
        return 2
    if cfg.branch:
        probe = subprocess.run(["git", "-C", cfg.repo, "rev-parse",
                                "--verify", cfg.branch],
                               capture_output=True, text=True)
        if probe.returncode != 0:
            log(f"error: branch/ref '{cfg.branch}' not found in {cfg.repo}: "
                f"{probe.stderr.strip()}")
            return 2

    log(f"kernel-bug-rca | model={cfg.model} backend={cfg.backend} "
        f"({cfg.base_url}) repo={cfg.repo}"
        + (f" branch={cfg.branch}" if cfg.branch else "")
        + (f" skills={cfg.skills_dir} ({len(skills)})" if skills else ""))
    rep = ingest(args.report)
    log(f"Ingested {args.report}: '{rep.subject}' "
        f"({rep.metadata.get('n_messages', 1)} msg(s), "
        f"crash={rep.crash_type}, trace={'yes' if rep.trace else 'no'})")

    llm = LLM(cfg)
    t0 = time.time()
    try:
        triage = run_triage(llm, rep, log, skills)
        context = run_context(llm, cfg, rep, triage, log, skills)
        rca = run_rca(llm, rep, triage, context, log, skills)
    except urllib.error.URLError as e:
        log(f"error: cannot reach LLM backend at {cfg.base_url}: {e}")
        return 2

    md = render_markdown(rep, triage, context, rca, cfg.model, cfg.repo)
    out = args.out or os.path.splitext(args.report)[0] + "-rca.md"
    with open(out, "w") as f:
        f.write(md)
    log(f"RCA written to {out} ({llm.calls} LLM calls, {time.time() - t0:.0f}s)")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"report": asdict(rep), "triage": triage,
                       "context": {k: v for k, v in context.items()
                                   if k != "_tool_calls"},
                       "rca": rca}, f, indent=2)
        log(f"JSON written to {args.json_out}")

    # Print the reply draft to stdout for immediate use.
    print(rca.get("reply_draft", "(no reply draft produced)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
