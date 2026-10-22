# kernel-bug-rca

Agentic root-cause analysis for Linux kernel bug reports. Feed it an `.mbox`
or plain-text bug report (from a mailing list, a user, or your CI system) and
it runs a multi-agent pipeline that triages the report, gathers evidence from
your local kernel git tree and lore.kernel.org, and produces a structured RCA
report with a ready-to-send reply draft.

Focused on the filesystem / zoned-block-device stack (btrfs, xfs, zonefs,
ZNS, blk-mq, dm), but works for any kernel subsystem.

## How it works

```
bug report (.mbox/.txt)
        |
        v
  1. INGEST   -- parse mail headers, body, MIME, extract OOPS/WARNING traces
        |
        v
  2. TRIAGE   -- LLM: subsystem, severity, crash signature, stack frames,
        |         applicable skills, questions to answer, search hints
        v
  3. CONTEXT  -- tool-calling agent over your kernel tree:
        |         git log / git show / git blame / git grep / file reads,
        |         lore.kernel.org full-text search
        v
  4. RCA      -- LLM: root cause, failure chain, culprit commits (+ fix
                  status), proposed fix & patch sketch, workarounds,
                  missing info, mailing-list reply draft
```

Output: a Markdown report (`<report>-rca.md`), optional JSON, and the reply
draft printed to stdout.

## Requirements

- Python 3.10+ (stdlib only — no pip dependencies)
- A git checkout of the kernel tree you want evidence gathered from
- One LLM backend:

| Backend | Endpoint | Auth |
|---|---|---|
| Ollama (default) | `http://localhost:11434/v1` | none |
| vLLM | `http://localhost:8000/v1` | optional `VLLM_API_KEY` |
| Mistral | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` |
| Claude | `https://api.anthropic.com` (native Messages API) | `ANTHROPIC_API_KEY` |

## Quick start

```sh
python3 kernel_bug_rca.py nightly-report.mbox \
    --repo ~/src/linux --branch for-next \
    --backend vllm --model Qwen3-32B \
    --out rca.md --json-out rca.json
```

The reply draft is printed to stdout; the full report goes to `--out`
(default: next to the input file).

### CLI reference

```
kernel_bug_rca.py report [--repo REPO] [--branch REF] [--skills DIR]
    [--backend {ollama,vllm,mistral,claude}] [--base-url URL] [--model M]
    [--api-key KEY] [--max-tokens N] [--max-turns N]
    [--out FILE] [--json-out FILE] [--no-json-mode] [-q]
```

Key options:

- `--branch` — analyze a ref other than the checked-out HEAD (e.g.
  `for-next`, `v6.12`, `origin/master`). All tools then read that ref's
  content (`git grep`, `git show <ref>:<path>`, ...) without touching your
  checkout.
- `--skills DIR` — reusable instruction packs (see below).
- `--max-turns` — tool-call budget for the evidence agent (default 10).
- `--no-json-mode` — for local models that lack JSON response support.

### Environment variables

| Variable | Purpose |
|---|---|
| `OLLAMA_URL` / `VLLM_URL` / `MISTRAL_URL` / `ANTHROPIC_URL` | override endpoints |
| `MISTRAL_API_KEY` / `ANTHROPIC_API_KEY` / `VLLM_API_KEY` | backend auth |
| `KERNEL_RCA_SKILLS` | default skills directory |
| `KERNEL_RCA_BRANCH` | default ref/branch to analyze |

## Skills

`--skills DIR` loads reusable instruction packs that tune agent behaviour:

- Claude-style folders: `<skills>/<name>/SKILL.md` with `name:` /
  `description:` frontmatter
- Flat snippets: `<skills>/<anything>.md`

The triage agent sees a menu of loaded skills and selects the applicable
ones (`applicable_skills` in the JSON output); their full bodies are then
injected into the RCA agent's prompt. Checklists and known-pattern catalogs
work well; interactive workflows ("ask the user before...") do not.

## MCP server

`kernel_bug_rca_mcp.py` exposes the pipeline as MCP tools over stdio:

- `kernel_bug_rca` — full analysis (report path or raw text)
- `kernel_bug_rca_draft` — fast triage-only pass (single LLM call)
- `kernel_bug_rca_search` — no LLM: lore + `git log --grep` lookup

Server config for an MCP client (e.g. Devin Desktop / Windsurf):

```json
"kernel-bug-rca": {
  "command": "python3",
  "args": ["/home/you/bin/kernel_bug_rca_mcp.py", "serve"],
  "env": {
    "KERNEL_RCA_SCRIPT": "/home/you/bin/kernel_bug_rca.py",
    "KERNEL_RCA_REPO": "/home/you/src/linux",
    "KERNEL_RCA_BACKEND": "vllm",
    "KERNEL_RCA_MODEL": "Qwen3-32B",
    "KERNEL_RCA_SKILLS": "/home/you/.kernel-skills"
  }
}
```

## mutt integration

Pipe the current message straight into an analysis:

```
# muttrc
macro index,pager <tab>B \
  "<pipe-message>python3 ~/.local/bin/kernel_bug_rca_mcp.py pipe<enter>" \
  "run kernel bug RCA on this message"
```

The reply draft is printed to the terminal; the full report lands in
`~/rca/rca-<stamp>.md` (override with `KERNEL_RCA_OUT_DIR`).

## CI / cron

```sh
python3 kernel_bug_rca_mcp.py call /mail/reports/10532.mbox --repo ~/src/linux
```

Caveat: with the `mistral`/`claude` backends the report content (mbox,
dmesg logs) leaves your network. For internal test reports, prefer the
local Ollama/vLLM path.

## License

GPL-2.0-or-later. Source files carry:

```
# SPDX-License-Identifier: GPL-2.0-or-later
```

## Disclaimer

Automated output is a starting point, not a substitute for review — verify
culprit commits and proposed fixes before acting on them.

The whole project is vibe coded using [Mistral](https://chat.mistral.ai)
