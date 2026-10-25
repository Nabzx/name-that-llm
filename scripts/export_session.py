"""Export a Claude Code session for the AI sessions requirement.

Writes a zip with the raw transcript (the complete log), any subagent
transcripts, and a readable Markdown version of the conversation.

    python scripts/export_session.py <session-id> [--name NAME]

The transcript is looked up under ~/.claude/projects/*/<session-id>.jsonl.
Output goes to ai-sessions/ in the repo, so the logs travel with the code. Pass --out to write elsewhere.
"""

import argparse
import json
import zipfile
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"
OUT = Path(__file__).resolve().parent.parent / "ai-sessions"
RESULT_CAP = 3000  # tool results are cut in the Markdown only, never in the raw file


def find_transcript(session_id):
    hits = list(PROJECTS.glob(f"*/{session_id}.jsonl"))
    assert len(hits) == 1, f"expected one transcript for {session_id}, found {hits}"
    return hits[0]


def render(jsonl):
    out = []
    for line in jsonl.open():
        entry = json.loads(line)
        kind = entry.get("type")
        if kind not in ("user", "assistant"):
            continue
        content = entry.get("message", {}).get("content")
        stamp = entry.get("timestamp", "")[:19].replace("T", " ")
        if isinstance(content, str):
            out.append(f"## {'Nabil' if kind == 'user' else 'Claude'} ({stamp} UTC)\n\n{content}\n")
            continue
        for block in content or []:
            t = block.get("type")
            if t == "text" and block.get("text", "").strip():
                who = "Nabil" if kind == "user" else "Claude"
                out.append(f"## {who} ({stamp} UTC)\n\n{block['text']}\n")
            elif t == "tool_use":
                args = json.dumps(block.get("input", {}), indent=1, ensure_ascii=False)
                out.append(f"**Tool call: {block.get('name')}**\n\n```json\n{args}\n```\n")
            elif t == "tool_result":
                body = block.get("content")
                if isinstance(body, list):
                    body = "\n".join(b.get("text", "") for b in body if isinstance(b, dict))
                body = str(body or "")
                if len(body) > RESULT_CAP:
                    body = body[:RESULT_CAP] + f"\n... ({len(body) - RESULT_CAP} more characters in the raw transcript)"
                out.append(f"**Tool result**\n\n```\n{body}\n```\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    jsonl = find_transcript(args.session_id)
    name = args.name or args.session_id
    out.mkdir(parents=True, exist_ok=True)
    md = out / f"{name}.md"
    md.write_text(render(jsonl))

    zpath = out / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl, f"{name}/transcript.jsonl")
        z.write(md, f"{name}/transcript.md")
        extra = jsonl.with_suffix("")
        if extra.is_dir():
            for f in sorted(extra.rglob("*")):
                if f.is_file():
                    z.write(f, f"{name}/{f.relative_to(extra)}")
    print(f"{zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    print(f"{md} ({md.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
