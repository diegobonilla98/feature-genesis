from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_research_report(
    root: str | Path,
    title: str,
    sections: list[tuple[str, Any]],
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cards = []
    for heading, content in sections:
        if isinstance(content, str):
            body = f"<p>{html.escape(content)}</p>"
        else:
            body = f"<pre>{html.escape(json.dumps(content, indent=2, sort_keys=True))}</pre>"
        cards.append(f"<section><h2>{html.escape(heading)}</h2>{body}</section>")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Inter, system-ui, sans-serif; max-width: 1180px; margin: 0 auto; padding: 40px 24px; background: #f5f6f8; color: #16181d; }}
h1 {{ font-size: 42px; margin: 0 0 28px; }}
section {{ background: white; border: 1px solid #dfe3ea; border-radius: 14px; padding: 24px; margin: 18px 0; box-shadow: 0 3px 16px rgba(0,0,0,.04); }}
h2 {{ margin-top: 0; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #11151c; color: #e8edf5; padding: 18px; border-radius: 10px; overflow: auto; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{''.join(cards)}
</body>
</html>
"""
    path = root / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
