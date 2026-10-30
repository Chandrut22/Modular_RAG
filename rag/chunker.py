

import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rag.loader import load_langchain_docs
from langchain_core.documents import Document

FENCE_RE = re.compile(r"^\s*```")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
ANNOT_RE = re.compile(r"\s*(?:#|//)\s*\[!code[^\]]*\]\s*$")  # "# [!code highlight]"
CARD_RE = re.compile(r"<Card\b.*?</Card>", re.S)
MDX_TAG_RE = re.compile(
    r"</?(?:Tip|Note|Warning|Info|Callout|CodeGroup|Icon|Card|Tabs|Tab|Steps|Step|div)\b[^>]*>"
)
RULE_RE = re.compile(r"^\s*(?:\*\*\*|---)\s*$", re.M)
MODEL_RE = re.compile(r'\bmodel(?:_name)?\s*=\s*"[^"]*"')

Block = Tuple[str, ...]


# ---------------------------------------------------------------------------
# 1. Split raw text into alternating prose / fenced-code segments
# ---------------------------------------------------------------------------
def split_segments(text: str) -> List[Tuple[bool, str]]:
    """Return [(is_code, text)]. Fences are detected line by line, so a '#'
    or '<Tag>' inside code is never mistaken for markdown/MDX."""
    segments: List[Tuple[bool, str]] = []
    buf: List[str] = []
    in_code = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            if not in_code:
                if buf:
                    segments.append((False, "\n".join(buf)))
                buf, in_code = [line], True
            else:
                buf.append(line)
                segments.append((True, "\n".join(buf)))
                buf, in_code = [], False
        else:
            buf.append(line)
    if buf:
        segments.append((in_code, "\n".join(buf)))
    return segments


def clean_prose(s: str) -> str:
    s = CARD_RE.sub("", s)          # "View example trace" link cards
    s = MDX_TAG_RE.sub("", s)       # <Tip>, <Icon .../>, <CodeGroup>, ...
    s = RULE_RE.sub("", s)          # *** / --- horizontal rules
    return s


def parse_code(block: str) -> Tuple[str, str, bool]:
    """-> (language, body, is_output_block). Strips fence info-string junk
    (theme={...}, title=...) and '# [!code highlight]' annotations."""
    lines = textwrap.dedent(block).splitlines()
    opening = lines[0]
    m = re.match(r"^\s*```(\w*)", opening)
    lang = m.group(1) if m else ""
    is_output = 'title="Output"' in opening
    body_lines = lines[1:]
    if body_lines and FENCE_RE.match(body_lines[-1]):
        body_lines = body_lines[:-1]
    body = "\n".join(ANNOT_RE.sub("", ln) for ln in body_lines).strip("\n")
    return lang, body, is_output


def strip_leading_blockquote(text: str) -> str:
    """Drop a leading '> Documentation Index ...' banner before the first heading."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith(">") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:])


# ---------------------------------------------------------------------------
# 2. Build sections: heading path + ordered blocks
# ---------------------------------------------------------------------------
def build_sections(text: str, skip_output: bool = False) -> List[Dict]:
    sections: List[Dict] = []
    stack: List[Tuple[int, str]] = []
    current: Dict = {"path": "", "blocks": []}
    para: List[str] = []
    last_code_mask: Optional[str] = None  # for de-duplicating tabbed samples

    def flush_para():
        nonlocal para, last_code_mask
        chunk = "\n".join(para).strip()
        if chunk:
            current["blocks"].append(("text", chunk))
            last_code_mask = None  
        para = []

    for is_code, seg in split_segments(strip_leading_blockquote(text)):
        if is_code:
            flush_para()
            lang, body, is_output = parse_code(seg)
            if not body.strip() or (skip_output and is_output):
                continue
            mask = MODEL_RE.sub('model=""', body)
            if mask == last_code_mask:  
                continue
            last_code_mask = mask
            current["blocks"].append(("code", lang, body))
            continue

        for line in clean_prose(seg).splitlines():
            m = HEADING_RE.match(line)
            if m:
                flush_para()
                level, title = len(m.group(1)), m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                if current["blocks"]:
                    sections.append(current)
                current = {"path": " > ".join(t for _, t in stack), "blocks": []}
                last_code_mask = None
            elif not line.strip():
                flush_para()
            else:
                para.append(line)
        flush_para()

    if current["blocks"]:
        sections.append(current)
    return sections


# ---------------------------------------------------------------------------
# 3. Pack blocks into size-bounded chunks
# ---------------------------------------------------------------------------
def render(block: Block) -> str:
    if block[0] == "code":
        return f"```{block[1]}\n{block[2]}\n```"
    return block[1]


def split_big_code(lang: str, body: str, max_chars: int) -> List[Block]:
    """Split an oversized code block at blank lines; every piece stays fenced."""
    groups: List[str] = []
    for g in re.split(r"\n\s*\n", body):
        if len(g) <= max_chars:
            groups.append(g)
            continue
        # one giant blank-line-free group -> fall back to splitting by lines
        buf: List[str] = []
        for ln in g.splitlines():
            if buf and len("\n".join(buf + [ln])) > max_chars:
                groups.append("\n".join(buf))
                buf = []
            buf.append(ln)
        if buf:
            groups.append("\n".join(buf))
    pieces: List[str] = []
    cur: List[str] = []
    for g in groups:
        if cur and len("\n\n".join(cur + [g])) > max_chars:
            pieces.append("\n\n".join(cur))
            cur = []
        cur.append(g)
    if cur:
        pieces.append("\n\n".join(cur))
    return [("code", lang, p) for p in pieces]


def split_big_text(text: str, max_chars: int) -> List[Block]:
    parts = re.split(r"(?<=[.!?])\s+|\n(?=[*\-] )", text)
    pieces: List[str] = []
    cur = ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > max_chars:
            pieces.append(cur)
            cur = ""
        cur = f"{cur} {p}".strip() if cur else p
    if cur:
        pieces.append(cur)
    return [("text", p) for p in pieces]


def pack_section(
    blocks: List[Block], max_chars: int, max_code_chars: int
) -> List[Tuple[List[Block], str]]:
    """-> [(blocks_in_chunk, chunk_type)]"""
    units: List[Block] = []
    for b in blocks:
        if b[0] == "code" and len(render(b)) > max_code_chars:
            units.extend(split_big_code(b[1], b[2], max_code_chars))
        elif b[0] == "text" and len(b[1]) > max_chars:
            units.extend(split_big_text(b[1], max_chars))
        else:
            units.append(b)

    out: List[Tuple[List[Block], str]] = []
    cur: List[Block] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            kinds = {b[0] for b in cur}
            ctype = "code" if kinds == {"code"} else "text" if kinds == {"text"} else "mixed"
            out.append((cur, ctype))
        cur, cur_len = [], 0

    for u in units:
        size = len(render(u)) + 2
        if cur and cur_len + size > max_chars:
            flush()
        cur.append(u)
        cur_len += size
    flush()
    return out


# ---------------------------------------------------------------------------
# 4. Public API
# ---------------------------------------------------------------------------
def chunk_markdown(
    text: str,
    source: str = "",
    max_chars: int = 1200,
    max_code_chars: int = 2000,
    skip_output: bool = False,
) -> List[Document]:
    chunks: List[Document] = []
    seen = set()
    for section in build_sections(text, skip_output=skip_output):
        path = section["path"]
        for blocks, ctype in pack_section(section["blocks"], max_chars, max_code_chars):
            body = "\n\n".join(render(b) for b in blocks)
            if not re.search(r"\w", body):
                continue
            fingerprint = re.sub(r"\s+", " ", MODEL_RE.sub('model=""', body)).lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            content = f"{path}\n\n{body}" if path else body  # contextual header
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "section": path,
                        "chunk_type": ctype,
                        "chunk_index": len(chunks),
                    },
                )
            )
    return chunks


def load_markdown_chunks(path: str, **kwargs) -> List[Document]:
    # encoding="utf-8" matters on Windows: the default (cp1252) breaks on
    # characters like ’ — … that appear in most docs.
    text = Path(path).read_text(encoding="utf-8")
    return chunk_markdown(text, source=Path(path).name, **kwargs)


if __name__ == "__main__":
    docs = load_langchain_docs()

    if not docs:
        print("No documents were loaded.")
        sys.exit(1)

    # Test chunking the first downloaded document
    source_doc = docs[0]

    chunks = chunk_markdown(
        source_doc.page_content,
        source=source_doc.metadata["source"],
        max_chars=1200,
        max_code_chars=2000,
        skip_output=False,
    )

    print(f"Source: {source_doc.metadata['source']}")
    print(f"{len(chunks)} chunks\n")

    for d in chunks:
        m = d.metadata

        print(
            f"[{m['chunk_index']:>3}] "
            f"{m['chunk_type']:<5} "
            f"{len(d.page_content):>5} chars | "
            f"{m['section']}"
        )

