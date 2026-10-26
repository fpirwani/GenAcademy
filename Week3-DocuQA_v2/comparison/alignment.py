"""Bounded paragraph alignment; original text and locations remain intact."""

import re
from collections import defaultdict, deque
from difflib import SequenceMatcher, unified_diff


def sections(blocks):
    result = []
    for block in blocks:
        # Paragraph-sized spans keep diffing independent of Q&A chunk identifiers.
        for paragraph in re.split(r"\n\s*\n", block["text"]):
            for start in range(0, len(paragraph), 1600):
                text = paragraph[start : start + 1600]
                if text.strip():
                    result.append({**block, "text": text, "section_id": f"s{len(result) + 1}"})
    return result


def normalized(text):
    return " ".join(text.split())


def align(earlier, later):
    old, new = sections(earlier), sections(later)
    buckets = defaultdict(deque)
    for j, item in enumerate(new):
        buckets[normalized(item["text"])].append(j)
    matches, used = {}, set()
    for i, item in enumerate(old):
        candidates = buckets[normalized(item["text"])]
        if candidates:
            j = candidates.popleft()
            matches[i] = (j, 1.0)
            used.add(j)
    # Nearby candidates plus lexical overlap avoid an unbounded all-pairs matcher.
    remaining = [j for j in range(len(new)) if j not in used]
    for i, item in enumerate(old):
        if i in matches:
            continue
        words = set(item["text"].lower().split())
        ranked = sorted(
            (j for j in remaining if j not in used),
            key=lambda j: (-len(words.intersection(new[j]["text"].lower().split())), abs(i - j)),
        )[:12]
        scored = [
            (
                SequenceMatcher(
                    None, normalized(item["text"]), normalized(new[j]["text"]), autojunk=False
                ).ratio(),
                j,
            )
            for j in ranked
        ]
        if scored:
            score, j = max(scored)
            if score >= 0.58:
                matches[i] = (j, score)
                used.add(j)
    records = []
    # Exact matches whose relative order changes are movement candidates.
    exact = [(i, j) for i, (j, score) in matches.items() if score == 1]
    moved = {i for i, j in exact if any((i - k) * (j - other) < 0 for k, other in exact)}
    for i, item in enumerate(old):
        other, confidence = matches.get(i, (None, 0))
        target = new[other] if other is not None else None
        kind = (
            "removed"
            if target is None
            else "moved"
            if i in moved
            else "unchanged"
            if confidence == 1
            else "modified"
        )
        records.append(
            {
                "earlier": item,
                "later": target,
                "kind": kind,
                "confidence": confidence,
                "alignment_uncertain": confidence < 0.8,
                "diff": "\n".join(
                    unified_diff(
                        item["text"].splitlines(),
                        target["text"].splitlines() if target else [],
                        fromfile="earlier",
                        tofile="later",
                        lineterm="",
                    )
                ),
            }
        )
    for j, item in enumerate(new):
        if j not in used:
            records.append(
                {
                    "earlier": None,
                    "later": item,
                    "kind": "added",
                    "confidence": 0,
                    "alignment_uncertain": True,
                    "diff": "\n".join("+" + line for line in item["text"].splitlines()),
                }
            )
    for i, record in enumerate(records):
        record["id"] = f"change-{i + 1}"
    return records
