"""Display-only normalization; stored evidence remains unchanged."""


def clean_text(text):
    for char in ("…", "–", "—", "’", "‘", "“", "”", "\u00a0"):
        broken = "".join(
            bytes([b]).decode("cp1252", errors="surrogateescape") for b in char.encode("utf-8")
        )
        text = text.replace(broken, char)
    return text.replace("\u00a0", " ")
