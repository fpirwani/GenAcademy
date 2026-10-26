"""Never accept model-created source identities or altered quotations."""


def validate_claims(claims, passages):
    by_ref = {p["ref"]: p for p in passages}
    accepted = []
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            continue
        sources = []
        evidence = claim.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            continue
        for item in evidence:
            if not isinstance(item, dict):
                break
            passage = by_ref.get(item.get("ref"))
            quote = item.get("quote")
            if (
                not passage
                or not isinstance(quote, str)
                or len(quote.strip()) < 15
                or " ".join(quote.split()) not in " ".join(passage["text"].split())
            ):
                break
            sources.append({**passage, "quote": quote})
        else:
            if claim["text"].strip():
                accepted.append({**claim, "sources": sources})
    return accepted
