def canonical_preference_pairs(records):
    """
    Returns: dictionary containing stable winner-loser preference pairs
    """
    pairs = []

    for record in records:
        prompt_id  = record["prompt_id"]
        candidates = sorted(record["candidates"], key=lambda c: c["rank"])

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                pairs.append({
                    "prompt_id": prompt_id,
                    "winner_id": candidates[i]["id"],
                    "loser_id":  candidates[j]["id"],
                })

    return {"pairs": pairs}