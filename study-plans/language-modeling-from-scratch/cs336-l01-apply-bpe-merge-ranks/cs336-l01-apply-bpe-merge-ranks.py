def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Apply learned BPE merge rules to UTF-8 byte IDs in priority order.
    Returns: list[int] containing token IDs after applying the ordered merge rules
    """
    # Start with UTF-8 byte IDs (each byte becomes one token ID 0-255)
    ids = list(text.encode("utf-8"))

    for left_id, right_id, new_id in merges:
        # Scan left to right, replacing non-overlapping matches
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == left_id and ids[i + 1] == right_id:
                new_ids.append(new_id)
                i += 2          # consume both tokens, skip overlap check
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids

    return ids


def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Reconstruct text by concatenating vocabulary bytes for each token ID.
    Returns: the Unicode string reconstructed from token IDs and vocabulary bytes
    """
    # Gather raw bytes from each token's vocab entry, then decode the full sequence
    raw_bytes = []
    for token_id in ids:
        raw_bytes.extend(vocab[token_id])
    return bytes(raw_bytes).decode("utf-8")