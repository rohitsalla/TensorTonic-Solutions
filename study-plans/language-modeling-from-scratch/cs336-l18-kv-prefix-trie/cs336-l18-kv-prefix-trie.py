def kv_prefix_trie(cached_sequences, requests):
    """
    Returns: dictionary containing named reusable-prefix matches
    """
    # Trie: list of dicts mapping token -> child node id
    # Node 0 is root
    children = [{}]   # children[node_id] = {token: child_node_id}
    next_id  = 1

    # ── Insert cached sequences ───────────────────────────────────────────
    for seq in cached_sequences:
        node = 0
        for token in seq:
            if token not in children[node]:
                children[node][token] = next_id
                children.append({})
                next_id += 1
            node = children[node][token]

    # ── Match each request ────────────────────────────────────────────────
    matches = []
    for req in requests:
        node   = 0
        depth  = 0
        for token in req:
            if token not in children[node]:
                break
            node  = children[node][token]
            depth += 1

        matches.append({
            "matched_length":  depth,
            "cache_node":      node,
            "uncached_suffix": list(req[depth:]),
        })

    return {"matches": matches}