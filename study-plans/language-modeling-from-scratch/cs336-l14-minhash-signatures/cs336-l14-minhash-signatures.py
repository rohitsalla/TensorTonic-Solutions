import hashlib
import json
import numpy as np

def minhash_signatures(documents, shingle_size, num_hashes, seed):
    """
    Returns: dictionary containing signature and estimated-similarity arrays
    """
    EMPTY_VAL = (1 << 32) - 1   # 2^32 - 1
    seed_str  = str(seed)
    n_docs    = len(documents)

    # ── Build shingle sets ────────────────────────────────────────────────
    shingle_sets = []
    for tokens in documents:
        shingles = set()
        for i in range(len(tokens) - shingle_size + 1):
            window  = tokens[i : i + shingle_size]
            encoded = json.dumps(window, ensure_ascii=False, separators=(',', ':'))
            shingles.add(encoded)
        shingle_sets.append(shingles)

    # ── Compute MinHash signatures ────────────────────────────────────────
    # signatures[doc, hash_col]
    signatures = np.full((n_docs, num_hashes), EMPTY_VAL, dtype=np.uint32)

    for doc_idx, shingles in enumerate(shingle_sets):
        if not shingles:
            continue   # stays EMPTY_VAL for all columns
        for h in range(num_hashes):
            prefix = f"{seed_str}:{h}:".encode("utf-8")
            min_val = EMPTY_VAL
            for shingle in shingles:
                data   = prefix + shingle.encode("utf-8")
                digest = hashlib.sha256(data).digest()
                val    = int.from_bytes(digest[:4], "big")
                if val < min_val:
                    min_val = val
            signatures[doc_idx, h] = min_val

    # ── Estimate Jaccard similarities ─────────────────────────────────────
    similarities = np.zeros((n_docs, n_docs), dtype=np.float64)

    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                similarities[i, j] = 1.0
                continue
            si_empty = len(shingle_sets[i]) == 0
            sj_empty = len(shingle_sets[j]) == 0
            if si_empty and sj_empty:
                similarities[i, j] = 1.0
            elif si_empty or sj_empty:
                similarities[i, j] = 0.0
            else:
                matches = np.sum(signatures[i] == signatures[j])
                similarities[i, j] = matches / num_hashes

    return {
        "signatures":   signatures,
        "similarities": similarities,
    }