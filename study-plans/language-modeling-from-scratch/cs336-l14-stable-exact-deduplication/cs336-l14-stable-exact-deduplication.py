import hashlib
import unicodedata
import re

def stable_exact_deduplication(documents, lowercase=True,
                               collapse_whitespace=True, hash_bits=64):
    """
    Returns: dictionary containing retained IDs and duplicate ownership
    """
    def normalize(text):
        # 1. Unicode NFKC
        text = unicodedata.normalize("NFKC", text)
        # 2. Optional case folding
        if lowercase:
            text = text.casefold()
        # 3. Optional whitespace collapse
        if collapse_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    mask = (1 << hash_bits) - 1

    # bucket -> {normalized_text -> earliest retained id}
    buckets: dict[int, dict[str, str]] = {}

    retained_ids       = []
    removed_to_retained = {}

    for doc in documents:
        doc_id  = doc["id"]
        norm    = normalize(doc["text"])

        # Low-order hash_bits bits of SHA-256
        digest  = hashlib.sha256(norm.encode("utf-8")).digest()
        # digest is big-endian; take as integer and mask low bits
        h_int   = int.from_bytes(digest, "big") & mask

        bucket  = buckets.setdefault(h_int, {})

        if norm in bucket:
            # Exact match with an earlier retained document
            removed_to_retained[doc_id] = bucket[norm]
        else:
            # First occurrence of this normalized text in this bucket
            bucket[norm] = doc_id
            retained_ids.append(doc_id)

    return {
        "retained_ids":        retained_ids,
        "removed_to_retained": removed_to_retained,
    }