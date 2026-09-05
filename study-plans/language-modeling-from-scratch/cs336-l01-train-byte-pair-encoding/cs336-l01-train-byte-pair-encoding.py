from collections import defaultdict

def train_bpe(corpus, vocab_size):
    """
    Train byte-level BPE from a list of strings.
    
    Starts with 256 single-byte tokens, repeatedly merges the most frequent
    adjacent pair (ties broken by lexicographically greater byte string pair)
    until vocab_size is reached or no adjacent pair remains.
    
    Returns: dict with keys:
        'vocab'  - [[token_id, byte_values_list], ...] for learned tokens only
        'merges' - [[left_id, right_id, new_id], ...] in creation order
    """
    # No merges possible if target is at or below base vocab
    if vocab_size <= 256:
        return {"vocab": [], "merges": []}

    # token_id -> bytes object for all tokens (base + learned)
    token_bytes = {i: bytes([i]) for i in range(256)}

    next_id = 256
    vocab_learned = []  # [token_id, list(bytes)]
    merges = []         # [left_id, right_id, new_id]

    # UTF-8 encode each corpus string into a mutable list of token ids
    sequences = [list(s.encode('utf-8')) for s in corpus]

    def count_pairs(seqs):
        """Count all adjacent (non-crossing) token pairs across sequences."""
        counts = defaultdict(int)
        for seq in seqs:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        return counts

    def pair_sort_key(pair):
        """Tie-break key: lexicographically greater byte string pair wins."""
        return (token_bytes[pair[0]], token_bytes[pair[1]])

    def apply_merge(seqs, left_id, right_id, new_id):
        """Replace all non-overlapping left-to-right occurrences of (left,right)."""
        new_seqs = []
        for seq in seqs:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == left_id and seq[i + 1] == right_id:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_seqs.append(new_seq)
        return new_seqs

    while next_id < vocab_size:
        counts = count_pairs(sequences)
        if not counts:
            break

        max_freq = max(counts.values())

        # Among max-frequency pairs, pick the lexicographically greatest byte pair
        best_pair = max(
            (pair for pair, freq in counts.items() if freq == max_freq),
            key=pair_sort_key
        )

        left_id, right_id = best_pair
        new_id = next_id
        next_id += 1

        # New token concatenates the bytes of both sides
        token_bytes[new_id] = token_bytes[left_id] + token_bytes[right_id]

        vocab_learned.append([new_id, list(token_bytes[new_id])])
        merges.append([left_id, right_id, new_id])

        sequences = apply_merge(sequences, left_id, right_id, new_id)

    return {"vocab": vocab_learned, "merges": merges}