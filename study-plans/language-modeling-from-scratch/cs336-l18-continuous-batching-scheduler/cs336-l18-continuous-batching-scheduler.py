def continuous_batching_scheduler(requests, kv_capacity):
    """
    Returns: ordered decode-step records
    """
    pending = [dict(request, order=index) for index, request in enumerate(requests)]
    pending.sort(key=lambda request: (request["arrival_step"], request["order"]))
    next_request = 0
    waiting = []
    active = []
    records = []
    if not pending:
        return records
    step = pending[0]["arrival_step"]

    while next_request < len(pending) or waiting or active:
        if not waiting and not active and next_request < len(pending):
            step = max(step, pending[next_request]["arrival_step"])
        while next_request < len(pending) and pending[next_request]["arrival_step"] <= step:
            waiting.append(pending[next_request])
            next_request += 1

        admitted = []
        finished = []

        def reserved_tokens():
            return sum(
                request["prompt_tokens"] + request["max_new_tokens"]
                for request in active
            )

        def admit_waiting():
            remaining = []
            for request in waiting:
                footprint = request["prompt_tokens"] + request["max_new_tokens"]
                if reserved_tokens() + footprint <= kv_capacity:
                    admitted.append(request["id"])
                    if request["max_new_tokens"] == 0:
                        finished.append(request["id"])
                    else:
                        active.append(dict(request, generated=0))
                else:
                    remaining.append(request)
            waiting[:] = remaining

        admit_waiting()
        completed = []
        for request in active:
            request["generated"] += 1
            if request["generated"] == request["max_new_tokens"]:
                completed.append(request)
        for request in completed:
            active.remove(request)
            finished.append(request["id"])
        admit_waiting()

        records.append({
            "step": step,
            "admitted": admitted,
            "active": [request["id"] for request in active],
            "finished": finished,
            "kv_tokens_in_use": sum(
                request["prompt_tokens"] + request["generated"]
                for request in active
            ),
        })
        step += 1
    return records
