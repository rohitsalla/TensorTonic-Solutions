def autograd(operations, input_values):
    """
    Returns: Dict with "output" (float) and "gradients" (list of floats), rounded to 4 decimals.
    """
    class Node:
        def __init__(self, data, prev=()):
            self.data = float(data)
            self.grad = 0.0
            self.prev = prev
            self._backward = lambda: None

    nodes = [Node(v) for v in input_values]
    n_inputs = len(input_values)

    for op in operations:
        if op[0] == "add":
            a, b = nodes[op[1]], nodes[op[2]]
            out = Node(a.data + b.data, (a, b))
            def _back(a=a, b=b, out=out):
                a.grad += out.grad
                b.grad += out.grad
            out._backward = _back
            nodes.append(out)
        elif op[0] == "mul":
            a, b = nodes[op[1]], nodes[op[2]]
            out = Node(a.data * b.data, (a, b))
            def _back(a=a, b=b, out=out):
                a.grad += b.data * out.grad
                b.grad += a.data * out.grad
            out._backward = _back
            nodes.append(out)
        elif op[0] == "neg":
            a = nodes[op[1]]
            out = Node(-a.data, (a,))
            def _back(a=a, out=out):
                a.grad += -out.grad
            out._backward = _back
            nodes.append(out)

    output_node = nodes[-1]
    output_node.grad = 1.0

    # Nodes are appended strictly in creation order (each op only references
    # existing nodes), so iterating the node list in reverse is already a
    # valid reverse-topological order — no separate DFS/topo-sort needed.
    for node in reversed(nodes):
        node._backward()

    gradients = [round(nodes[i].grad, 4) for i in range(n_inputs)]
    return {"output": round(output_node.data, 4), "gradients": gradients}