from graphviz import Digraph
from value import Value

def trace_computation_graph(root: Value):
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()
    def build(value: Value):
        if value in nodes:
            return
        nodes.add(value)
        for child in value._prev:
            edges.add((child, value))
            build(child)
    build(root)
    return nodes, edges

def draw_computation_graph(root: Value):
    dot = Digraph(
        format='svg',
        graph_attr={ 'rankdir': 'LR' }
    )
    nodes, edges = trace_computation_graph(root)
    for node in nodes:
        node_id = str(id(node))
        dot.node(
            name=node_id,
            label="{ data %.4f | grad %.4f }" % (node.data, node.grad),
            shape='record'
        )
        if node._op:
            op_id = node_id + node._op
            dot.node(name=op_id, label=node._op)
            dot.edge(op_id, node_id)
    for node_a, node_b in edges:
        node_a_id = str(id(node_a))
        node_b_op_id = str(id(node_b)) + node_b._op
        dot.edge(node_a_id, node_b_op_id)

    return dot