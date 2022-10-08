from nn import MLP
from viz import draw_computation_graph

def main():
    # nn
    x = [2.0, 3.0, -1]
    n = MLP(3, [4, 4, 1])
    r = n(x)
    o = r[0]

    # back prop
    o.zero_grad()
    o.backward()

    # draw
    cg = draw_computation_graph(o)
    cg.render('./diagrams/mlp', format='png', cleanup=True)

if __name__ == '__main__':
    main()
