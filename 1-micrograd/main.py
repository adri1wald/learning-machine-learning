from nn import MLP
from value import Value

def main():
    # data
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    # nn
    mlp = MLP(3, [4, 4, 1])

    # training loop
    for i in range(10000):
        ypreds = [mlp(x)[0] for x in xs]
        loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypreds)), Value(0))
        if i % 1000 == 0:
            print('Current loss', loss.data)
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data += - 0.1 * p.grad

if __name__ == '__main__':
    main()
