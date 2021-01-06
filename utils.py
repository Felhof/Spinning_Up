import torch


def mlp(sizes, activations):
    layers = []
    connections = [
        (in_dim, out_dim) for in_dim, out_dim in zip(sizes[:-1], sizes[1:])
    ]
    for connection, activation in zip(connections, activations):
        layers.append(torch.nn.Linear(connection[0], connection[1]))
        layers.append(activation())

    return torch.nn.Sequential(*layers)
