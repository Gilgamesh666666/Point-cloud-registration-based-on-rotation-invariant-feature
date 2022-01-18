import torch

def element_distance(a, b):
    # torch:[*, c, n], torch:[*, c, m]
    assert len(a.shape) == len(b.shape)
    n, m = a.shape[-1], b.shape[-1]
    dist = (a.unsqueeze(len(a.shape)).expand(len(a.shape)*[-1]+[m]) - b.unsqueeze(len(a.shape)-1).expand((len(a.shape)-1)*[-1]+[n]+[-1])).norm(dim=-3)
    return dist #[*, n, m]