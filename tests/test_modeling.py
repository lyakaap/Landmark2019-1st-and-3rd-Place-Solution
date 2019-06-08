import pytest
import torch
from src import metrics


@pytest.fixture(scope='function', autouse=True)
def logits():
    torch.manual_seed(7)
    return [torch.rand(2, 3, 64, 64).cuda() for _ in range(3)]


@pytest.fixture(scope='function', autouse=True)
def gt_label():
    torch.manual_seed(7)
    return torch.randint(low=0, high=5, size=(2, 64, 64)).cuda()


def test_accuracy():
    x = torch.rand(8, 10)
    y = torch.rand(8).long()
    acc = metrics.accuracy(x, y)

    assert type(acc) is float
    assert 0.0 <= acc <= 1.0
