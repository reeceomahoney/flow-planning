import torch

from flow_planning.policy import pick_plan


def test_pick_plan_reductions():
    # 1 world, 2 modes x 3 samples. Mode 0 has one lucky sample but is bad
    # overall; mode 1 is consistently good. Flat argmin takes the lucky outlier.
    s = torch.tensor([[0.0, 9.0, 9.0, 1.0, 2.0, 3.0]])
    assert pick_plan(s, ns=3, reduce="min").tolist() == [0]
    assert pick_plan(s, ns=3, reduce="median").tolist() == [3]  # mode 1, best of it
    assert pick_plan(s, ns=3, reduce="max").tolist() == [3]


def test_pick_plan_batches_worlds_independently():
    # same two modes, swapped between worlds; the pick must swap with them
    s = torch.tensor([[0.0, 9.0, 9.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 0.0, 9.0, 9.0]])
    assert pick_plan(s, ns=3, reduce="median").tolist() == [3, 0]


if __name__ == "__main__":  # no pytest in this env: `uv run python tests/...`
    test_pick_plan_reductions()
    test_pick_plan_batches_worlds_independently()
    print("ok")
