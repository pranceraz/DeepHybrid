import torch
import pytest

from my_env import OperationSelectionEnv
from generator import MyJSSPGenerator


# =========================================================
# FIXTURE
# =========================================================

@pytest.fixture(params=[(3, 2),(6,6)])
def env(request):
    j, m = request.param
    return OperationSelectionEnv(MyJSSPGenerator(j, m, max_processing_time=10))


def make_td(env, bs):
    td = env.generator(batch_size=[bs])
    return env.reset(td)


# =========================================================
# PHASE 1 TESTS
# =========================================================

def test_step_runs(env):
    td = make_td(env, 2)

    mask = td["action_mask"]
    action = torch.where(mask[0])[0][0].repeat(2)

    td["action"] = action
    td = env.step(td)
    td = td["next"]

    assert "time" in td


def test_wait_does_not_go_backwards(env):
    td = make_td(env, 2)

    action = torch.full((2,), env.WAIT, device=td.device)

    td["action"] = action
    td2 = env.step(td)
    td2 = td2["next"]

    assert (td2["time"] >= td["time"]).all()


def test_op_marked_scheduled(env):
    td = make_td(env, 1)

    op = torch.where(td["action_mask"][0][:-1])[0][0]

    td["action"] = op.unsqueeze(0)
    td = env.step(td)
    td = td["next"]

    assert td["op_scheduled"][0, op]


# =========================================================
# PHASE 2 TESTS (precedence)
# =========================================================

def test_precedence_only_first_ops(env):
    td = make_td(env, 1)

    mask = td["action_mask"][0]
    allowed_ops = torch.where(mask[:-1])[0]

    start_ops = td["start_op_per_job"][0]

    for op in allowed_ops:
        assert op in start_ops


def test_job_progression(env):
    td = make_td(env, 1)

    before = td["job_next_step"].clone()

    op = torch.where(td["action_mask"][0][:-1])[0][0]

    td["action"] = op.unsqueeze(0)
    td = env.step(td)
    td = td["next"]

    after = td["job_next_step"]

    assert (after >= before).all()


def test_job_step_never_exceeds_length(env):
    td = make_td(env, 4)

    for _ in range(10):
        mask = td["action_mask"]

        actions = []
        for i in range(mask.shape[0]):
            valid = torch.where(mask[i])[0]
            a = valid[torch.randint(len(valid), (1,))]
            actions.append(a.item())

        action = torch.tensor(actions, device=td.device)

        td["action"] = action
        td = env.step(td)
        td = td["next"]

        job_len = td["end_op_per_job"] - td["start_op_per_job"] + 1

        assert (td["job_next_step"] <= job_len).all()


# =========================================================
# PHASE 3 TESTS (machine constraints)
# =========================================================

def test_machine_blocks_same_machine(env):
    td = make_td(env, 1)

    machines = td["op_machine"][0]

    unique, counts = torch.unique(machines, return_counts=True)
    multi_machine = unique[counts > 1]

    if len(multi_machine) == 0:
        pytest.skip()

    machine = multi_machine[0]
    ops = torch.where(machines == machine)[0]
    op = ops[0]

    prev_time = td["time"].clone()

    td["action"] = op.unsqueeze(0)
    td = env.step(td)
    td = td["next"]

    mask = td["action_mask"][0]

    machine_time = td["machine_available"][0, machine]
    current_time = td["time"][0]

    if machine_time > current_time:
        for o in ops:
            if not td["op_scheduled"][0, o]:
                assert not mask[o]
    else:
        assert current_time > prev_time


def test_wait_required_when_blocked(env):
    td = make_td(env, 1)

    for _ in range(100):
        mask = td["action_mask"][0]
        valid_ops = torch.where(mask[:-1])[0]

        if len(valid_ops) == 0:
            break

        td["action"] = valid_ops[0].unsqueeze(0)
        td = env.step(td)
        td = td["next"]

    mask = td["action_mask"][0]

    if not mask[:-1].any():
        assert mask[-1]
    else:
        assert td["time"].item() > 0


def test_wait_advances_time(env):
    td = make_td(env, 1)

    op = torch.where(td["action_mask"][0][:-1])[0][0]

    td["action"] = op.unsqueeze(0)
    td = env.step(td)
    td = td["next"]

    prev_time = td["time"].clone()

    action = torch.tensor([env.WAIT], device=td.device)

    td["action"] = action
    td = env.step(td)
    td = td["next"]

    assert (td["time"] > prev_time).all()

def test_advance_time_no_inf_under_stress(env):
    td = env.reset(batch_size=[8])

    for _ in range(20):
        # random machine times around current time
        td["machine_available"] = torch.rand_like(td["machine_available"]) * 5
        td["time"] = torch.rand_like(td["time"]) * 5

        td = env._advance_time(td)

        assert torch.isfinite(td["time"]).all()
        
# =========================================================
# FULL ROLLOUT TEST
# =========================================================

def test_solution_validity(env):
    td = make_td(env, 1)

    for _ in range(env.num_ops * 2):
        mask = td["action_mask"][0]
        valid = torch.where(mask)[0]

        td["action"] = valid[0].unsqueeze(0)
        td = env.step(td)
        td = td["next"]

        if td["op_scheduled"].all():
            break

    assert td["op_scheduled"].all()
    assert td["op_scheduled"].sum() == env.num_ops

    job_len = td["end_op_per_job"] - td["start_op_per_job"] + 1

    assert (td["job_next_step"] == job_len).all()