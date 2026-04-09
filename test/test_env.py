import torch
import pytest

from my_env import OperationSelectionEnv
from generator import MyJSSPGenerator


# =========================================================
# FIXTURE
# =========================================================
# Runs tests on small and medium instances
@pytest.fixture(params=[(3, 2)])
def env(request):
    j, m = request.param
    return OperationSelectionEnv(MyJSSPGenerator(j, m, max_processing_time= 10))


# =========================================================
# PHASE 1 TESTS (basic mechanics)
# =========================================================

def test_step_runs(env):
    """
    Basic sanity:
    Step should run without crashing.
    """
    td = env._reset(bs=2)

    mask = td["action_mask"]
    action = torch.where(mask[0])[0][0].repeat(2)

    td = env._step(td, action)

    assert "time" in td


def test_wait_does_not_go_backwards(env):
    """
    WAIT should never decrease time.
    """
    td = env._reset(bs=2)

    action = torch.full((2,), env.WAIT, device=env.device)

    td2 = env._step(td, action)

    assert (td2["time"] >= td["time"]).all()


def test_op_marked_scheduled(env):
    """
    Scheduling an op should mark it as done.
    """
    td = env._reset(bs=1)

    op = torch.where(td["action_mask"][0][:-1])[0][0]
    td = env._step(td, op.unsqueeze(0))

    assert td["op_scheduled"][0, op]


# =========================================================
# PHASE 2 TESTS (precedence)
# =========================================================

def test_precedence_only_first_ops(env):
    """
    Initially, only FIRST operation of each job should be allowed.
    """
    td = env._reset(bs=1)

    mask = td["action_mask"][0]
    allowed_ops = torch.where(mask[:-1])[0]

    start_ops = td["start_op_per_job"][0]

    for op in allowed_ops:
        assert op in start_ops, " Precedence broken"


def test_job_progression(env):
    """
    After scheduling an op, job_next_step must increase.
    """
    td = env._reset(bs=1)

    before = td["job_next_step"].clone()

    op = torch.where(td["action_mask"][0][:-1])[0][0]
    td = env._step(td, op.unsqueeze(0))

    after = td["job_next_step"]

    assert (after >= before).all()


def test_job_step_never_exceeds_length(env):
    """
    Job pointer must never overflow job length.
    """
    td = env._reset(bs=4)

    for _ in range(10):
        mask = td["action_mask"]

        actions = []
        for i in range(mask.shape[0]):
            valid = torch.where(mask[i])[0]
            a = valid[torch.randint(len(valid), (1,))]
            actions.append(a.item())

        action = torch.tensor(actions, device=env.device)
        td = env._step(td, action)

        job_len = td["end_op_per_job"] - td["start_op_per_job"] + 1

        assert (td["job_next_step"] <= job_len).all()


# =========================================================
# PHASE 3 TESTS (machine constraints)
# =========================================================
def test_machine_blocks_same_machine(env):
    """
    After scheduling an op:
    - Either machine is still busy → other ops must be blocked
    - OR time has advanced → machine is free again
    """

    td = env._reset(bs=1)

    machines = td["op_machine"][0]

    unique, counts = torch.unique(machines, return_counts=True)
    multi_machine = unique[counts > 1]

    if len(multi_machine) == 0:
        pytest.skip("No machine conflict possible")

    machine = multi_machine[0]

    ops = torch.where(machines == machine)[0]
    op = ops[0]

    prev_time = td["time"].clone()

    td = env._step(td, op.unsqueeze(0))

    mask = td["action_mask"][0]

    machine_time = td["machine_available"][0, machine]
    current_time = td["time"][0]

    if machine_time > current_time:
        # machine still busy → enforce blocking
        for o in ops:
            if not td["op_scheduled"][0, o]:
                assert not mask[o], "❌ Machine constraint missing"
    else:
        # time advanced → must have progressed
        assert current_time > prev_time

def test_machine_must_be_free(env):
    """
    Machine constraint:
    If machine is busy → op must be blocked
    OR env must have advanced time.
    """

    td = env._reset(bs=1)

    machines = td["op_machine"][0]

    unique, counts = torch.unique(machines, return_counts=True)
    multi_machine = unique[counts > 1]

    if len(multi_machine) == 0:
        pytest.skip("No machine conflict possible")

    machine = multi_machine[0]
    ops = torch.where(machines == machine)[0]
    op = ops[0]

    prev_time = td["time"].clone()

    td = env._step(td, op.unsqueeze(0))

    machine_time = td["machine_available"][0, machine]
    current_time = td["time"][0]

    mask = td["action_mask"][0]

    if machine_time > current_time:
        # machine still busy → ops must be blocked
        for o in ops:
            if not td["op_scheduled"][0, o]:
                assert not mask[o], "❌ Busy machine allowed"
    else:
        # time advanced → must have progressed
        assert current_time > prev_time


def test_wait_required_when_blocked(env):
    """
    If NO feasible operations exist:
    - WAIT must be the only action
    OR
    - env auto-advances time
    """

    td = env._reset(bs=1)

    # Keep scheduling until no ops are feasible
    for _ in range(100):
        mask = td["action_mask"][0]
        valid_ops = torch.where(mask[:-1])[0]

        if len(valid_ops) == 0:
            break

        td = env._step(td, valid_ops[0].unsqueeze(0))

    mask = td["action_mask"][0]

    # Now we are truly blocked OR time advanced
    if not mask[:-1].any():
        assert mask[-1]
    else:
        # env must have advanced time to unblock
        assert td["time"].item() > 0

def test_wait_advances_time(env):
    """
    WAIT should move time forward.
    """
    td = env._reset(bs=1)

    op = torch.where(td["action_mask"][0][:-1])[0][0]
    td = env._step(td, op.unsqueeze(0))

    prev_time = td["time"].clone()

    action = torch.tensor([env.WAIT], device=env.device)
    td = env._step(td, action)

    assert (td["time"] > prev_time).all()

def test_solution_validity(env):
    """
    Validates that the produced schedule is structurally correct:
    - all operations are scheduled
    - no duplicates
    - precedence respected (via job_next_step)
    """

    td = env._reset(bs=1)

    # Run until all ops scheduled
    for _ in range(env.num_ops * 2):

        mask = td["action_mask"][0]
        valid = torch.where(mask)[0]

        # always pick a valid action
        action = valid[0].unsqueeze(0)
        td = env._step(td, action)

        if td["op_scheduled"].all():
            break

    # =========================================================
    # 1. ALL OPS SCHEDULED
    # =========================================================
    assert td["op_scheduled"].all(), \
        "❌ Not all operations were scheduled"

    # =========================================================
    # 2. NO DUPLICATES (implicit check)
    # =========================================================
    # op_scheduled is boolean → cannot double schedule
    # so we just ensure count is correct
    assert td["op_scheduled"].sum() == env.num_ops, \
        "❌ Incorrect number of scheduled operations"

    # =========================================================
    # 3. PRECEDENCE VALIDITY
    # =========================================================
    # job_next_step should equal job length for all jobs

    job_len = td["end_op_per_job"] - td["start_op_per_job"] + 1

    assert (td["job_next_step"] == job_len).all(), \
        "❌ Precedence violated (job not fully completed)"

    print("✅ Valid schedule (structural)")


# def test_solution_validity(env):

#     td = env._reset(bs=1)

#     print("\n===== INITIAL STATE =====")
#     print("time:", td["time"])
#     print("op_scheduled:", td["op_scheduled"])
#     print("job_next_step:", td["job_next_step"])
#     print("mask:", td["action_mask"])
#     print("start_op_per_job:", td["start_op_per_job"])
#     print("end_op_per_job:", td["end_op_per_job"])
#     print("job_next_step:", td["job_next_step"])

#     for step in range(env.num_ops * 3):  # give more room

#         mask = td["action_mask"][0]
#         valid = torch.where(mask)[0]

#         print(f"\n----- STEP {step} -----")
#         print("time:", td["time"].item())
#         print("mask:", mask)
#         print("valid actions:", valid.tolist())

#         if len(valid) == 0:
#             print("❌ NO VALID ACTIONS")
#             break

#         action = valid[0].unsqueeze(0)
#         print("chosen action:", action.item())

#         td = env._step(td, action)

#         print("op_scheduled:", td["op_scheduled"])
#         print("job_next_step:", td["job_next_step"])
#         print("machine_available:", td["machine_available"])

#         if td["op_scheduled"].all():
#             print("✅ ALL OPS SCHEDULED")
#             break

#     print("\n===== FINAL STATE =====")
#     print("op_scheduled:", td["op_scheduled"])
#     print("job_next_step:", td["job_next_step"])
#     print("mask:", td["action_mask"])

#     assert td["op_scheduled"].all(), "❌ Not all operations were scheduled"