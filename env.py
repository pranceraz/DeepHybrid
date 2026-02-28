import torch
from einops import reduce
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.scheduling.fjsp.env import INIT_FINISH
from rl4co.envs.scheduling.fjsp.utils import get_job_ops_mapping, calc_lower_bound


class OperationSelectionEnv(RL4COEnvBase):
    """
    Clean JSSP environment with operation-level actions.
    Action space = operation index (0 ... n_ops-1)
    """

    name = "jssp_ops"

    def __init__(self, generator, stepwise_reward=False):
        super().__init__(check_solution=False)
        self.generator = generator
        self.num_jobs = generator.num_jobs
        self.num_mas = generator.num_mas
        self.n_ops_max = generator.n_ops_max
        self.stepwise_reward = stepwise_reward
        self._make_spec()

    # =====================================================
    # SPECS
    # =====================================================

    def _make_spec(self):
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.n_ops_max,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    # =====================================================
    # RESET
    # =====================================================

    def _reset(self, td: TensorDict = None, batch_size=None):
        td = td.clone()
        bs = td.batch_size

        # Decode graph structure (job precedence)
        td, _ = self._decode_graph_structure(td)

        td.update({
            "time": torch.zeros((*bs,)),
            "start_times": torch.zeros((*bs, self.n_ops_max)),
            "finish_times": torch.full((*bs, self.n_ops_max), INIT_FINISH),
            "busy_until": torch.zeros((*bs, self.num_mas)),
            "op_scheduled": torch.zeros((*bs, self.n_ops_max), dtype=torch.bool),
            "job_in_process": torch.zeros((*bs, self.num_jobs), dtype=torch.bool),
            "job_done": torch.zeros((*bs, self.num_jobs), dtype=torch.bool),
            "done": torch.zeros((*bs, 1), dtype=torch.bool),
        })

        td["ops_ma_adj"] = (td["proc_times"] > 0).float()
        td["num_eligible"] = td["ops_ma_adj"].sum(1)
        td["lbs"] = calc_lower_bound(td)

        td.set("action_mask", self.get_action_mask(td))
        return td

    # =====================================================
    # ACTION MASK (NO FJSP DEPENDENCIES)
    # =====================================================
    def get_action_mask(self, td):

        # predecessor adjacency (bs, ops, ops)
        pred_adj = td["ops_adj"][..., 0]

        # Get finish times expanded
        finish = td["finish_times"].unsqueeze(1)  # (bs,1,ops)

        # predecessor finish times masked
        pred_finish_times = pred_adj * finish  # (bs, ops, ops)

        # predecessor exists mask
        has_pred = pred_adj.sum(-1) > 0

        # check all predecessors finished
        preds_finished = (pred_finish_times <= td["time"].unsqueeze(1).unsqueeze(2)) | (pred_adj == 0)

        # reduce across predecessor dimension
        preds_finished = preds_finished.all(-1)

        ready = (~has_pred) | preds_finished

        not_scheduled = ~td["op_scheduled"]

        machine_of_op = td["ops_ma_adj"].argmax(1)

        machine_free = td["busy_until"].gather(
            1, machine_of_op
        ) <= td["time"].unsqueeze(1)

        feasible = ready & not_scheduled & machine_free

        return feasible

    # =====================================================
    # STEP
    # =====================================================

    def _step(self, td: TensorDict):
        td = td.clone()
        bs = td.size(0)
        batch_idx = torch.arange(bs)

        op = td["action"]

        machine = td["ops_ma_adj"][batch_idx, :, op].argmax(1)
        job = td["ops_job_map"][batch_idx, op]
        proc_time = td["proc_times"][batch_idx, machine, op]

        # Schedule operation
        td["start_times"][batch_idx, op] = td["time"]
        td["finish_times"][batch_idx, op] = td["time"] + proc_time
        td["busy_until"][batch_idx, machine] = td["time"] + proc_time
        td["op_scheduled"][batch_idx, op] = True
        td["job_in_process"][batch_idx, job] = True

        # Remove operation from machine availability
        td["proc_times"][batch_idx, :, op] = 0
        td["ops_ma_adj"] = (td["proc_times"] > 0).float()

        # Advance time if needed
        td = self._advance_time(td)

        td["done"] = td["op_scheduled"].all(1, keepdim=True)
        td.set("action_mask", self.get_action_mask(td))

        return td

    # =====================================================
    # TIME ADVANCE
    # =====================================================

    def _advance_time(self, td):

        while True:

            # ALWAYS recompute mask first
            td.set("action_mask", self.get_action_mask(td))

            feasible = td["action_mask"].any(1)
            done = td["op_scheduled"].all(1)

            need_advance = ~feasible & ~done

            if not need_advance.any():
                break

            # advance to next machine release time
            available_time = torch.where(
                td["busy_until"] > td["time"].unsqueeze(1),
                td["busy_until"],
                torch.inf,
            ).min(1).values

            td["time"] = torch.where(
                need_advance,
                available_time,
                td["time"],
            )

            # free finished jobs
            finished = td["finish_times"] <= td["time"].unsqueeze(1)
            td["job_in_process"][finished.any(1)] = False

        return td

    # =====================================================
    # REWARD
    # =====================================================

    def _get_reward(self, td, actions=None):
        assert td["op_scheduled"].all()
        return (
            -td["finish_times"]
            .masked_fill(td["pad_mask"], -torch.inf)
            .max(1)
            .values
        )

    # =====================================================
    # GRAPH STRUCTURE (BORROWED CLEANLY FROM FJSP)
    # =====================================================

    def _decode_graph_structure(self, td):
        start = td["start_op_per_job"]
        end = td["end_op_per_job"]
        pad_mask = td["pad_mask"]
        n_ops = pad_mask.size(-1)

        ops_job_map, ops_job_bin_map = get_job_ops_mapping(
            start, end, n_ops
        )

        ops_job_bin_map[pad_mask.unsqueeze(1).expand_as(ops_job_bin_map)] = 0

        ops_seq_order = torch.sum(
            ops_job_bin_map * (ops_job_bin_map.cumsum(2) - 1), dim=1
        )

        pred = torch.diag_embed(torch.ones(n_ops - 1), offset=-1)[None].expand(
            *td.batch_size, -1, -1
        )
        pred = pred * ops_seq_order.gt(0).unsqueeze(-1)

        succ = torch.diag_embed(torch.ones(n_ops - 1), offset=1)[None].expand(
            *td.batch_size, -1, -1
        )
        succ = succ * torch.cat(
            (ops_seq_order[:, 1:], ops_seq_order.new_full((*td.batch_size, 1), 0)),
            dim=1
        ).gt(0).unsqueeze(-1)

        ops_adj = torch.stack((pred, succ), dim=3)

        td.update({
            "ops_adj": ops_adj,
            "job_ops_adj": ops_job_bin_map,
            "ops_job_map": ops_job_map,
            "ops_sequence_order": ops_seq_order,
        })

        return td, n_ops
