import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import Generator
from rl4co.envs.scheduling.fjsp.env import INIT_FINISH
from rl4co.envs.scheduling.fjsp.utils import get_job_ops_mapping, calc_lower_bound

class OperationSelectionEnv(RL4COEnvBase):
    """
    My take on the JSSP env for ACO, vectorized, event driven with Delay!
    """

    name = "jssp_ops"

# You are ONLY testing:

    def __init__(self, generator, stepwise_reward=False):
        super().__init__(check_solution=False)
        self.generator = generator
        self.num_jobs = generator.num_jobs
        self.num_machines = generator.num_mas
        self.num_ops = generator.n_ops_max

        self.WAIT = self.num_ops

    def _reset(self, td: TensorDict = None, batch_size=None):

        if td is None:
            td = self.generator(batch_size=batch_size)

        td = td.clone()
        device = td.device
        bs = td.batch_size

        td["time"] = torch.zeros((*bs,), device=device)
        td["machine_available"] = torch.zeros((*bs, self.num_machines), device=device) # time at which machine is available 
        td["job_available"] = torch.zeros((*bs, self.num_jobs), device=device) # time at which each job can start its next op
        td["op_scheduled"] = torch.zeros((*bs, self.num_ops), dtype= torch.bool, device=device)
        td["job_next_step"] = torch.zeros((*bs, self.num_jobs), dtype= torch.long, device=device)
        #testing random assignment
        td["op_machine"] = td["machine_id"]
        td['proc_time'] = td["proc_times"].max(1).values #((*bs,), num_ops)
        ops = torch.arange(self.num_ops, device=device)

        start = td["start_op_per_job"]  # ((*bs,), num_jobs)
        end = td["end_op_per_job"]

        op_to_job = (
            (ops.unsqueeze(0).unsqueeze(-1) >= start.unsqueeze(1)) &
            (ops.unsqueeze(0).unsqueeze(-1) <= end.unsqueeze(1))
        ).float().argmax(-1)

        td["op_to_job"] = op_to_job
        
        td["action_mask"] = self._get_action_mask(td)
        td["current_node"] = torch.zeros(
            (*bs, 1),
            dtype=torch.long,
            device=device
        )
        return td 

    def _get_action_mask(self, td: TensorDict):
        device = td.device
        bs = td.batch_size[0]
        allowed = torch.zeros(bs, self.num_ops, dtype=torch.bool, device=device)
        batch_idx = torch.arange(bs, device=device)

        for j in range(self.num_jobs):

            step = td["job_next_step"][:,j] #(bs, job)
            start = td["start_op_per_job"][:,j]
            end = td["end_op_per_job"][:, j]

            op = step + start
            valid = op <= end
        
            allowed[batch_idx[valid], op[valid]] = True # bullshit indexing 

        # remove already scheduled  
        feasible = allowed & (~td["op_scheduled"])

        machines = td["op_machine"]#(bs, num_ops)
        #machine available is (bs, machines)
        machine_ready = td["machine_available"].gather(1, machines) # when each machine will be ready 
        jobs = td["op_to_job"]
        job_ready = td["job_available"].gather(1, jobs) # when each job is ready for its next op

        current_time = td["time"].unsqueeze(1) #-> (bs,1) # BROADCASTING

        machine_free = machine_ready <= current_time
        job_free = job_ready <= current_time

        feasible = feasible & machine_free & job_free

        # wait = torch.ones(bs, 1, dtype= torch.bool, device=device)
        
        # return torch.cat([feasible, wait], dim=1)
        return feasible

    
    def _step(self, td: TensorDict): # batch step is a better name 
            td = td.clone()
            action = td["action"]
            is_act = torch.ones_like(action, dtype=torch.bool)  # all actions are valid ops now

            # action = td["action"]
            # is_wait = action == self.WAIT # action is the also a tensor 
            # is_act = ~is_wait

            # if is_wait.any():
            #     # td_wait = {k: v[is_wait] for k, v in td.items()} 
            #     td_wait = td[is_wait]# split out the samples that require waiting
            #     td_wait = self._advance_time(td_wait) # advance the split samples
            #     # for k in td:
            #     #     td[k][is_wait] = td_wait[k] 
            #     td[is_wait] = td_wait# replace original samples with the split ones
                    
            if is_act.any():
                #make the ants do a step advance time choose the machine, mark operation as scheduled 
                idx = torch.arange(td.batch_size[0], device=td.device) # index of all samples
                ops = action # list of operations that were picked for each sample
                machines = td["op_machine"][idx, ops] #crazy indexing magic pairwise indexing of operations to machine lookup matrix (vectorized) 
                job = td["op_to_job"][idx, ops]
                proc = td["proc_time"][idx, ops]
                start = torch.maximum(
                    torch.maximum(
                        td["time"],
                        td["machine_available"][idx, machines]
                    ),
                    td["job_available"][idx, job]
                )

                # start_i = max(current_time_i, machine_free_time_i)
                finish = start + proc
                
                td["machine_available"][idx, machines] = finish
                td["job_available"][idx, job] = finish
                td["op_scheduled"][idx, ops] = True

                #advance job pointer
                td["current_node"][idx] = ops.unsqueeze(-1)
                td["job_next_step"][idx, job] += 1
                

            mask = self._get_action_mask(td)  # no wait column anymore
            # not_feasible = ~mask.any(1) # mask.shape = (batch_size, num_ops)

            # if not_feasible.any(): # makes time jump within the step if no feasable actions after doing one 
            #     # td_not_feasible = {k: v [not_feasible] for k, v in td.items()}
            #     td_not_feasible = td[not_feasible] # split out non feasable samples 

            #     # advance time in no feasable action samples
            #     td_not_feasible = self._advance_time(td_not_feasible)

            #     # add no feasable samples into original td
            #     # for k in td:
            #     #     td[k][not_feasible] = td_not_feasible[k] 
            #     td[not_feasible] = td_not_feasible
            #     mask = self._get_action_mask(td)


            # Keep advancing unfinished rows until at least one action exists.
            done_flat = self._get_done(td).squeeze(-1)
            need_advance = (~mask.any(1)) & (~done_flat)
            while need_advance.any():
                td_nf = td[need_advance]
                td_nf = self._advance_time(td_nf)
                td[need_advance] = td_nf

                mask = self._get_action_mask(td)
                done_flat = self._get_done(td).squeeze(-1)
                need_advance = (~mask.any(1)) & (~done_flat)

            done = self._get_done(td)

            reward = torch.zeros((*td.batch_size, 1), device=td.device)

            td.update({
                "action_mask": mask,
                "done": done,
                "reward": reward,
            })

            return td
            # return TensorDict({"next": td}, batch_size=td.batch_size)

    
    def _advance_time(self, td: TensorDict):

        current_time = td["time"].unsqueeze(1)  # (bs, 1)

        machine_times = td["machine_available"]  # (bs, num_machines)

        # Mask out machines that are already available NOW or in the past
        future_times = machine_times.clone()
        inf = torch.tensor(float("inf"), device=td.device)
        future_times[future_times <= current_time] = inf

        # Get next event (earliest future machine completion)
        next_time = future_times.min(dim=1).values

        # prevent inf (no future events case)
        next_time = torch.where(
            torch.isinf(next_time),
            td["time"],   # stay at current time instead of inf
            next_time
        )

        td["time"] = next_time

        return td
    
    def _get_reward(self, td, actions=None):
        done = td["op_scheduled"].all(dim=1)

        makespan = td["machine_available"].max(dim=1).values
        # Keep reward 1D [batch] to match RL4CO/DeepACO loss expectations.
        reward = -makespan

        reward[~done] = 0.0  # safety for partial batches
        return reward

    def _get_done(self, td):
        return td["op_scheduled"].all(dim=1, keepdim=True)
    
