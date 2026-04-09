import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import Generator
from rl4co.envs.scheduling.fjsp.env import INIT_FINISH
from rl4co.envs.scheduling.fjsp.utils import get_job_ops_mapping, calc_lower_bound

class OperationSelectionEnv(): #(RL4COEnvBase):
    """
    My take on the JSSP env for ACO, vectorized, event driven with Delay!
    """

    name = "jssp_ops"

# You are ONLY testing:

    def __init__(self, generator, device=None, stepwise_reward=False): # remove num ops num machines from here and take from generator when testing  and add genereator
        #super().__init__(check_solution=False)
        self.generator = generator
        self.num_jobs = generator.num_jobs
        self.num_machines = generator.num_mas
        self.num_ops = generator.n_ops_max
        self.device = device if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.WAIT = self.num_ops
    def _reset(self, bs):
        td = self.generator._generate(batch_size=(bs,))
        td = {k: v.to(self.device) for k, v in td.items()}
        td["time"] = torch.zeros(bs, device=self.device)
        td["machine_available"] = torch.zeros(bs, self.num_machines, device=self.device) # time at which machine is available 
        td["op_scheduled"] = torch.zeros(bs, self.num_ops, dtype= torch.bool, device=self.device)
        td["job_next_step"] = torch.zeros(bs, self.num_jobs, dtype= torch.long, device=self.device)
        #testing random assignment
        td["op_machine"] = td["machine_id"]
        td['proc_time'] = td["proc_times"].max(1).values #(bs, num_ops)
        ops = torch.arange(self.num_ops, device=self.device)

        start = td["start_op_per_job"]  # (bs, num_jobs)
        end = td["end_op_per_job"]

        op_to_job = (
            (ops.unsqueeze(0).unsqueeze(-1) >= start.unsqueeze(1)) &
            (ops.unsqueeze(0).unsqueeze(-1) <= end.unsqueeze(1))
        ).float().argmax(-1)

        td["op_to_job"] = op_to_job
        
        td["action_mask"] = self._get_action_mask(td)
        return td 

    def _get_action_mask(self, td):
        
        bs = td["time"].shape[0]
        allowed = torch.zeros(bs, self.num_ops, dtype=torch.bool, device=self.device)
        batch_idx = torch.arange(bs, device=self.device)

        for j in range(self.num_jobs):

            step = td["job_next_step"][:,j] #(bs, job)
            start = td["start_op_per_job"][:,j]
            end = td["end_op_per_job"][:, j]

            op = step + start
            valid = op <= end

        allowed[batch_idx[valid], op[valid]] = True

        # remove already scheduled  
        feasible = allowed & (~td["op_scheduled"])
        wait = torch.ones(bs, 1, dtype= torch.bool, device=self.device)
        
        return torch.cat([feasible, wait], dim=1)
    
    def _step(self, td, action): # batch step is a better name 
        
        is_wait = action == self.WAIT # action is the also a tensor 
        is_act = ~is_wait

        if is_wait.any():
            td_wait = {k: v[is_wait] for k, v in td.items()} # split out the samples that require waiting
            td_wait = self._advance_time(td_wait) # advance the split samples
            for k in td:
                td[k][is_wait] = td_wait[k] # replace original samples with the split ones
                
        if is_act.any():
            #make the ants do a step advance time choose the machine, mark operation as scheduled 
            idx = is_act.nonzero().squeeze(-1) # index of non wait actions
            ops = action[idx] # list of operations that were picked for each non wait sample
            machines = td["op_machine"][idx, ops] #crazy indexing magic pairwise indexing of operations to machine lookup matrix (vectorized) 
            proc = td["proc_time"][idx, ops]
            start = torch.maximum(
                td["time"][idx],
                td["machine_available"][idx, machines]
            )

            # start_i = max(current_time_i, machine_free_time_i)
            finish = start + proc
            
            td["machine_available"][idx, machines] = finish
            td["op_scheduled"][idx, ops] = True

            #advance job pointer
            job = td["op_to_job"][idx, ops]
            td["job_next_step"][idx, job] += 1
            

        mask = self._get_action_mask(td)[:, :-1] # this is to remove the wait column _get_action mask returns [bs,2] where 2 is feasable and wait 
        not_feasible = ~mask.any(1) # mask.shape = (batch_size, num_ops)

        if not_feasible.any():
            td_not_feasible = {k: v [not_feasible] for k, v in td.items()} # split out non feasable samples 

            # advance time in no feasable action samples
            td_not_feasible = self._advance_time(td_not_feasible)

            # add no feasable samples into original td
            for k in td:
                td[k][not_feasible] = td_not_feasible[k] 

        td["action_mask"] = self._get_action_mask(td)
        return td
    def _advance_time(self,td):
        
        next_time = td['machine_available'].min(dim =1).values
        td['time'] = next_time

        return td
    def _get_reward(self,td):
        pass


# env = OperationSelectionEnv(num_ops=3, num_machines= 2)
# td = env._reset(bs=2)
# print(td)
# action = torch.tensor([0, 3]) 
# # sample 0 → op 0
# # sample 1 → WAIT (assuming WAIT = 3)
# td = env._step(td, action)
# action = torch.tensor([3, 1])
# td = env._step(td, action)
# print("time:", td["time"])
# print("machine_available:", td["machine_available"])
# print("op_scheduled:", td["op_scheduled"])