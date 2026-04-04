import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.scheduling.fjsp.env import INIT_FINISH
from rl4co.envs.scheduling.fjsp.utils import get_job_ops_mapping, calc_lower_bound

class OperationSelectionEnv(RL4COEnvBase):
    """
    My take on the JSSP env for ACO, vectorized, event driven with Delay!
    """

    name = "jssp_ops"

# You are ONLY testing:

# ✅ action handling (including WAIT)
# ✅ batching works
# ✅ time moves forward correctly
    def __init__(self, generator, num_ops, num_machines, stepwise_reward=False): # remove num ops num machines from here and take from generator when testing 
        super().__init__(check_solution=False)
        self.num_ops = num_ops#generator.num_ops
        self.num_machines = num_machines
        self.WAIT = num_ops 

    def _reset(self, bs):
        td = {}
        td["time"] = torch.zeros(bs)
        td["machine_available"] = torch.zeros(bs, self.num_machines)
        td["op_scheduled"] = torch.zeros(bs, self.num_ops, dtype= torch.bool)
        
        #testing random assignment
        td["op_machine"] = torch.randint(0,self.num_machines, (bs, self.num_ops))
        td['proc_time'] = torch.randint(1,5, (bs,self.num_ops))
        
        td["action_mask"] = self._get_action_mask(td)
        return td 

    def _get_action_mask(self, td):
        not_scheduled = ~td["op_scheduled"] # feasable actions are non-scheduled 


        
        wait = torch.ones(td["time"].shape[0], 1 , dtype= torch.bool)

        return torch.cat([not_scheduled, wait], dim=1)
    
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
            td["machine_available"][idx, machines] = finish

            # start_i = max(current_time_i, machine_free_time_i)
            finish = start + proc
            td["op_scheduled"][idx, ops] = True
            
            pass

    
    def _advance_time(self,td):
        
        next_time = td['machine_available'].min(dim =1).values
        td['time'] = next_time

        return td
    