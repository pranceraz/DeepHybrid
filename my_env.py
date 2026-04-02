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
        not_scheduled = ~td["op_scheduled"]
        wait = torch.ones(td["time"].shape[0], 1 , dtype= torch.bool)

        return torch.cat([not_scheduled, wait], dim=1)
    
    def _step(self, td):
        return super().step(td)
    
    def _advance_time(self,td):
        pass
    