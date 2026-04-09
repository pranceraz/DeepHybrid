# import rl4co
# import torch
# import job_shop_lib
# from tensordict import TensorDict
# from job_shop_lib.graphs import build_disjunctive_graph 
# from rl4co.envs.scheduling.jssp.generator import JSSPGenerator
# from generator import MyJSSPGenerator
# # from job_shop_lib.generation import (
# #     modular_instance_generator,
# #     get_default_machine_matrix_creator,
# #     get_default_duration_matrix_creator,
# # )


from rl4co.models.rl.common.base import RL4COLitModule
# generator = JSSPGenerator(num_jobs=3, num_machines=3)
# mygenerator = MyJSSPGenerator(num_jobs=3,num_machines=3)
# # td = generator._generate(batch_size=[1])
# # print('proc_times')
# # print(td['proc_times'])
# # print('start_op_per_job')
# # print(td['start_op_per_job'])
# # print('end_op_per_job')
# # print(td['end_op_per_job'])
# # proc_time = td['proc_times'].sum(1)
# # pos_in_job = torch.reshape(proc_time,(:,3,))#replace 3 with td['num_jobs'])
# # print(proc_time)
# # pos = torch.arange(1, 4 + 1)  # [1,2,3]
# # pos = pos.unsqueeze(0).unsqueeze(0)  # shape (1,1,3)
# # pos = pos.expand(1,3,4)
# # pos = pos.reshape(1, -1)
# # print('this is the test one ')
# # print(pos)


# my_td = mygenerator._generate(batch_size = [1])
# proc_times = my_td['proc_times'][0]
# print('proc times')
# # print(proc_times.shape[1])
# print(proc_times)


# num_ops = proc_times.shape[1]

# num_jobs = 9 // 3
# op_ids = torch.arange(num_ops)
# op_ids = op_ids.view(num_jobs, 3) # view here is better cause .arange is contiguous...probably
# src = op_ids[:,:-1]
# dst = op_ids[:,1:]
# job_index = torch.stack([src.reshape(-1),dst.reshape(-1)],dim=0)
# print(job_index)


# machine_mask = proc_times > 0  
# machine_edges = []
# print('machine edges')
# ops = torch.where(machine_mask[0])[0]
# # for m in range(num_machines): 
# #     ops = torch.where(machine_mask[m])[0]

# if len(ops) > 1:
# #         # create all pair combinations
#     pairs = torch.combinations(ops, r=2)
# print(ops)
# print(pairs)
# #         # make bidirectional
# rev_pairs = pairs[:, [1, 0]]
# print(rev_pairs)
# all_pairs = torch.cat([pairs, rev_pairs], dim=0)
# print(all_pairs)
# machine_edges.append(all_pairs) 
# # print(machine_edges)                
# machine_edge_index = torch.cat(machine_edges, dim=0).T
# print(machine_edge_index)


# edge_index = torch.cat([job_index,machine_edge_index],dim=1)
# print (edge_index)


#     #Create edge_attr (one-hot)
    
# num_job_edges = job_index.shape[1]
# num_machine_edges = machine_edge_index.shape[1]

# job_attr = torch.tensor([1., 0.], device=proc_times.device).repeat(num_job_edges, 1)
# machine_attr = torch.tensor([0., 1.], device=proc_times.device).repeat(num_machine_edges, 1)

# edge_attr = torch.cat([job_attr, machine_attr], dim=0)


# print('machine_id')
# print(my_td['machine_id'].dtype)
# # # print(my_td['pos_in_job'])
# # pos_in_job = my_td['pos_in_job']
# # pos_in_job = pos_in_job.squeeze(0)
# # pos_in_job = pos_in_job.reshape(-1,3)  
# # print(pos_in_job)
# # # # fixed size example: 10 jobs, 5 machines
# # machine_creator = get_default_machine_matrix_creator(
# #     size_selector=lambda rng: (10, 5),
# #     with_recirculation=False,
# # )

# # duration_creator = get_default_duration_matrix_creator(
# #     duration_range=(1, 99),
# # )

# # gen = modular_instance_generator(
# #     machine_matrix_creator=machine_creator,
# #     duration_matrix_creator=duration_creator,
# #     seed=42,
# # )


# # def create_td(instance: job_shop_lib.JobShopInstance):
# #     td = TensorDict()
# #     proc_times = []
# #     positions = []
# #     machine_loads= instance.machine_loads()
# #     machines = []
# #     for i,job in enumerate(instance.jobs):
# #         for operation in job:
# #             positions.append(operation.position_in_job)
# #             proc_times.append(operation.duration)
# #             machines.append(operation.machines)
            
# #     return proc_times, positions
        
# #     proc_times = torch.tensor(proc_times, dtype= torch.float32)
# #     positions = torch.tensor(positions, dtype= torch.float32)
# # def build_graph(instance = job_shop_lib.JobShopInstance):
# #     graph = build_disjunctive_graph(instance)
# #     # graph.
# #     return None
# # instance = next(gen)

# # G = build_disjunctive_graph(instance)
# # nx_graph = G.graph
# # print(type(nx_graph))
# # for node, attr in nx_graph.nodes(data=True):
# #     print(node, attr)
# #     break

# # for u, v, attr in nx_graph.edges(data=True):
# #     print(u, v, attr)
# #     break

# # # print(dir(G))
# # print("Number of nodes:", nx_graph.number_of_nodes())
# # print("Number of edges:", nx_graph.number_of_edges())
# import torch

# from generator import MyJSSPGenerator
# from env import OperationSelectionEnv   # your new env file

# # --- Setup 3x3 instance ---
# generator = MyJSSPGenerator(num_jobs=3, num_machines=3)
# env = OperationSelectionEnv(generator)

# # Generate instance
# td_instance = generator._generate(batch_size=[1])

# # Reset env (IMPORTANT)
# td = env.reset(td_instance)
# proc = td["proc_times"][0]  # (machines, ops)
# print(proc)

# print("Initial action mask shape:", td["action_mask"].shape)
# print("Initial feasible ops:", td["action_mask"][0].nonzero())

# step = 0
# while not td["done"].all():

#     feasible = td["action_mask"][0].nonzero().squeeze(-1)

#     # remove NO_OP unless it's the only option
#     real_ops = feasible[feasible != 0]

#     if len(real_ops) > 0:
#         action = real_ops[0]
#     else:
#         action = feasible[0]

#     td.set("action", action.unsqueeze(0))
#     td = env.step(td)["next"]

#     print("Scheduled:", action.item())


# print("DONE")
# print("Makespan:", -env.get_reward(td, None).item())
# ===============================
# LOAD FT06
# ===============================
# from job_shop_lib.benchmarking import load_benchmark_instance
# import torch
# import numpy as np
# from tensordict import TensorDict
# instance = load_benchmark_instance("ft06")

# num_jobs = instance.num_jobs
# num_machines = instance.num_machines

# print(f"Loaded FT06: {num_jobs} jobs, {num_machines} machines")

# # Extract machine order + processing times
# machine_orders = []
# proc_times_list = []

# for job in instance.jobs:
#     machines = []
#     times = []
#     for op in job:
#         machines.append(op.machine_id)
#         times.append(op.duration)
#     machine_orders.append(machines)
#     proc_times_list.append(times)

# machine_orders = np.array(machine_orders)
# proc_times_array = np.array(proc_times_list)

# # ===============================
# # CONVERT TO TENSORDICT
# # ===============================

# # Number of operations
# n_ops = num_jobs * num_machines

# # Build processing tensor: (machines, ops)
# proc_tensor = torch.zeros(num_machines, n_ops)

# ops_job_map = []
# start_op_per_job = []
# end_op_per_job = []

# op_counter = 0

# for j in range(num_jobs):
#     start_op_per_job.append(op_counter)
#     for k in range(num_machines):
#         m = machine_orders[j, k]
#         t = proc_times_array[j, k]
#         proc_tensor[m, op_counter] = t
#         ops_job_map.append(j)
#         op_counter += 1
#     end_op_per_job.append(op_counter - 1)

# pad_mask = torch.zeros(n_ops).bool()

# td = TensorDict({
#     "proc_times": proc_tensor.unsqueeze(0),  # batch dim
#     "start_op_per_job": torch.tensor(start_op_per_job).unsqueeze(0),
#     "end_op_per_job": torch.tensor(end_op_per_job).unsqueeze(0),
#     "pad_mask": pad_mask.unsqueeze(0),
# }, batch_size=[1])#.to(DEVICE)

# print(td['proc_times'])


import torch
from rl4co.envs.scheduling import JSSPEnv
from rl4co.envs.routing.atsp i
from my_env import OperationSelectionEnv
from generator import MyJSSPGenerator

gen = MyJSSPGenerator(num_jobs=2, num_machines=2)
env = OperationSelectionEnv(generator=gen)


def test_env(env, bs=2, steps=10):

    td = env._reset(bs)

    print("=== INITIAL STATE ===")
    print("job_next_step:\n", td["job_next_step"])
    print("action_mask:\n", td["action_mask"])
    print()

    for t in range(steps):

        print(f"=== STEP {t} ===")

        mask = td["action_mask"]

        # sample random VALID actions
        actions = []
        for i in range(bs):
            valid_actions = torch.where(mask[i])[0]
            a = valid_actions[torch.randint(len(valid_actions), (1,))]
            actions.append(a.item())

        action = torch.tensor(actions, device=env.device)

        print("action:", action)

        td = env._step(td, action)

        print("time:", td["time"])
        print("job_next_step:\n", td["job_next_step"])
        print("op_scheduled:\n", td["op_scheduled"])
        print("mask:\n", td["action_mask"])
        print()

        # ✅ CHECK 1: no illegal ops scheduled
        assert td["op_scheduled"].sum() <= env.num_ops * bs

        # ✅ CHECK 2: job_next_step never exceeds job length
        max_len = td["end_op_per_job"] - td["start_op_per_job"] + 1
        assert (td["job_next_step"] <= max_len).all()

    print("✅ TEST PASSED")

test_env(env)