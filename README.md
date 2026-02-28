# DeepHybrid
Building hybrid metaheuristics for JSSPs
Building on top of DeepACO [https://arxiv.org/abs/2309.14032]

We utilize the RL4CO implementation as suggested by the DeepACO repo.

To adapt the the DeepACO paper for JSSP, we swapped the embeddings in the NARGNN Encoder, Adapted the JSSP Generator to provide some extra information in the tensordict, built a new env for JSSP modeled as a chain of operations as a schedule instead of the rl4co implementation of a job schedule.

Rl4CO does most of the heavy lifting!

We retrofit the encoder and onto the policy.  
