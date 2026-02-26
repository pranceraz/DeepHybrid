import torch
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict
from types import UnionType
try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import to_undirected
except ImportError:
    Batch = Data = None

class JSSPInitEmbedding(nn.Module):
    '''Initial Embedding for JSSP (static)
    Embed the following node features to the embedding space:
        - proc_time:  TODO:normalization 
        - position: position in job
        - machine: machine id
        - TODO: machine load
        TODO: earliest start time ,

    '''

    def __init__(
            self,
            embed_dim:int,
            num_machines:int,
            linear_bias: bool = True,
            num_feats:int = 3,
            machine_embed_dim:int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scalar_embed = nn.Linear(num_feats, embed_dim,linear_bias)
        self.machine_embed = nn.Embedding(num_machines, machine_embed_dim)

        self.final_proj = nn.Linear(embed_dim + machine_embed_dim, embed_dim)


    def features(self,td:TensorDict):
        #return a tensor of proc time, pos in job using the td(proc_times)
        proc_times = td['proc_times'].sum(1)#(bs,operations)
        position_in_job = td['position_in_job']#(bs,operations)  # WARNING: shape mismatch i think
        feats = [proc_times,position_in_job]
        return torch.stack(feats,dim=-1) #too have each feat matrix have one of each
    
    def get_pos_in_job(self,td:TensorDict)-> torch.Tensor :
        #TODO: Implement a fn that takes in  td['proc_times'] proc_times and returns position in the job(maybe OHE) for every operation  
        pass

    def forward(self, td:TensorDict):
        feats= self.features(td)
        scalar_emb = self.scalar_embed(feats)
        machine_emb = self.machine_embed(td['machine_id'])
        emb = torch.cat([scalar_emb, machine_emb], dim=-1)
        emb = self.final_proj(emb)
        return emb
    
class JsspEdgeEmbedding(nn.Module):
    '''Edge embeddings for static JSSP instance
        -Conjunctive Edge: 0
        - Disjunctive Edge: 1
    '''
    def __init__(self, embed_dim, self_loop=False, **kwargs):
        assert Batch is not None, (
            "torch_geometric not found. Please install torch_geometric using instructions from "
            "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html."
        )
    
    def forward(self, td:TensorDict, init_embeddings: Tensor):
        proc_times = td['proc_times'] 
        batch = self._proc_times_to_graph(proc_times, init_embeddings)
        return batch
    
    def _make_edge_attributes(self, proc_times:Tensor,num_machines:int):
        '''
        Docstring for _make_edge_attributes
        
        :param self: Description
        :param proc_times: Description
        :type proc_times: Tensor
        :param num_machines: Description
        :type num_machines: int
        '''
        num_ops = proc_times.shape[1]
        num_jobs = num_ops // num_machines
        op_ids = torch.arange(num_ops)
        op_ids = op_ids.view(num_jobs, num_machines) # view here is better cause .arange is contiguous...probably
        src = op_ids[:,:-1]
        dst = op_ids[:,1:]
        edge_index = torch.stack([src.reshape(-1),dst.reshape(-1)],dim=0)
        
        #shitty ai solution
        machine_mask = proc_times > 0  
        machine_edges = []

        for m in range(num_machines): 
            ops = torch.where(machine_mask[m])[0]

            if len(ops) > 1:
                # create all pair combinations
                pairs = torch.combinations(ops, r=2)

                # make bidirectional
                rev_pairs = pairs[:, [1, 0]]

                all_pairs = torch.cat([pairs, rev_pairs], dim=0)

                machine_edges.append(all_pairs) 
                
        machine_edge_index = torch.cat(machine_edges, dim=0).T
        #shitty ai solution end 
        #return edge_index,edge_attr
        pass
    
    def _proc_times_to_graph(self, batch_proc_times: Tensor, batch_pos_in_job: Tensor,init_embeddings: Tensor):
        """Convert batched cost_matrix to batched PyG graph, and calculate edge embeddings.

        Args:
            batch_cost_matrix: Tensor of shape [batch_size, n, n]
            init_embedding: init embeddings
        """
        graph_data = []
        num_machines = batch_proc_times.shape[1] # 
        for index, proc_times in enumerate(batch_proc_times): 
            # insert get edge index logic plus edge attribute 
            edge_index,edge_attr = self._make_edge_attributes(proc_times,num_machines=num_machines)
            graph = Data(
                x=init_embeddings[index], edge_index=edge_index, edge_attr=edge_attr
            )
            graph_data.append(graph)
        
        batch = Batch.from_data_list(graph_data)
        batch.edge_attr = self.edge_embed(batch.edge_attr)
        return batch



class TSPEdgeEmbedding(nn.Module):
    """Edge embedding module for the Traveling Salesman Problem (TSP) and related problems.
    This module converts the cost matrix or the distances between nodes into embeddings that can be
    used by the neural network. It supports sparsification to focus on a subset of relevant edges,
    which is particularly useful for large graphs.
    """

    node_dim = 1

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
        sparsify=True,
        k_sparse: Union[int, Callable[[int], int], None] = None,
    ):
        assert Batch is not None, (
            "torch_geometric not found. Please install torch_geometric using instructions from "
            "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html."
        )

        super(TSPEdgeEmbedding, self).__init__()

        if k_sparse is None:
            self._get_k_sparse = lambda n: max(n // 5, 10)
        elif isinstance(k_sparse, int):
            self._get_k_sparse = lambda n: k_sparse
        elif callable(k_sparse):
            self._get_k_sparse = k_sparse
        else:
            raise ValueError("k_sparse must be an int or a callable")

        self.sparsify = sparsify
        self.edge_embed = nn.Linear(self.node_dim, embed_dim, linear_bias)

    def forward(self, td, init_embeddings: Tensor):
        cost_matrix = get_distance_matrix(td["locs"])
        batch = self._cost_matrix_to_graph(cost_matrix, init_embeddings)
        return batch 


    def _cost_matrix_to_graph(self, batch_cost_matrix: Tensor, init_embeddings: Tensor):
        """Convert batched cost_matrix to batched PyG graph, and calculate edge embeddings.

        Args:
            batch_cost_matrix: Tensor of shape [batch_size, n, n]
            init_embedding: init embeddings
        """
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data = []
        for index, cost_matrix in enumerate(batch_cost_matrix):
            if self.sparsify:
                edge_index, edge_attr = sparsify_graph(
                    cost_matrix, k_sparse, self_loop=False
                )
            else:
                edge_index = get_full_graph_edge_index(
                    cost_matrix.shape[0], self_loop=False
                ).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            graph = Data(
                x=init_embeddings[index], edge_index=edge_index, edge_attr=edge_attr
            )
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)
        batch.edge_attr = self.edge_embed(batch.edge_attr)
        return batch
