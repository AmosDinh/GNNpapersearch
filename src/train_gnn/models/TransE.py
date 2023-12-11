from torch_geometric.nn.kge import TransE

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.kge import KGEModel

# adapted and taken from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/kge/transe.py

class TransE(KGEModel):
    r"""The TransE model from the `"Translating Embeddings for Modeling
    Multi-Relational Data" <https://proceedings.neurips.cc/paper/2013/file/
    1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>`_ paper.

    :class:`TransE` models relations as a translation from head to tail
    entities such that

    .. math::
        \mathbf{e}_h + \mathbf{e}_r \approx \mathbf{e}_t,

    resulting in the scoring function:

    .. math::
        d(h, r, t) = - {\| \mathbf{e}_h + \mathbf{e}_r - \mathbf{e}_t \|}_p

    .. note::

        For an example of using the :class:`TransE` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (int, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        p_norm (int, optional): The order embedding and distance normalization.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        self.p_norm = p_norm
        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                    out=self.rel_emb.weight.data)

    def forward(
        self,
        head_embeddings: Tensor,
        rel_type,
        tail_embeddings: Tensor,
    ) -> Tensor:
        #head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)  # Amos: only learn the relation embeddings, others are learned with GNN
        #tail = self.node_emb(tail_index)

        head = F.normalize(head_embeddings, p=self.p_norm, dim=-1)
        tail = F.normalize(tail_embeddings, p=self.p_norm, dim=-1)

        # Calculate *negative* TransE norm:
        return -((head + rel) - tail).norm(p=self.p_norm, dim=-1)

    
    def get_embedding(self,
                      embedding,
                      rel_type,
                        have_head_or_tail
                      ):
        rel = self.rel_emb(rel_type)
        embedding = F.normalize(embedding, p=self.p_norm, dim=-1)
        if have_head_or_tail == 'head':
            return embedding + rel
        else:
            return embedding - rel
    
    
    def loss(
        self,
        head_embeddings: Tensor,
        rel_type: Tensor,
        tail_embeddings: Tensor,
        labels: Tensor, # labels 0 or 1
    ) -> Tensor:
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        pos_score = self(head_embeddings[pos_mask], rel_type, tail_embeddings[pos_mask])
        neg_score = self(head_embeddings[neg_mask], rel_type, tail_embeddings[neg_mask])

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score), # 1 for similarity, -1 for dissimilarity
            margin=self.margin,
        )