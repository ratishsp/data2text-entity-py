import torch
import torch.nn as nn

from onmt.Utils import aeq, sequence_mask


class EntityMemoryUpdation(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.
    .. mermaid::
       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G
    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].
    However they
    differ on how they compute the attention score.
    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """
    def __init__(self, dim, entity_dim, coverage=False, attn_type="dot"):
        super(EntityMemoryUpdation, self).__init__()

        self.dim = dim
        self.entity_dim = entity_dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")

        self.linear_context = nn.Linear(entity_dim, entity_dim, bias=True)
        self.linear_query = nn.Linear(dim, entity_dim, bias=True)
        self.linear_gamma = nn.Linear(dim, entity_dim, bias=True)
        self.linear_h = nn.Linear(dim, entity_dim, bias=False)

        self.sm = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, entity_representation, memory_lengths=None, coverage=None):
        """
        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          entity_representation (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = entity_representation.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(self.entity_dim, dim)
        aeq(self.dim, dim_)

        # compute attention scores, as in Luong et al.
        #align = self.score(input, entity_representation)
        gamma = self.sigmoid(self.linear_gamma(input))
        wq = self.linear_query(input.view(-1, dim_))
        wq = wq.view(batch, targetL, 1, dim)
        wq = wq.expand(batch, targetL, sourceL, dim)

        uh = self.linear_context(entity_representation.contiguous().view(-1, dim))
        uh = uh.view(batch, 1, sourceL, dim)
        uh = uh.expand(batch, targetL, sourceL, dim)
        delta = self.sigmoid(wq + uh).view(batch, sourceL, dim)
        delta = gamma * delta
        entity_representation = (1-delta)*entity_representation + delta* self.linear_h(input).expand(-1, sourceL, -1)

        entity_representation = entity_representation.transpose(0, 1).contiguous()

        return entity_representation
