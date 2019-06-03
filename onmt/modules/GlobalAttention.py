import torch
import torch.nn as nn

from onmt.Utils import aeq, sequence_mask


class GlobalAttention(nn.Module):
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
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.entity_dim = entity_dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
            self.linear_in_entity = nn.Linear(dim, entity_dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s, entity_attn=False):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(self.dim, tgt_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                if entity_attn:
                    h_t_ = self.linear_in_entity(h_t_)
                else:
                    h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input, memory_bank, entity_representation, memory_lengths=None, coverage=None, count_entities=None,
                total_entities_list=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          entity_representation (`FloatTensor`): source vectors `[batch x num_entities x eu_k_dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
          count_entities (`LongTensor`): entity lengths `[batch]`
          total_entities_list (`FloatTensor`): source vectors `[batch x num_entities x src_len]`

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

        batch, sourceL, dim = memory_bank.size()
        batch__, sourceL__, dim__ = entity_representation.size()
        batch_, targetL, dim_ = input.size()
        batch___, num_entities, src_len = total_entities_list.size()
        aeq(batch, batch_)
        aeq(batch, batch__)
        aeq(self.dim, dim_)
        aeq(self.entity_dim, dim__)
        aeq(self.dim, dim)
        aeq(sourceL, src_len)
        aeq(num_entities, sourceL__)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = self.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        entity_align = self.score(input, entity_representation, entity_attn=True)

        if count_entities is not None:
            count_entities_mask = sequence_mask(count_entities.data)
            count_entities_mask = count_entities_mask.unsqueeze(1)  # Make it broadcastable.
            entity_align.data.masked_fill_(1 - count_entities_mask, -float('inf'))
        entity_align_vectors = self.sm(entity_align.view(batch * targetL, sourceL__))
        entity_align_vectors = entity_align_vectors.unsqueeze(2).expand(-1, -1, sourceL)

        align = self.score(input, memory_bank)
        align = align.unsqueeze(2).expand(-1, -1, sourceL__, -1)
        total_entities_list = total_entities_list.unsqueeze(1).expand(-1, targetL, -1, -1)
        align = align*total_entities_list  # apply mask of records belonging to entities
        mask = total_entities_list.eq(0)
        align.data.masked_fill_(mask.data, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL*sourceL__, sourceL))

        count_entities_mask = count_entities_mask.unsqueeze(3)#.expand(-1, -1, -1, sourceL)
        align_vectors = align_vectors.view(batch, targetL, sourceL__, sourceL)
        align_vectors.data.masked_fill_(1-count_entities_mask, 0)
        align_vectors = align_vectors.view(batch*targetL, sourceL__, sourceL)

        align_vectors = (entity_align_vectors*align_vectors).sum(1)
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        return attn_h, align_vectors
