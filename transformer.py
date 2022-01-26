import math
import torch
import torch.jit as jit
import torch._jit_internal as _jit_internal
import copy
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_, constant_, xavier_normal_


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(hi, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is par you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(hi, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, bias=False, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': _jit_internal.Optional[torch.Tensor],
        'bias_v': _jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim), requires_grad=True)
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim), requires_grad=True)
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim), requires_grad=True)
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim), requires_grad=True)
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim), requires_grad=True)
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim), requires_grad=True)
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim), requires_grad=True)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        super(MultiheadAttention, self).__setstate__(state)

        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if 'self._qkv_same_embed_dim' not in self.__dict__:
            self._qkv_same_embed_dim = True

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, _jit_internal.Optional[torch.Tensor], bool, _jit_internal.Optional[torch.Tensor]) -> torch.Tuple[torch.Tensor, _jit_internal.Optional[torch.Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def multi_head_attention_forward(query,                           # type: torch.Tensor
                                 key,                             # type: torch.Tensor
                                 value,                           # type: torch.Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: torch.Tensor
                                 in_proj_bias,                    # type: torch.Tensor
                                 bias_k,                          # type: _jit_internal.Optional[torch.Tensor]
                                 bias_v,                          # type: _jit_internal.Optional[torch.Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: torch.Tensor
                                 out_proj_bias,                   # type: torch.Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: _jit_internal.Optional[torch.Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: _jit_internal.Optional[torch.Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: _jit_internal.Optional[torch.Tensor]
                                 k_proj_weight=None,              # type: _jit_internal.Optional[torch.Tensor]
                                 v_proj_weight=None,              # type: _jit_internal.Optional[torch.Tensor]
                                 static_k=None,                   # type: _jit_internal.Optional[torch.Tensor]
                                 static_v=None                    # type: _jit_internal.Optional[torch.Tensor]
                                 ):
    # type: (...) -> torch.Tuple[torch.Tensor, _jit_internal.Optional[torch.Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        # ######blocks that can be ignore###### # (start)
        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
        # ######blocks that can be ignore###### #(end)
    # ######blocks that can be ignore###### #(start)
    else:
        q_proj_weight_non_opt = jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    # ######blocks that can be ignore###### #(end)
    q = q * scaling

    # ######blocks that can be ignore###### #(start)
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                      torch.zeros((attn_mask.size(0), 1),
                                                  dtype=attn_mask.dtype,
                                                  device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None
    # ######blocks that can be ignore###### #(end)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # ######blocks that can be ignore###### #(start)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    # ######blocks that can be ignore###### #(end)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    # ######blocks that can be ignore###### # (start)
    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)
    # ######blocks that can be ignore###### # (end)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        # attn_mask has the shape(N*num_heads, L, S)

        # additive mode
        # attn_output_weights += attn_mask

        # masked mode
        attn_output_weights = attn_output_weights.masked_fill(
            attn_mask,
            1e-30
        )

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class TwoStreamMultiAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=False):
        super(TwoStreamMultiAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim), requires_grad=True)
        self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim), requires_grad=True)
            self.q_proj_bias = nn.Parameter(torch.empty(embed_dim), requires_grad=True)
            self.k_proj_bias = nn.Parameter(torch.empty(embed_dim), requires_grad=True)
            self.v_proh_bias = nn.Parameter(torch.empty(embed_dim), requires_grad=True)
        else:
            self.register_parameter('in_proj_bias', None)
            self.register_parameter('q_proj_bias', None)
            self.register_parameter('k_proj_bias', None)
            self.register_parameter('v_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.q_proj_bias, 0.)
            constant_(self.k_proj_bias, 0.)
            constant_(self.v_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query1, query2, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, training=True):
        return two_tream_multi_head_attention_forward(query1,
                                                      query2,
                                                      key,
                                                      value,
                                                      self.embed_dim,
                                                      self.num_heads,
                                                      q_proj_weight=self.q_proj_weight,
                                                      k_proj_weight=self.k_proj_weight,
                                                      v_proj_weight=self.v_proj_weight,
                                                      q_proj_bias=self.q_proj_bias,
                                                      k_proj_bias=self.k_proj_bias,
                                                      v_proj_bias=self.v_proj_bias,
                                                      dropout_p=self.dropout,
                                                      out_proj_weight=self.out_proj.weight,
                                                      out_proj_bias=self.out_proj.bias,
                                                      training=training,
                                                      key_padding_mask=key_padding_mask,
                                                      need_weights=need_weights,
                                                      attn_mask=attn_mask)


def two_tream_multi_head_attention_forward(query1, query2, key, value, embed_dim_to_check, num_heads,
                                           q_proj_weight, k_proj_weight, v_proj_weight, q_proj_bias,
                                           k_proj_bias, v_proj_bias, dropout_p, out_proj_weight,
                                           out_proj_bias, training=True, key_padding_mask=None,
                                           need_weights=True, attn_mask=None):
    tgt_len, bsz, embed_dim = query1.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    q1 = F.linear(query1, q_proj_weight, q_proj_bias)
    q2 = F.linear(query2, q_proj_weight, q_proj_bias)
    k = F.linear(key, k_proj_weight, k_proj_bias)
    v = F.linear(value, v_proj_weight, v_proj_bias)
    q1 = q1 * scaling
    q2 = q2 * scaling

    q1 = q1.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    q2 = q2.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    # emo shift stream
    attn_output_weights_s = torch.bmm(q2, k.transpose(1, 2))
    assert list(attn_output_weights_s.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        # attn_mask has the shape(N*num_heads, L, S)

        # additive mode
        # attn_output_weights_s += attn_mask

        # masked mode
        attn_output_weights_s = attn_output_weights_s.masked_fill(attn_mask, 1e-30)

    if key_padding_mask is not None:
        attn_output_weights_s = attn_output_weights_s.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights_s = attn_output_weights_s.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights_s = attn_output_weights_s.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_s = F.softmax(attn_output_weights_s, dim=-1)
    attn_output_weights_s = F.dropout(attn_output_weights_s, p=dropout_p, training=training)

    attn_output_s = torch.bmm(attn_output_weights_s, v)

    # emotion detection stream
    # # parallel mode
    # k = k + q2
    # v = k + q2

    # stack mode
    k = k + attn_output_s
    v = v + attn_output_s

    attn_output_weights_d = torch.bmm(q1, k.transpose(1, 2))
    assert list(attn_output_weights_d.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        # additive mode
        # attn_output_weights_d += attn_mask
        # masked mode
        attn_output_weights_d = attn_output_weights_d.masked_fill(attn_mask, 1e-30)

    if key_padding_mask is not None:
        attn_output_weights_d = attn_output_weights_d.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights_d = attn_output_weights_d.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights_d = attn_output_weights_d.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_d = F.softmax(attn_output_weights_d, dim=-1)
    attn_output_weights_d = F.dropout(attn_output_weights_d, p=dropout_p, training=training)

    attn_output_d = torch.bmm(attn_output_weights_d, v)

    assert list(attn_output_s.size()) == [bsz * num_heads, tgt_len, head_dim]
    assert list(attn_output_d.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output_s = attn_output_s.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_s = F.linear(attn_output_s, out_proj_weight, out_proj_bias)

    attn_output_d = attn_output_d.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_d = F.linear(attn_output_d, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights_s = attn_output_weights_s.view(bsz, num_heads, tgt_len, src_len)
        attn_s = attn_output_weights_s.sum(dim=1) / num_heads
        attn_output_weights_d = attn_output_weights_d.view(bsz, num_heads, tgt_len, src_len)
        attn_d = attn_output_weights_d.sum(dim=1) / num_heads
        return attn_output_d, attn_d, attn_output_s, attn_s
    else:
        return attn_output_d, None, attn_output_s, None


class TwoTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TwoTransformerLayer, self).__init__()
        self.self_attn = TwoStreamMultiAttention(d_model, nhead, dropout, False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, shiftw, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            shiftw: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, _, sw, _ = self.self_attn(src, shiftw, src, src, need_weights=False, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        # 2*tgt_len, bsz, emb_dim
        ss = torch.cat((src, shiftw), dim=0)
        # 2*tgt_len, bsz, emb_dim
        ss2 = torch.cat((src2, sw), dim=0)

        ss = ss + self.dropout1(ss2)
        ss = self.norm1(ss)
        if hasattr(self, "activation"):
            ss2 = self.linear2(self.dropout(self.activation(self.linear1(ss))))
        else:
            ss2 = self.linear2(self.dropout(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        src, sw = torch.chunk(ss, 2, dim=0)
        return src, sw


class PositionEmbedding(nn.Module):
    def __init__(self, input_dim, max_len=200, emb_dropout=0.1):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., input_dim, 2) * -(math.log(10000.)/input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        # (1, Max_len, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x(N, L, D)   pe_clip(1, L, D)
        pe_clip = self.pe[:, :x.size(1)]
        x = x + pe_clip
        x = self.emb_dropout(x)
        return x


def build_mask(utt_mask, spk_mask=None, bidirectional=False):
    """
    :param utt_mask: size(N, L)
    :param spk_mask: size(N, L)
    :param bidirectional: bool
    :return: (N, L, L) or [(N, L, L), (N, L, L)]
    """
    utt_mask = torch.matmul(utt_mask.unsqueeze(2), utt_mask.unsqueeze(1))
    if bidirectional is False:
        utt_mask = utt_mask.tril(0)
    umask = utt_mask.eq(0)
    if spk_mask is not None:
        batch_size = spk_mask.size(0)
        seq_len = spk_mask.size(1)
        mask1 = spk_mask.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        mask2 = spk_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        smask = torch.eq(mask1, mask2)
        smask = torch.masked_fill(smask, umask, False)
        smask = torch.eq(smask, False)
        return umask, smask
    return umask, None


class TwoStreamTransformer(nn.Module):
    def __init__(self, emb_dim, num_class, encoder_layer, num_layers, nheads, snheads=0, norm=None):
        super(TwoStreamTransformer, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.nheads = nheads
        self.snheads = snheads

        self.norm = norm
        self.pe = PositionEmbedding(emb_dim, 120, 0.0)
        self.shift_rep = nn.Parameter(torch.empty((1, 1, emb_dim)), requires_grad=True)
        self.d_classifier = nn.Linear(emb_dim, num_class)
        self.s_classifier = nn.Linear(emb_dim, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.shift_rep)
        xavier_uniform_(self.d_classifier.weight)
        constant_(self.d_classifier.bias, 0.)
        xavier_uniform_(self.s_classifier.weight)
        constant_(self.s_classifier.bias, 0.)

    def forward(self, src, umask=None, smask=None, src_key_padding_mask=None):
        """
        :param src: the sequence to the tool kit (required). Shape [bsz, seq_len, emb_dim]
        :param umask: the mask for the src sequence (optional). Shape [bsz, seq_len]
        :param smask: the mask identifying different speakers for the src sequence (optional). Shape [bsz, seq_len]
        :param src_key_padding_mask: the mask for the src keys per batch (optional). Shape [seq_len]
        :return:
        """
        bsz = src.size(0)
        seq_len = src.size(1)
        emb_dim = src.size(2)

        sw = self.shift_rep.expand((bsz, seq_len, emb_dim))
        # (bsz, seq_len, emb_dim)
        src = self.pe(src)
        sw = self.pe(sw)
        # (seq_len, bsz, emb_dim)
        output = src.transpose(0, 1)
        sw = sw.transpose(0, 1)

        if umask is not None:
            # (bsz, seq_len, seq_len)
            umask, smask = build_mask(umask, smask, False)
            umask = umask.unsqueeze(1).expand((bsz, self.nheads-self.snheads, seq_len, seq_len))
            if smask is not None:
                smask = smask.unsqueeze(1).expand((bsz, self.snheads, seq_len, seq_len))
                umask = torch.cat((umask, smask), dim=1)
            # (bsz*nheads, seq_len, seq_len)
            umask = umask.contiguous().view((-1, seq_len, seq_len))

        for i in range(self.num_layers):
            output, sw = self.layers[i](output, sw, src_mask=umask,
                                        src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)
            sw = self.norm(sw)

        # bsz, seq_len, emb_dim
        output = output.transpose(0, 1)
        sw = sw.transpose(0, 1)

        d_log_prob = F.log_softmax(self.d_classifier(output), dim=-1)
        s_log_prob = F.log_softmax(self.s_classifier(sw), dim=-1)

        return d_log_prob, s_log_prob


class BaseTransformer(nn.Module):
    def __init__(self, emb_dim, feed_dim, num_layers, num_head, s_num_head, num_class, dropout, activation, norm=None):
        super(BaseTransformer, self).__init__()
        self.num_layers = num_layers
        self.nheads = num_head
        self.snheads = s_num_head
        self.pe = PositionEmbedding(emb_dim, 120, 0.0)
        layer = TransformerEncoderLayer(emb_dim, num_head, feed_dim, dropout, activation)
        self.transformer = TransformerEncoder(layer, num_layers, None)
        self.classifier = nn.Linear(emb_dim, num_class)
        self.norm = norm

    def _reset_parameters(self):
        xavier_uniform_(self.classifier.weight)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, umask=None, smask=None, src_key_padding_mask=None):
        bsz = src.size(0)
        seq_len = src.size(1)

        src = self.pe(src)

        output = src
        # (seq_len, bsz, emb_dim)
        output = output.transpose(0, 1)

        if umask is not None:
            # (bsz, seq_len, seq_len)
            umask, smask = build_mask(umask, smask, False)
            umask = umask.unsqueeze(1).expand((bsz, self.nheads-self.snheads, seq_len, seq_len))
            if smask is not None:
                smask = smask.unsqueeze(1).expand((bsz, self.snheads, seq_len, seq_len))
                umask = torch.cat((umask, smask), dim=1)
            # (bsz*nheads, seq_len, seq_len)
            umask = umask.contiguous().view((-1, seq_len, seq_len))

        output = self.transformer(output, mask=umask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        # (bsz, seq_len, emb_dim)
        output = output.transpose(0, 1)

        log_prob = F.log_softmax(self.classifier(output), dim=-1)
        return log_prob


class MLTTransformer(BaseTransformer):
    def __init__(self, emb_dim, feed_dim, num_layers, num_head, s_num_head, num_class, dropout, activation, norm=None):
        super(MLTTransformer, self).__init__(emb_dim, feed_dim, num_layers, num_head,
                                             s_num_head, num_class, dropout, activation, norm)
        hidden_size = 50
        self.mlp2 = nn.Linear(emb_dim, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, 2)

        self.mlp = nn.Linear(emb_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_class)

    def forward(self, src, umask=None, smask=None, src_key_padding_mask=None):
        bsz = src.size(0)
        seq_len = src.size(1)

        src = self.pe(src)

        output = src
        # (seq_len, bsz, emb_dim)
        output = output.transpose(0, 1)

        if umask is not None:
            # (bsz, seq_len, seq_len)
            umask, smask = build_mask(umask, smask, False)
            umask = umask.unsqueeze(1).expand((bsz, self.nheads - self.snheads, seq_len, seq_len))
            if smask is not None:
                smask = smask.unsqueeze(1).expand((bsz, self.snheads, seq_len, seq_len))
                umask = torch.cat((umask, smask), dim=1)
            # (bsz*nheads, seq_len, seq_len)
            umask = umask.contiguous().view((-1, seq_len, seq_len))

        output = self.transformer(output, mask=umask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        # (bsz, seq_len, emb_dim)
        output = output.transpose(0, 1)

        log_prob1 = F.log_softmax(self.classifier(F.relu(self.mlp(output))), dim=-1)
        log_prob2 = F.log_softmax(self.classifier2(F.relu(self.mlp2(output))), dim=-1)

        return log_prob1, log_prob2


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0., bias=False, reuse=False):
        super(RelativeMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)
        self.r_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim), requires_grad=True)

        self.reuse = reuse
        if reuse is False:
            self.spk_embedding = nn.Embedding(2, self.embed_dim)
        else:
            self.spk_embedding = None
        if bias:
            self.q_proj_bias = nn.Parameter(torch.empty(embed_dim), requires_grad=True)
            self.k_proj_bias = nn.Parameter(torch.empty(embed_dim), requires_grad=True)
            self.v_proj_bias = nn.Parameter(torch.empty(embed_dim), requires_grad=True)
            self.r_proj_bias = nn.Parameter(torch.empty(embed_dim), requires_grad=True)
        else:
            self.register_parameter('q_proj_bias', None)
            self.register_parameter('k_proj_bias', None)
            self.register_parameter('v_proj_bias', None)
            self.register_parameter('r_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.r_proj_weight)
        if self.q_proj_bias is not None:
            constant_(self.q_proj_bias, 0.)
            constant_(self.k_proj_bias, 0.)
            constant_(self.v_proj_bias, 0.)
            constant_(self.r_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.reuse is False:
            xavier_uniform_(self.spk_embedding.weight)

    def _rel_shift(self, r, zero_tril=True):
        # r: (N, tgt_len, rel_len)
        bsz, tlen, rlen = r.size()
        zero_pad = torch.ones((bsz, tlen, 1), device=r.device, dtype=r.dtype)
        # (N, tgt_len, rel_len+1)
        r_padded = torch.cat((zero_pad, r), dim=2)
        # (N, rel_len+1, tgt_len)
        r_padded = r_padded.contiguous().view(bsz, rlen+1, tlen)
        # (N, tgt_len, rel_len)
        r = r_padded[:, 1:, :].view_as(r)
        if zero_tril:
            r = torch.tril(r, 0)
        return r

    def forward(self, query, key, value, rel_pe, attn_mask, utt_mask=None, spk_mask=None,
                key_padding_mask=None, s_q=None, s_k=None, need_weights=False):
        """
        multi-head self-attention with speaker and relative position encoding.
        src_len = tgt_len = rel_len
        :param query: (tgt_len, bsz, embed_dim) required.
        :param key: (tgt_len, bsz, embed_dim) required.
        :param value: (tgt_len, bsz, embed_dim) required.
        :param rel_pe: (rel_len, bsz, embed_dim) required.
        :param attn_mask: (bsz*nhead, tgt_len, tgt_len) required.
        :param utt_mask: (bsz*nhead, tgt_len, tgt_len, 1) optional. None if reuse
        :param spk_mask: (bsz, tgt_len, tgt_len) optional. None if reuse
        :param key_padding_mask: optional.
        :param s_q: optional. None if not reuse
        :param s_k: optional. None if not reuse
        :param need_weights: False by default.
        :return:
        """
        tgt_len, bsz, embed_dim = query.size()
        rel_len = rel_pe.size(1)
        assert key.size() == value.size()

        scaling = float(self.head_dim) ** -0.5

        q = F.linear(query, self.q_proj_weight, self.q_proj_bias)
        k = F.linear(query, self.k_proj_weight, self.q_proj_bias)
        v = F.linear(query, self.v_proj_weight, self.v_proj_bias)
        # (rel_len, bsz, nheads * hdim)
        r = F.linear(rel_pe, self.r_proj_weight, self.r_proj_bias)

        # attention computing
        # (bsz * nheads, tgt_len, hdim)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # (bsz * nheads, rel_len, hdim)
        r = r.contiguous().view(rel_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if self.reuse is False:
            idx = torch.zeros((tgt_len, bsz), device=query.device, dtype=torch.long)

            # (tgt_len, bsz, nheads * hdim) -> (bsz * nheads, tgt_len, hdim)
            s_q = self.spk_embedding(idx)
            s_q = s_q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            # (bsz, tgt_len, tgt_len, nheads * hdim)
            s_k = self.spk_embedding(spk_mask)
            s_k = s_k.contiguous().view(bsz, tgt_len * tgt_len, -1).transpose(0, 1)
            s_k = s_k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            # (bsz * nheads, tgt_len, tgt_len, hdim)
            s_k = s_k.contiguous().view(-1, tgt_len, tgt_len, self.head_dim)
            # mask par the padding embeddings
            s_k = s_k * utt_mask
        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        new_q = q + s_q

        # q*k
        # (bsz*nheads, tgt_len, hdim) * (bsz*nheads, hdim, tgt_len) -> (bsz*nheads, tgt_len, tgt_len)
        a1 = torch.bmm(new_q, k.transpose(1, 2))
        # q*s_k
        # (bsz*nheads, tgt_len, hdim) * (bsz*nheads, tgt_len, tgt_len, hdim) -> (bsz*nheads, tgt_len, tgt_len)
        a2 = torch.einsum('nid,nijd->nij', (new_q, s_k))
        # q*r
        # (bsz*nheads, tgt_len, hdim) * (bsz*nheads, hdim, rel_len) -> (bsz*nheads, tgt_len, rel_len)
        a3 = torch.bmm(s_q, r.transpose(1, 2))
        a3 = self._rel_shift(a3, True)
        attn_output_weights = a1 + a2 + a3
        attn_output_weights = attn_output_weights * scaling
        if attn_mask is not None:
            # attn_mask has the shape(N*num_heads, L, S)

            # additive mode
            # attn_output_weights_s += attn_mask

            # masked mode
            attn_output_weights = attn_output_weights.masked_fill(attn_mask, 1e-30)

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # (bsz*nheads, tgt_len, tgt_len)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        # (bsz*nheads, tgt_len, tgt_len) * (bsz*nheads, tgt_len, hdim) -> (bsz*nheads, tgt_len, hdim)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn = attn_output_weights.sum(dim=1) / self.num_heads
            return attn_output, attn
        else:
            return attn_output, None


class PositionEncoding(nn.Module):
    # Relative Position (abs(i-j)) pe[:, max_len-1] is the embedding of distance 1
    def __init__(self, input_dim, max_len=200):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(max_len-1, -1., -1.).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., input_dim, 2) * -(math.log(10000.) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # (1, Max_len, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        # x(N, L, D)   pe_clip(1, L, D)
        pe_clip = self.pe[:, self.max_len-seq_len:]
        pemb = pe_clip.expand_as(x)
        return pemb


class RelativeTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", reuse=False):
        super(RelativeTransformerLayer, self).__init__()
        self.self_attn = RelativeMultiHeadAttention(nhead, d_model, dropout, False, reuse=reuse)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, spk_mask, utt_mask, rel_pe, src_mask=None, src_key_padding_mask=None, s_q=None, s_k=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            spk_mask: the mask for the speaker's identification (required).
            utt_mask: the mask to identify the padding of key (required).
            rel_pe: the relatively positional embeddings (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            s_q: speaker encoding of self side reused by par layers.
            s_k: speaker encoding of other side reused by par layers.
        Shape:
            ...
        """
        src2, _ = self.self_attn(src, src, src, rel_pe, attn_mask=src_mask, utt_mask=utt_mask, spk_mask=spk_mask,
                                 key_padding_mask=src_key_padding_mask, s_q=s_q, s_k=s_k, need_weights=False)

        ss = src + self.dropout1(src2)
        ss = self.norm1(ss)
        if hasattr(self, "activation"):
            ss2 = self.linear2(self.dropout(self.activation(self.linear1(ss))))
        else:
            ss2 = self.linear2(self.dropout(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        return ss


class RelativeTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(RelativeTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, spk_mask, utt_mask, rel_pe, mask, src_key_padding_mask=None, s_q=None, s_k=None):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, spk_mask, utt_mask, rel_pe, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask, s_q=s_q, s_k=s_k)
        if self.norm:
            output = self.norm(output)

        return output


class RelativeTransformer(nn.Module):
    def __init__(self, emb_dim, nhead, snhead, pff_dim, num_layer, num_class,
                 dropout=0.1, max_len=200, activation='relu', reuse=False):
        super(RelativeTransformer, self).__init__()
        self.nhead = nhead
        self.snhead = snhead
        self.head_dim = emb_dim // nhead
        self.pe = PositionEncoding(emb_dim, max_len)
        self.reuse = reuse
        if reuse:
            self.spk_embedding = nn.Embedding(2, emb_dim)
        else:
            self.spk_embedding = None
        trans = RelativeTransformerLayer(emb_dim, nhead, pff_dim, dropout, activation, reuse)
        self.transformer = RelativeTransformerEncoder(trans, num_layer, None)
        self.classifier = nn.Linear(emb_dim, num_class)

    def forward(self, src, utt_mask, spk_mask):
        """
        :param src: (bsz, seq_len, emb_dim)
        :param utt_mask: (bsz, seq_len)
        :param spk_mask: (bsz, seq_len)
        :return:
        """
        bsz = src.size(0)
        seq_len = src.size(1)
        # (bsz, seq_len, emb_dim)
        rel_pe = self.pe(src)

        # (bsz, seq_len, seq_len)
        uttm, spkm = build_mask(utt_mask, spk_mask, False)
        mask1 = uttm.unsqueeze(1).expand(bsz, self.nhead, seq_len, seq_len)
        mask1 = mask1.contiguous().view((-1, seq_len, seq_len))
        src_mask = uttm.unsqueeze(1).expand(bsz, self.nhead-self.snhead, seq_len, seq_len)
        if self.snhead != 0:
            spkm1 = spkm.unsqueeze(1).expand(bsz, self.snhead, seq_len, seq_len)
            src_mask = torch.cat((src_mask, spkm1), dim=1)
        src_mask = src_mask.contiguous().view((-1, seq_len, seq_len))
        # (bsz, seq_len, seq_len)
        spk_mask_idx = torch.zeros(spkm.size(), device=src.device, dtype=torch.long)
        spk_mask_idx = torch.masked_fill(spk_mask_idx, spkm, 1)
        # (bsz*nhead, seq_len, seq_len)
        utt_mask_pad = torch.ones(mask1.size(), device=src.device, dtype=src.dtype)
        utt_mask_pad = torch.masked_fill(utt_mask_pad, mask1, 0)
        # (bsz*nhead, seq_len, seq_len, 1)
        utt_mask_pad = utt_mask_pad.unsqueeze(3)
        # (seq_len, bsz, emb_dim)
        src = src.transpose(0, 1)

        if self.reuse is True:
            tgt_len = src.size(0)
            bsz = src.size(1)
            idx = torch.zeros((tgt_len, bsz), device=src.device, dtype=torch.long)
            # (tgt_len, bsz, nheads * hdim) -> (bsz * nheads, tgt_len, hdim)
            s_q = self.spk_embedding(idx)
            s_q = s_q.contiguous().view(tgt_len, bsz * self.nhead, self.head_dim).transpose(0, 1)
            # (bsz, tgt_len, tgt_len, nheads * hdim)
            s_k = self.spk_embedding(spk_mask_idx)
            s_k = s_k.contiguous().view(bsz, tgt_len * tgt_len, -1).transpose(0, 1)
            s_k = s_k.contiguous().view(-1, bsz * self.nhead, self.head_dim).transpose(0, 1)
            # (bsz * nheads, tgt_len, tgt_len, hdim)
            s_k = s_k.contiguous().view(-1, tgt_len, tgt_len, self.head_dim)
            # mask par the padding embeddings
            s_k = s_k * utt_mask_pad
            output = self.transformer(src=src,
                                      spk_mask=None,
                                      utt_mask=None,
                                      rel_pe=rel_pe,
                                      mask=src_mask,
                                      src_key_padding_mask=None,
                                      s_q=s_q,
                                      s_k=s_k)
        else:
            output = self.transformer(src=src,
                                      spk_mask=spk_mask_idx,
                                      utt_mask=utt_mask_pad,
                                      rel_pe=rel_pe,
                                      mask=src_mask,
                                      src_key_padding_mask=None,
                                      s_q=None,
                                      s_k=None)

        # (bsz, seq_len, emb_dim)
        output = output.transpose(0, 1)

        log_prob = F.log_softmax(self.classifier(output), dim=-1)

        return log_prob


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """pred: size(N*L, C)
           target: size(N*L)
           mask: (N, L) if <pad> the element equals 0"""
        mask1 = mask.view(-1, 1)                 # (N*L, 1)
        target1 = target.view(-1)                # (N*L)
        if self.weight is None:
            loss = self.loss(pred*mask1, target1) / torch.sum(mask1)
        else:
            loss = self.loss(pred*mask1, target1) / torch.sum(self.weight[target1]*mask1.squeeze(1))

        return loss


class TwoStreamNLLLoss(nn.Module):
    def __init__(self, d_weight=None, s_weight=None, c1=1, c2=1):
        super(TwoStreamNLLLoss, self).__init__()
        self.d_weight = d_weight
        self.s_weight = s_weight
        self.d_loss = nn.NLLLoss(weight=d_weight, reduction='sum')
        self.s_loss = nn.NLLLoss(weight=s_weight, reduction='sum')
        self.c1 = c1
        self.c2 = c2

    def forward(self, d_pred, s_pred, d_target, s_target, mask):
        mask1 = mask.view(-1, 1)
        d_target1 = d_target.view(-1)
        s_target1 = s_target.view(-1)
        if self.d_weight is None:
            loss1 = self.d_loss(d_pred*mask1, d_target1) / torch.sum(mask1)
        else:
            loss1 = self.d_loss(d_pred*mask1, d_target1) / torch.sum(self.d_weight[d_target1]*mask1.squeeze(1))
        if self.s_weight is None:
            loss2 = self.s_loss(s_pred*mask1, s_target1) / torch.sum(mask1)
        else:
            loss2 = self.s_loss(s_pred*mask1, s_target1) / torch.sum(self.s_weight[s_target1]*mask1.squeeze(1))
        loss = loss1 * self.c1 + loss2 * self.c2
        return loss, loss1, loss2
