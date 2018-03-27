import torch
import torch.nn as nn

class GlobalAttention(nn.Module):
    def __init__(self, dim, out_dim=None):
        super(GlobalAttention, self).__init__()
        if out_dim is None:
            self.linear_in = nn.Linear(dim, dim, bias=False)
        else:
            self.linear_in = nn.Linear(dim, out_dim, bias=False)
        self.sm = nn.Softmax()

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        sm_attn = self.sm(attn)
        attn3 = sm_attn.view(sm_attn.size(0), 1, sm_attn.size(1))  # batch x 1 x sourceL
        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        return weightedContext, sm_attn, attn

class Attention_wrap(nn.Module):
    def __init__(self, dim, out_dim=None):
        super(Attention_wrap, self).__init__()
        self.attn_core = GlobalAttention(dim, out_dim)

    def forward(self, input, context):
        return self.attn_core(input, context)