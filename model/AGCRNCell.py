import torch
import torch.nn as nn

from model.AGCN import AVWGCN


class AGCRNCell(nn.Module):
    def __init__(self, args, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(args, dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = AVWGCN(args, dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, adj_matrix):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, adj_matrix))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, adj_matrix))
        h = r *state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        zero_state = torch.zeros(batch_size, self.node_num, self.hidden_dim)
        nn.init.orthogonal_(zero_state)
        return zero_state
