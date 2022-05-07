import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.AGCRNCell import AGCRNCell


class AVWDCRNN(nn.Module):
    def __init__(self, args, node_num, dim_in, dim_out, cheb_k, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(args, node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(args, node_num, dim_out, dim_out, cheb_k))
        
    def forward(self, x, init_state, adj_matrix):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num

        # temporal_attention_state  = self.Tat(x)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, adj_matrix)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = x + torch.stack(inner_states, dim=1)
            # current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class ATGCN(nn.Module):
    def __init__(self, args):
        super(ATGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.adj_matrix = torch.from_numpy(args.adj_matrix).cuda()


        self.encoder = AVWDCRNN(args, args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.num_layers)
        # self.Tat = Temporal_Attention_layer(args.input_dim, args.num_nodes, args.horizon)

        #predictor
        self.end_conv = nn.Conv2d(12, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        # self.period_conv = nn.Conv2d(12, 12, (1, 3))
        #predictor
        # self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        # temporal_attention_state  = self.Tat(source)
        # x_hat = self.period_conv(source)

        source = source[:, :, : , :1]

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.adj_matrix)      #B, T, N, hidden
        output = output[:, :12, :, :]                                 #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output

