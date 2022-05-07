import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    def __init__(self, args, dim_in, dim_out, cheb_k):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k, dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.adj_embeddings = nn.Parameter(torch.randn(args.embed_dim, args.num_nodes))

    def forward(self, x, adj_matrix):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num, _ = adj_matrix.shape
        supports = F.softmax(F.relu(torch.mm(adj_matrix, self.adj_embeddings)), dim=1)

        # np.save('./supports.npy', supports.cpu().numpy())
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias     #b, N, dim_out
        return x_gconv
