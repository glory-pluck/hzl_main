import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Define the linear transformation weight matrices for each attention head
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=self.dropout)
        
    def forward(self, input, adj_matrix):
        # Compute the attention coefficients using the attention mechanism
        h = torch.matmul(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
        attention = F.softmax(e + adj_matrix, dim=1)
        attention = self.dropout(attention)
        
        # Compute the output of the GAT layer
        h_prime = torch.matmul(attention, h)
        h_prime = self.dropout(h_prime)
        
        return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        self.layer1 = GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
        self.layer2 = GATLayer(nhid, nclass, dropout=dropout, alpha=alpha)
        
    def forward(self, x, adj_matrix):
        x = self.layer1(x, adj_matrix)
        x = F.elu(x)
        x = self.layer2(x, adj_matrix)
        x = F.softmax(x, dim=1)
        
        return x
