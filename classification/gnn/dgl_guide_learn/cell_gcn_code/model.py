import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
# class SAGE(nn.Module):
#     def __init__(self,in_feats,hid_feats,out_feats):
#         super().__init__()
#         # 实例化SAGEConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
#         self.conv1 = dglnn.SAGEConv(in_feats=in_feats,out_feats=hid_feats,aggregator_type="mean")
#         self.conv2 = dglnn.SAGEConv(in_feats=hid_feats,out_feats=out_feats,aggregator_type="mean")
    
#     def forward(self,graph, inputs):
#         # 输入是节点的特征
#         h = self.conv1(graph,inputs)
#         h = F.relu(h)
#         h = self.conv2(graph,inputs)
#         return h
    
# gcn_model = SAGE(1024,512,7)
# print(gcn_model)

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size,device):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)
        self.device = device

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            graph = graph.to(self.device)
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
