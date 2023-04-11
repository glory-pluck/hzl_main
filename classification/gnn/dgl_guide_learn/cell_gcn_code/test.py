from dgl.data.citation_graph import CiteseerGraphDataset

# 导入数据
dataset = CiteseerGraphDataset(raw_dir='/repository03/hongzhenlong_data/dgl_data')
graph = dataset[0]
print(graph)
# # 获取划分的掩码
train_mask = graph.ndata['train_mask']
# val_mask = graph.ndata['val_mask']
# test_mask = graph.ndata['test_mask']
print(train_mask.shape)
# # 获取节点特征
# feats = graph.ndata['feat']
print(graph.ndata['feat'].shape)
# # 获取标签
# labels = graph.ndata['label']