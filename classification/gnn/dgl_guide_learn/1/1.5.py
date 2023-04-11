##1.5 异构图
"""
相比同构图，异构图里可以有不同类型的节点和边。
这些不同类型的节点和边具有独立的ID空间和特征。 
例如在下图中，”用户”和”游戏”节点的ID都是从0开始的，而且两种节点具有不同的特征。
https://data.dgl.ai/asset/image/user_guide_graphch_2.png
"""
#"创建异构图"
"""
在DGL中，一个异构图由一系列子图构成，一个子图对应一种关系。子图是异构还是同构？
每个关系由一个字符串三元组 定义 (源节点类型, 边类型, 目标节点类型) 。
由于这里的关系定义消除了边类型的歧义，DGL称它们为规范边类型。
"""
#DGL中创建异构图的示例
import dgl
import torch as th
# 创建一个具有3种节点类型和3种边类型的异构图
graph_data = {
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
}
g = dgl.heterograph(graph_data)

print(g.ntypes)#节点类型
print(g.etypes)#边类型
print(g.canonical_etypes)##异构图类型？

##注意，同构图和二分图只是一种特殊的异构图，它们只包括一种关系。
# #一个同构图
# dgl.heterograph({('node_type', 'edge_type', 'node_type'): (u, v)})
# # 一个二分图
# dgl.heterograph({('source_type', 'edge_type', 'destination_type'): (u, v)})
"""
与异构图相关联的 metagraph 就是图的模式。它指定节点集和节点之间的边的类型约束。 
metagraph 中的一个节点 u对应于相关异构图中的一个节点类型。 
metagraph 中的边 (u,v)表示在相关异构图中存在从 u型节点到 v型节点的边。
"""
print(g)
print(g.metagraph().edges())
##使用多种类型
"""
当引入多种节点和边类型后，用户在调用DGLGraph API以获取特定类型的信息时，
需要指定具体的节点和边类型。此外，不同类型的节点和边具有单独的ID。
"""
# 获取图中所有节点的数量
print(g.num_nodes())
# 获取drug节点的数量
print(g.num_nodes('drug'))
# 不同类型的节点有单独的ID。因此，没有指定节点类型就没有明确的返回值。
print(g.nodes('drug'))
# g.nodes()#DGLError: Node type name must be specified if there are more than one node types.
"""
为了设置/获取特定节点和边类型的特征，DGL提供了两种新类型的语法： 
g.nodes[‘node_type’].data[‘feat_name’] 和
g.edges[‘edge_type’].data[‘feat_name’] 
"""
# 设置/获取"drug"类型的节点的"hv"特征
g.nodes['drug'].data['hv'] = th.ones(3, 1)
print(g.nodes['drug'].data['hv'])
# 设置/获取"treats"类型的边的"he"特征
g.edges['treats'].data['he'] = th.zeros(1, 1)
print(g.edges['treats'].data['he'])
"""
如果图里只有一种节点或边类型，则不需要指定节点或边的类型。
"""
g = dgl.heterograph({
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'is similar', 'drug'): (th.tensor([0, 1]), th.tensor([2, 3]))
})
print(g.nodes())
# 设置/获取单一类型的节点或边特征，不必使用新的语法
g.ndata['hv'] = th.ones(4, 1)
print(g.ndata['hv'])
"""
当边类型唯一地确定了源节点和目标节点的类型时，
用户可以只使用一个字符串而不是字符串三元组来指定边类型。
例如， 对于具有两个关系 ('user', 'plays', 'game') 和 ('user', 'likes', 'game') 的异构图，
只使用 'plays' 或 'like' 来指代这两个关系是可以的。
"""

#边类型子图
"""
用户可以通过指定要保留的关系来创建异构图的子图，相关的特征也会被拷贝。
"""
g = dgl.heterograph({
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
})
g.nodes['drug'].data['hv'] = th.ones(3, 1)
# 保留关系 ('drug', 'interacts', 'drug') 和 ('drug', 'treats', 'disease') 。
# 'drug' 和 'disease' 类型的节点也会被保留
eg = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug'),
                                ('drug', 'treats', 'disease')])
print(eg)
# 相关的特征也会被拷贝
print(eg.nodes['drug'].data['hv'])

###将异构图转化为同构图
"""
异构图为管理不同类型的节点和边及其相关特征提供了一个清晰的接口。这在以下情况下尤其有用:

1不同类型的节点和边的特征具有不同的数据类型或大小。

2用户希望对不同类型的节点和边应用不同的操作。

如果上述情况不适用，并且用户不希望在建模中区分节点和边的类型，
则DGL允许使用 dgl.DGLGraph.to_homogeneous() API将异构图转换为同构图。 具体行为如下:

1用从0开始的连续整数重新标记所有类型的节点和边。

2对所有的节点和边合并用户指定的特征。
"""
g = dgl.heterograph({
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))})
g.nodes['drug'].data['hv'] = th.zeros(3, 1)
g.nodes['disease'].data['hv'] = th.ones(3, 1)
g.edges['interacts'].data['he'] = th.zeros(2, 1)
g.edges['treats'].data['he'] = th.zeros(1, 2)
# 默认情况下不进行特征合并
hg = dgl.to_homogeneous(g)
print('hv' in hg.ndata)
# 拷贝边的特征
# 对于要拷贝的特征，DGL假定不同类型的节点或边的需要合并的特征具有相同的大小和数据类型
# hg = dgl.to_homogeneous(g, edata=['he'])
#Cannot concatenate column he with shape Scheme(shape=(2,), dtype=torch.float32) and shape Scheme(shape=(1,), dtype=torch.float32)
# 拷贝节点特征
hg = dgl.to_homogeneous(g, ndata=['hv'])
print(hg.ndata['hv'])

"""原始的节点或边的类型和对应的ID被存储在 ndata 和 edata 中。"""
# 异构图中节点类型的顺序
print(g.ntypes)
# 原始节点类型
print(hg.ndata[dgl.NTYPE])
# 原始的特定类型节点ID
print(hg.ndata[dgl.NID])
# 异构图中边类型的顺序
print(g.etypes)
# 原始边类型
print(hg.edata[dgl.ETYPE])
# 原始的特定类型边ID
print(hg.edata[dgl.EID])