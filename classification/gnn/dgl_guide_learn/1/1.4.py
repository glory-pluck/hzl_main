# 1.4 从外部源创建图
"""
可以从外部来源构造一个 DGLGraph 对象，包括：

从用于图和稀疏矩阵的外部Python库（NetworkX 和 SciPy）创建而来。

从磁盘加载图数据。

"""
"""
从外部库创建图
"""
import dgl
import torch as th
import scipy.sparse as sp
spmat = sp.rand(100, 100, density=0.05) # 5%非零项
dgl.from_scipy(spmat)                   # 来自SciPy
import networkx as nx
nx_g = nx.path_graph(5) # 一条链路0-1-2-3-4
print(dgl.from_networkx(nx_g)) # 来自NetworkX
"""
注意，当使用 nx.path_graph(5) 进行创建时， DGLGraph 对象有8条边，而非4条。 
这是由于 nx.path_graph(5) 构建了一个无向的NetworkX图 networkx.Graph ，
而 DGLGraph 的边总是有向的。 所以当将无向的NetworkX图转换为 DGLGraph 对象时，
DGL会在内部将1条无向边转换为2条有向边。 使用有向的NetworkX图 networkx.DiGraph 可避免该行为。
"""
"""
DGL在内部将SciPy矩阵和NetworkX图转换为张量来创建图。因此，这些构建方法并不适用于重视性能的场景。
"""
nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])
print(dgl.from_networkx(nxg))
"""
从磁盘加载图{}
逗号分隔值（CSV）;JSON/GML 格式;DGL 二进制格式
"""