"""1.6 在GPU上使用DGLGraph"""
"""用户可以通过在构造过程中传入两个GPU张量来创建GPU上的 DGLGraph 。 
另一种方法是使用 to() API将 DGLGraph 复制到GPU，这会将图结构和特征数据都拷贝到指定的设备。"""
import dgl
import torch as th
u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata['x'] = th.randn(5, 3)   # 原始特征在CPU上
print(g.device)
cuda_g = g.to('cuda:0')         # 接受来自后端框架的任何设备对象
print(cuda_g.device)
print(cuda_g.ndata['x'].device)        # 特征数据也拷贝到了GPU上
# 由GPU张量构造的图也在GPU上
u, v = u.to('cuda:0'), v.to('cuda:0')
g = dgl.graph((u, v))
print(g.device)
"""任何涉及GPU图的操作都是在GPU上运行的。
因此，这要求所有张量参数都已经放在GPU上，其结果(图或张量)也将在GPU上。 
此外，GPU图只接受GPU上的特征数据"""
print(cuda_g.in_degrees())
print(cuda_g.in_edges([2, 3, 4]))                          # 可以接受非张量类型的参数
print(cuda_g.in_edges(th.tensor([2, 3, 4]).to('cuda:0')))  # 张量类型的参数必须在GPU上
cuda_g.ndata['h'] = th.randn(5, 4)                  # ERROR! 特征也必须在GPU上！