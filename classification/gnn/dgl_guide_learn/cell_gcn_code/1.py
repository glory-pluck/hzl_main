# import dgl
# import torch
# src_ids = torch.tensor([2, 3, 4])
# dst_ids = torch.tensor([1, 2, 3])
# # g1 = dgl.graph((src_ids, dst_ids))
# # g2 = dgl.graph((src_ids, dst_ids), num_nodes=100)
# g3 = dgl.DGLGraph()
# g3.add_nodes(num=100)
# a = torch.randn(100, 2)
# g3.ndata["feat"] = a
# g3.add_edges(src_ids,dst_ids)
# print(a[0],g3.ndata["feat"][0])
# print(g3)

# import numpy as np

# def _sample_mask(idx, l):
#     """Create mask."""
#     mask = np.zeros(l)
#     mask[idx] = 1
#     return mask
# idx_train = range(10)
# # Create a mask of length 5 with a value of 1 at index 2
# mask = _sample_mask(idx_train, 20)
# print(mask)  # Output: [0. 0. 1. 0. 0.]


import torch
from torch import cosine_similarity
from time import time
# start_time = time()
# for i in range(70000):
#     similarity = cosine_similarity(torch.randn([2048]),torch.randn([2048]),dim=0)
# end_time = time()
# print(end_time-start_time)
# print((end_time-start_time)/70000 )


import numpy as np
# starttime = time()
# for i in range(70000):
#     vec1 = np.random.rand(2048)
#     vec2 = np.random.rand(2048)
#     cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
# endtime = time()
# print(endtime-starttime)
# print((endtime-starttime)/70000)

# import dgl
# g = dgl.DGLGraph()
# g.add_nodes(num= 1000)
# u_list = [1,2,3,4,5]
# v_list = [1,2,3,4,5]
# g.add_edges(u_list,v_list)
# g.add_edges(v_list,u_list)
# bg = dgl.to_bidirected(g)##创建两个方向的边
# print(g)


# match_count = 0

# for i in range(70000):
#     for j in range(70000):
#         match_count += 1
# print(match_count)####4900000000

# match_count = 0

# for i in range(70000):
#     for j in range(i+1,70000):
#         match_count += 1
# print(match_count)####2449965000



