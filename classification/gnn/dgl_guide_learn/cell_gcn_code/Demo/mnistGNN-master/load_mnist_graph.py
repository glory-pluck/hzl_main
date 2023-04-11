import numpy as np
import gzip
import torch
from torch_geometric.data import Data

def load_mnist_graph(data_size=60000):
    data_list = []
    labels = 0
    with gzip.open('/home/hongzhenlong/root/dgl_guide_learn/cell_gcn_code/Demo/mnistGNN-master/dataset/train-labels-idx1-ubyte.gz', 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
    for i in range(data_size):
        edge = torch.tensor(np.load('/home/hongzhenlong/root/dgl_guide_learn/cell_gcn_code/Demo/mnistGNN-master/dataset/graphs/'+str(i)+'.npy').T,dtype=torch.long)
        x = torch.tensor(np.load('/home/hongzhenlong/root/dgl_guide_learn/cell_gcn_code/Demo/mnistGNN-master/dataset/node_features/'+str(i)+'.npy')/28,dtype=torch.float) 

        d = Data(x=x, edge_index=edge.contiguous(),t=int(labels[i]))
        data_list.append(d)
        if i==10:
            break
            print("\rData loaded "+ str(i+1), end="  ")

    print("Complete!")
    return data_list

data_list = load_mnist_graph()
print(data_list)
