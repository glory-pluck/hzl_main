import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl import save_graphs
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return (dot_product / (norm_a * norm_b) + 1) / 2
def data2graph(features):
    """
    返回图结构（有多少节点，那些边连接），不包括数据
    """
    u_list = []
    v_list = []
    lenth_feature = len(features)
    g = dgl.DGLGraph()
    g.add_nodes(num= lenth_feature)
    # g.ndata["feature"] = feature
    print("lenth_feature",lenth_feature)
    # lenth_feature=100
    weights = []  # 每条边的权重
    for i in range(lenth_feature):
        for j in range(i+1,lenth_feature):###只计算上三角
            # print(torch.tensor(feature[i]).shape)
            # similarity = cosine_similarity(torch.tensor(features[i]),torch.tensor(features[j]),dim=0) 
            ##改用numpy计算
            vec1 = features[i]
            vec2 = features[j]
            similarity  = cosine_similarity(vec1,vec2)
            # print(similarity)
            if similarity>=0.8:
                u_list.append(i)
                v_list.append(j)
                weights.append(torch.tensor(similarity))
            # else:
            #     print(similarity)
        if i%100==0:
            print(i,"个完成")
    g.add_edges(u_list,v_list)###添加正向边；上三角
    g.add_edges(v_list,u_list)###添加反向边：下三角
    g.add_edges([u for u in range(lenth_feature)],[v for v in range(lenth_feature)])###添加对角线；对角线
    weights.extend(weights)
    weights.extend(torch.ones(lenth_feature))
    g.edata['w'] = torch.tensor(weights)  # 将其命名为 'w'
    ##对图节点赋值
    # print(g.edges())
    # bg = dgl.to_bidirected(g)##创建两个方向的边
    return g 
def processFeature2graph():
    features = np.load(file="/repository03/hongzhenlong_data/dgl_data/cell_data/train/all_features.npy")
    labels = np.load(file="/repository03/hongzhenlong_data/dgl_data/cell_data/train/all_labels.npy")
    features = features[:10000]
    labels = labels[:10000]
    g = data2graph(features)
    ###添加数据
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels) 
    g.ndata["feature"] = features
    g.ndata["labels"] = labels
    save_graphs(filename="/repository03/hongzhenlong_data/dgl_data/cell_data/train/train.bin",
                g_list=g,labels= {'labels': labels})
    print(g)

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

