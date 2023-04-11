from  utils import evaluate, getDataloader, train
from feature_extract import featureExtract
import torch
import numpy as np
import os
from utils import data2graph
import dgl
from dgl import save_graphs
from myDataset import MyDGLDataset
from model import SAGE
import torch.nn.functional as F
# def saveFeature2npy():
#     device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#     train_loader,val_loader  = getDataloader("/repository03/hongzhenlong_data/cell_data/12_cell_data_7class/train",256)
#     all_features,all_labels = featureExtract(val_loader,device)
#     sava_path = "/repository03/hongzhenlong_data/dgl_data/cell_data/val"
#     np.save(file=os.path.join(sava_path,"all_features.npy"),arr =all_features)
#     np.save(file=os.path.join(sava_path,"all_labels.npy"),arr = all_labels)
# saveFeature()
def processFeature2graph():
    features = np.load(file="/repository03/hongzhenlong_data/dgl_data/cell_data/train/all_features.npy")
    labels = np.load(file="/repository03/hongzhenlong_data/dgl_data/cell_data/train/all_labels.npy")
    features = features[:8000]
    labels = labels[:8000]
    g = data2graph(features)
    ###添加数据
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels) 
    g.ndata["feature"] = features
    g.ndata["labels"] = labels
    save_graphs(filename="/repository03/hongzhenlong_data/dgl_data/cell_data/train/train.bin",
                g_list=g,labels= {'labels': labels})
    print(g)

def main():
    data_path = "/repository03/hongzhenlong_data/dgl_data/cell_data/train/train.bin"
    data = MyDGLDataset(raw_dir=data_path)
    g = data[0]
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    features = g.ndata["feature"]
    labels = g.ndata["labels"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"]
    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = int(labels.max().item() + 1)
    model = SAGE(in_size, 16, out_size,device).to(device)
    # model training
    print("Training...")
    train(model,g, features, labels, masks,device)
    # test the model
    print("Testing...")
    acc = evaluate(model,g, features, labels, g.ndata["test_mask"],device)
    print("Test accuracy {:.4f}".format(acc))


if __name__=="__main__":
    processFeature2graph()
    # main()