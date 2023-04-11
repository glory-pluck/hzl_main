import torch
import dgl
from dgl.data.utils import generate_mask_tensor
from torch import cosine_similarity
import os
import json
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from myDataset import MyDataSet
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()
    return train_images_path, train_images_label, val_images_path, val_images_label


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


def getDataloader(data_path,batch_size):
    data_transform = {
    "train": transforms.Compose([transforms.Resize(size=[128,128]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(size=[128,128]),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                            images_class=train_images_label,
                            transform=data_transform["train"])
    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=data_transform["val"])
    train_loader = DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=16,
                                            collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=16,
                                            collate_fn=val_dataset.collate_fn)

    return train_loader,val_loader


def evaluate(model, graph, features, labels, mask,device):
    model.eval()
    features = features.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(model,g, features, labels, masks,device):
    # define train/val samples, loss function and optimizer
    train_mask, val_mask = masks
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    features = features.to(device)
    labels = labels.to(device)
    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(model,g, features, labels, val_mask, device)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


