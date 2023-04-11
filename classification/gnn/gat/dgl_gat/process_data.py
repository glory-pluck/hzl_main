#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   process_data.py
@Time    :   2023/04/10 10:54:28
@Author  :   hzl 
@Version :   1.0
@summary :   从wsi读取数据，提取特征，将多个wsi构成图，保存
'''

import timm
import torch
import torch.nn as nn
def feature_extract(weight_path:str,device:torch.device("cpu")):
    model = timm.create_model(model_name="swin_large_patch4_window7_224",pretrained=False)
    # model.head = torch.nn.Linear(model.head.in_features, 5)
    model.head = nn.Identity()
    model.load_state_dict(torch.load(weight_path,map_location=device))
    for param in model.parameters():
        param.requires_grad = False
    in_put = torch.randn((1,3,224,224))
    out_put = model(in_put)
    print(out_put.shape)


if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    feature_extract(weight_path = "/home/lichaowei/deeplearning/classification/GNN/timm_gnn/workplace_noloss_adam/weights/best.pth",device=device)

