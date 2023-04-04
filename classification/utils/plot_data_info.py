#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   get_data_info.py
@Time    :   2023/04/04 10:01:03
@Author  :   hzl 
@Version :   1.0
'''
import os
from matplotlib import pyplot as plt
def plot_cnn_data(cnn_data_path,save_info_img_path=os.getcwd()):
    """_summary_
    五分类
    生成cnn数据的分布信息
    Args:
        path (_type_): _description_
        save_info_img_name (str, optional): _description_. Defaults to "cnn_data_info".
    """
    print(save_info_img_path)
    niml_count = 0
    ascus_count = 0
    asch_count = 0
    lsil_count = 0
    hsil_count = 0
    train_count = 0
    val_count = 0
    directory = cnn_data_path
    for dirname in os.listdir(directory):
        for label_name in os.listdir(os.path.join(directory,dirname)):
            for filename in os.listdir(os.path.join(directory,dirname,label_name)):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    if label_name=="00_NIML":
                        niml_count+=1
                    if label_name=="01_ASC-US":
                        ascus_count+=1
                    if label_name=="02_LSIL":
                        lsil_count+=1
                    if label_name=="03_ASC-H":
                        asch_count+=1
                    if label_name=="04_HSIL":
                        hsil_count+=1
                    if dirname=="train":
                        train_count+=1
                    if dirname=="val":
                        val_count+=1
    counts = [niml_count,ascus_count,lsil_count,asch_count,hsil_count,train_count,val_count]

    types = ["00_NIML", "01_ASC-US", "02_LSIL","03_ASC-H","04_HSIL","train","val"]
    plt.bar(types, counts)
    for a,b in zip(range(len(types)),counts):   #柱子上的数字显示
        plt.text(a,b,b,ha='center',va='bottom',fontsize=7)
    plt.xlabel("Image Types")
    plt.ylabel("Image Counts")
    plt.title("CNN_data_Distribution")
    plt.savefig("%s/%s.png"%(save_info_img_path,"cnn_data_info"))
    plt.close()
def plot_gnn_data(gnn_data_path,save_info_img_path=os.getcwd()):
    """_summary_
    五分类
    生成gnn数据的分布信息

    Args:
        path (_type_): _description_
        save_info_img_path (_type_, optional): _description_. Defaults to os.getcwd().
    """
    niml_count = 0
    ascus_count = 0
    asch_count = 0
    lsil_count = 0
    hsil_count = 0
    train_count = 0
    val_count = 0
    directory = gnn_data_path
    ################################################################################################
    for wsi_name in os.listdir(directory):
        for dirname in os.listdir(os.path.join(directory,wsi_name)):
            for label_name in os.listdir(os.path.join(directory,wsi_name,dirname)):
                for filename in os.listdir(os.path.join(directory,wsi_name,dirname,label_name)):
                    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                        if label_name=="00_NIML":
                            niml_count+=1
                        if label_name=="01_ASC-US":
                            ascus_count+=1
                        if label_name=="02_LSIL":
                            lsil_count+=1
                        if label_name=="03_ASC-H":
                            asch_count+=1
                        if label_name=="04_HSIL":
                            hsil_count+=1
                        if dirname=="train":
                            train_count+=1
                        if dirname=="val":
                            val_count+=1
    counts = [niml_count,ascus_count,lsil_count,asch_count,hsil_count,train_count,val_count]
    types = ["00_NIML", "01_ASC-US", "02_LSIL","03_ASC-H","04_HSIL","train","val"]
    plt.bar(types, counts)
    for a,b in zip(range(len(types)),counts):   #柱子上的数字显示
        plt.text(a,b,b,ha='center',va='bottom',fontsize=7)
    plt.xlabel("Image Types")
    plt.ylabel("Image Counts")
    plt.title("GNN_data Distribution")
    plt.savefig("%s/%s.png"%(save_info_img_path,"gnn_data_info"))
    plt.close()
def plot_origin_data(origin_data_path,save_info_img_path=os.getcwd()):
    """_summary_

    Args:
        origin_data_path (_type_): 源数据
            origin_data_path:
                wsi1name:
                    01_ASC-US:
                        .jpg
                    02_LSIL
                    03_ASC-H
                    04_HSIL  
        save_info_img_path (_type_, optional): _description_. Defaults to os.getcwd().
    """
    niml_count = 0
    ascus_count = 0
    asch_count = 0
    lsil_count = 0
    hsil_count = 0
    directory = origin_data_path
    ####origin_data
    for wsi_name in os.listdir(directory):
        for dirname in os.listdir(os.path.join(directory,wsi_name)):
            for filename in os.listdir(os.path.join(directory,wsi_name,dirname)):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    if dirname=="00_NIML":
                        niml_count+=1
                    if dirname=="01_ASC-US":
                        ascus_count+=1
                    if dirname=="02_LSIL":
                        lsil_count+=1
                    if dirname=="03_ASC-H":
                        asch_count+=1
                    if dirname=="04_HSIL":
                        hsil_count+=1
    counts = [niml_count,ascus_count,lsil_count,asch_count,hsil_count]
    types = ["00_NIML", "01_ASC-US", "02_LSIL","03_ASC-H","04_HSIL"]
    plt.bar(types, counts)
    for a,b in zip(range(len(types)),counts):   #柱子上的数字显示
        plt.text(a,b,b,ha='center',va='bottom',fontsize=7)
    plt.xlabel("Image Types")
    plt.ylabel("Image Counts")
    plt.title("origin_wsi_data Distribution")
    plt.savefig("%s/%s.png"%(save_info_img_path,"origin_wsi_data_info"))
    plt.close()

# if __name__ == '__main__':
#     plot_cnn_data("/home/hongzhenlong/hzl_main/data/final_wsi_cnn_data")
#     plot_origin_data("/home/hongzhenlong/hzl_main/data/abnormal_data")
#     plot_gnn_data("/home/hongzhenlong/hzl_main/data/final_wsi_gnn_data")


