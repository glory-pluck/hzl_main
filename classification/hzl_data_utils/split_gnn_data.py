#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   split_gnn_data.py
@Time    :   2023/04/03 21:18:55
@Author  :   hzl 
@Version :   1.0
'''
import os
import shutil

def split_wsi2_gnn_data(wsi_data_path,save_path):
    print(wsi_data_path,"处理中")
    # save_path = "/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/wsi_small_gnn_data"
    wsi_name = wsi_data_path.split("/")[-1]
    save_path_wsi = os.path.join(save_path,wsi_name)
    label_names = os.listdir(wsi_data_path)
    for label in label_names:
        label_path = os.path.join(wsi_data_path,label)##每个label的
        data_path  = [os.path.join(label_path,name) for name in os.listdir(label_path)]
        train_path_list = data_path[:int(len(data_path)*0.8)]
        val_path_list = data_path[len(train_path_list):]
        for train_path in train_path_list:
            src_path = train_path
            dst_path = os.path.join(save_path_wsi,"train",label) 
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(src_path, dst_path)

        for val_path in val_path_list:
            src_path = val_path
            dst_path = os.path.join(save_path_wsi,"val",label) 
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            shutil.copy(src_path, dst_path)
    print(wsi_data_path,"处理完毕")


def process_wsi2_gnn_data(wsi_dir,save_dir,process=False):
    """_summary_

    Args:
        wsi_dir (_type_): _description_
        save_dir (_type_): _description_
        process (bool, optional): _description_. Defaults to False.
    """

    wsi_path_list = [os.path.join(wsi_dir,wsi_name) for wsi_name in os.listdir(wsi_dir)]
    if not process:###不使用多进程
        for wsi_data_path in wsi_path_list:
            split_wsi2_gnn_data(wsi_data_path,save_dir)###划分一张wsi
    else:
        from multiprocessing import Pool
        pool = Pool(processes=4) 
        for wsi_data_path in wsi_path_list:
            pool.apply_async(func=split_wsi2_gnn_data,args=(wsi_data_path,save_dir))
        pool.close()
        pool.join()
    # print("总数据：%s\n 训练数据：%s+测试数据：%s = %s\n 训练数据/测试数据：%s\n"%(count_all,count_train,count_val,count_train+count_val,count_train/count_val))
           
        
    
# if __name__=="__main__":
#     wsi_dir = "/home/hongzhenlong/hzl_main/data/abnormal_data"     ###起源数据
#     save_path = "/home/hongzhenlong/hzl_main/data/final_wsi_gnn_data"### gnn 数据保存
#     process_wsi2_gnn_data(wsi_dir,save_path,process=True)

