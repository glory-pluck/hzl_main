#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File  :   split_cnn_data.py
@Time  :   2023/04/03 21:00:58
@Author :   hzl 
@Version :   1.0
'''
import os
import shutil
from multiprocessing import Pool

def process_wsi2_cnn_data(wsi_dir,cnn_data_dir):
    """_summary_

  Args:
      wsi_dir (_type_): 为wsi上cut下来的
        wsi_dir:
          wsi1name:
              01_ASC-US:
                  .jpg
              02_LSIL
              03_ASC-H
              04_HSIL  
      train_save_dir (_type_): 格式为cnn的train目录
      val_save_dir (_type_): 格式为cnn的val目录
    """
    assert os.path.exists(wsi_dir),"%s未知路径"%wsi_dir
    assert os.path.exists(cnn_data_dir),"%s未知路径"%cnn_data_dir
    train_save_dir = os.path.join(cnn_data_dir,"train")
    val_save_dir =  os.path.join(cnn_data_dir,"val")
    wsi_name_list = os.listdir(wsi_dir)
    ###控制wsi数量
    # wsi_name_list = wsi_name_list[:100]
    wsi_path_list = [os.path.join(wsi_dir,wsi_name) for wsi_name in wsi_name_list]
    asc_us_path_list = []
    asc_h_path_list = []
    lsil_path_list = []
    hsil_path_list = []
    niml_path_list = []
    ##存储每个类别的路径信息
    map_dict = {
        "00_NIML":niml_path_list,
        "01_ASC-US":asc_us_path_list,
        "03_ASC-H":asc_h_path_list,
        "02_LSIL":lsil_path_list,
        "04_HSIL":hsil_path_list,
    }
    for wsi_path in wsi_path_list:###遍历所有wsi，并将每个类别的路径信息写入list
        label_list = os.listdir(wsi_path)
        for label in label_list:###一张wsi
            data_name_list = os.listdir(os.path.join(wsi_path,label))
            data_path_list = [os.path.join(os.path.join(wsi_path,label),name) for name in data_name_list ]
            map_dict[label].extend(data_path_list)
    all_length = len(niml_path_list)+len(asc_us_path_list)+len(asc_h_path_list)+len(lsil_path_list)+len(hsil_path_list)  

    ###划分train
    train_asc_us_path_list = asc_us_path_list[:int(len(asc_us_path_list)*0.8)]
    train_asc_h_path_list = asc_h_path_list[:int(len(asc_h_path_list)*0.8)]
    train_lsil_path_list = lsil_path_list[:int(len(lsil_path_list)*0.8)]
    train_hsil_path_list = hsil_path_list[:int(len(hsil_path_list)*0.8)]
    train_niml_path_list = niml_path_list[:int(len(niml_path_list)*0.8)]
    train_length = len(train_asc_us_path_list)+len(train_asc_h_path_list)+len(train_lsil_path_list)+len(train_hsil_path_list)+len(train_niml_path_list)
    ###划分val
    val_asc_us_path_list = asc_us_path_list[len(train_asc_us_path_list):]
    val_asc_h_path_list = asc_h_path_list[len(train_asc_h_path_list):]
    val_lsil_path_list = lsil_path_list[len(train_lsil_path_list):]
    val_hsil_path_list = hsil_path_list[len(train_hsil_path_list):]
    val_niml_path_list = niml_path_list[len(train_niml_path_list):]
    val_length = len(val_asc_us_path_list)+len(val_asc_h_path_list)+len(val_lsil_path_list)+len(val_hsil_path_list)+len(val_niml_path_list)

    val_map_dict = {    ##存储每个类别的路径信息
        "00_NIML":val_niml_path_list,
        "01_ASC-US":val_asc_us_path_list,
        "03_ASC-H":val_asc_h_path_list,
        "02_LSIL":val_lsil_path_list,
        "04_HSIL":val_hsil_path_list,
    }
    train_map_dict = {    ##存储每个类别的路径信息
        "00_NIML":train_niml_path_list,
        "01_ASC-US":train_asc_us_path_list,
        "03_ASC-H":train_asc_h_path_list,
        "02_LSIL":train_lsil_path_list,
        "04_HSIL":train_hsil_path_list,
    }
    count_train = 0
    count_val = 0
    pool = Pool(processes=4)
    
    for key in train_map_dict.keys():
        dst_path = os.path.join(train_save_dir,key)
        if len(train_map_dict[key])>0:       
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            for data_path in train_map_dict[key]: 
                filename = os.path.basename(data_path)
                final_dst_path = os.path.join(dst_path, filename)
                pool.apply_async(func=shutil.copyfile,args=(data_path, final_dst_path))
                # shutil.copyfile(data_path, final_dst_path)
                count_train+=1
                print("train:",count_train)
    for key in val_map_dict.keys():
        dst_path = os.path.join(val_save_dir,key)
        if len(val_map_dict[key])>0:
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            for data_path in val_map_dict[key]:
                filename = os.path.basename(data_path)
                final_dst_path = os.path.join(dst_path, filename)
                pool.apply_async(func=shutil.copyfile,args=(data_path, final_dst_path))
                # shutil.copyfile(data_path, final_dst_path)
                count_val+=1
                print("val:",count_val)
    print(" 所有数据：%s\n 训练数据：%s + 测试数据：%s = %s \n 训练数据/测试数据：%s\n "%(all_length,train_length,val_length,train_length+val_length,train_length/val_length))
        
            
# if __name__=="__main__":
#     cnn_data_dir = "/home/hongzhenlong/hzl_main/data/final_wsi_cnn_data"
#     wsi_dir ="/home/hongzhenlong/hzl_main/data/abnormal_data"###起源数据
#     process_wsi2_cnn_data(wsi_dir,cnn_data_dir)

