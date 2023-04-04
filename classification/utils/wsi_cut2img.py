#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   patch2img.py
@Time    :   2023/04/03 19:06:47
@Author  :   hzl 
@Version :   1.0
'''

import openslide 
import os 
def data_transform(all_point):
    """
    将四点坐标转换为左上角与宽高
    """
    new_all_point = []
    for i in range(len(all_point)):
        x1, y1, x2, y2, x3, y3, x4, y4 = [float(coord) for coord  in all_point[i]]
        width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
        height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
        x_min = min(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        new_all_point.append([int(x_min),int(y_min),int(width),int(height)])
    return new_all_point

def cut_onewsi_data2img(slide_path,tree_path,save_dir):
    """_summary_
        
    Args:
        slide_path (_type_): 一个wsi路径
        tree_path (_type_): 对应的xml路径
        save_dir (_type_): 保存目录
    """
    import xml.etree.ElementTree as ET
    print(slide_path,"处理中")
    # 打开wsi
    slide = openslide.OpenSlide(slide_path)
    # 解析包含注释的 XML 文件
    tree = ET.parse(tree_path)
    root = tree.getroot()
    all_point = []
    label_list = []
    for child_1 in root:
        if child_1.tag=="AnnotationGroups":
            continue
        for child_2 in child_1:
            label = child_2.attrib["PartOfGroup"]
            
            """_summary_
            label_map_dict 映射保存的文件夹名字
            """
            label_map_dict = {
                "0":"01_ASC-US",
                "1":"02_LSIL",
                "2":"03_ASC-H",
                "3":"04_HSIL",
                "ASC-US":"01_ASC-US",
                "LSIL":"02_LSIL",
                "ASC-H":"03_ASC-H",
                "HSIL":"04_HSIL",
            }
            if label in label_map_dict.keys():
                label = label_map_dict[label]
            label_list.append(label)
            for child_3 in child_2:
                x1 = child_3[0].attrib["X"]
                y1 = child_3[0].attrib["Y"]
                x2 = child_3[1].attrib["X"]
                y2 = child_3[1].attrib["Y"]
                x3 = child_3[2].attrib["X"]
                y3 = child_3[2].attrib["Y"]
                x4 = child_3[3].attrib["X"]
                y4 = child_3[3].attrib["Y"]
                point = [x1,y1,x2,y2,x3,y3,x4,y4]
                all_point.append(point)
    new_all_point = data_transform(all_point)###数据转换
    wsi_name = None
    if ".svs" in slide_path or ".ndpi" in slide_path:
        wsi_name = slide_path.replace(".ndpi","")
        wsi_name = wsi_name.replace(".svs","")
    wsi_name = wsi_name.split("/")[-1]
    save_wsi_path = os.path.join(save_dir,wsi_name)
    for label,point in zip(label_list,new_all_point):##保存
        img = slide.read_region(location=(point[0], point[1]),level=0,size=(point[2],point[3]))
        img_save_path = os.path.join(save_wsi_path,label)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        img_name = "%s^%s^%s.jpg"%(wsi_name,label,str(len(os.listdir(img_save_path))))
        final_path = os.path.join(img_save_path,img_name)
        img.convert("RGB").save(final_path)
    print(slide_path,"处理完成") 
    
def process_wsi_data2img(wsi_dir,xml_dir,save_dir,process=False):
    """_summary_    
    处理医生标注的异常细胞，裁剪

    Args:
        wsi_dir (_type_): 要处理 wsi 目录
            wsi_dir:
                wsi1name.npdi
                wsi2name.svs
                wsi3name.svs
        xml_dir (_type_): 标注数据的xml目录
            xml_dir:
                wsi1name.xml
                wsi2name.xml
                wsi3name.xml
        save_dir (_type_): 保存裁剪出的异常细胞的目录
            save_dir:
                wsi1name:
                    01_ASC-US:
                        .jpg
                    02_LSIL
                    03_ASC-H
                    04_HSIL     
        process (bool, optional): 是否使用多进程，默认False.
    """
    wsi_path_list = [os.path.join(wsi_dir,wsi_name) for wsi_name in os.listdir(wsi_dir)]
    xml_path_list = [os.path.join(xml_dir,xml_name) for xml_name in os.listdir(xml_dir)]
    wsi_path_list.sort()
    xml_path_list.sort()
    if not process:#不使用多进程
        for wsi_path,xml_path in zip(wsi_path_list,xml_path_list):
            cut_onewsi_data2img(wsi_path,xml_path,save_dir)
    else:
        from multiprocessing import Pool      
        pool = Pool(processes=4)##4个进程的进程池
        # pool.map(process_onewsi2patch, wsi_path_list)
        for wsi_path,xml_path in zip(wsi_path_list,xml_path_list):
            pool.apply_async(func=cut_onewsi_data2img,args=(wsi_path,xml_path,save_dir))
        """
        对Pool对象调用join()方法会等待所有子进程执行完毕，
        调用join()之前必须先调用close()，调用close()之后就不能继续添加新的Process了。
        """
        pool.close()
        pool.join()
        

# if __name__ == '__main__':
#     wsi_path = "/home/hongzhenlong/hzl_main/data/wsi_dir"
#     xml_path = "/home/hongzhenlong/hzl_main/data/xml"
#     save_dir = "/home/hongzhenlong/hzl_main/data/abnormal_data"
#     process_wsi_data2img(wsi_path,xml_path,save_dir,True)

