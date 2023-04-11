import json
def data2yolo(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        for img_dict in data:
            left,top,w,h = img_dict["true_rect"]
            patch_name = img_dict["patch_name"]
            img_name = img_dict["img_name"]
            label  = img_dict["classes"][0]
            ###转换yolo格式
            x_c = (left+w/2)/1024
            y_c = (top+h/2)/1024
            w_ =w/1024
            h_ = h/1024
            txt_data = "%s %s %s %s %s\n"%(label,x_c,y_c,w_,h_)
            with open("/home/hongzhenlong/my_main/classification/utils/%s.txt"%patch_name,mode="a") as f:
               f.writelines(txt_data) 
data2yolo("/home/hongzhenlong/my_main/classification/utils/abnormal_cell_info.json")