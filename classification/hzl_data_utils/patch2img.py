import os
from PIL import Image
from multiprocessing import Pool
def get_normal_bbox(txt_path,imgsz):
    """
    输入：每张图片的bbox txt
    返回：normal 的 x,y,w,h
    """
    all_normal_box =[]
    with open(txt_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            if values[0]=="0":
                x = float(values[1])*imgsz[0]
                y = float(values[2])*imgsz[1]
                w = float(values[3])*imgsz[0]
                h = float(values[4])*imgsz[1]
                left = x - w/2
                top = y - h/2
                right = x + w/2
                bottom = y + h/2
                all_normal_box.append([int(left-2),int(top-2),int(right+2),int(bottom+2)])
    return all_normal_box  



def save_img(all_normal_box,patch_img,wsi_name,save_dir):
    for normal_box in all_normal_box:
        # Crop the image
        cropped_image = patch_img.crop((normal_box))
        # save_dir = "/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/cut_data"
        # save_dir = "/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/final_wsi_gnn_data"
        save_path = os.path.join(save_dir,wsi_name,"00_NIML")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_name = str(len(os.listdir(save_path)))+".jpg"
        final_path = os.path.join(save_path,img_name)
        cropped_image.save(final_path)
        
def process_wsi(wsi_path,save_dir):
    wsi_name = wsi_path.split("/")[-1]
    txt_name_list = [patch for patch in os.listdir(wsi_path) if ".txt" in patch]
    txt_path_list = [os.path.join(wsi_path,name) for name in txt_name_list]
    for txt_path in txt_path_list:
        img_path = txt_path.replace("txt","jpg")
        img = Image.open(img_path)
        all_normal_box = get_normal_bbox(txt_path,img.size)
        save_img(all_normal_box,img,wsi_name,save_dir)
    print(wsi_path,"完成") 
            
def process_patch2img(wsi_dir,save_dir):
    wsi_path_list= [os.path.join(wsi_dir,name) for name in os.listdir(wsi_dir)]
    pool = Pool(processes=4)
    for wsi_path in wsi_path_list:
        pool.apply_async(func=process_wsi,args=(wsi_path,save_dir))