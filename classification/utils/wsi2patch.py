import openslide
import os
"""
get_wsi_patch 处理wsi划分patch

"""
def process_onewsi2patch(wsi_path,save_dir,patch_size=1024,):
    print(wsi_path,"处理中")
    with openslide.OpenSlide(wsi_path) as slide:
        width, height = slide.dimensions
        if ".svs" in wsi_path or ".ndpi" in wsi_path:
            wsi_name = wsi_path.replace(".ndpi","")
            wsi_name = wsi_name.replace(".svs","")
        wsi_name = wsi_name.split("/")[-1]
        patch_save_path = os.path.join(save_dir,wsi_name)
        if not os.path.exists(patch_save_path):
                os.makedirs(patch_save_path)
        
        for i in range(0, width, patch_size):
            for j in range(0, height, patch_size):
                region = slide.read_region((i, j), 0, (patch_size, patch_size))
                patch_name = wsi_name+"^position^x^%sy^%s.jpg"%(str(i),str(j))
                final_path = os.path.join(patch_save_path,patch_name)
                region.convert("RGB").save(final_path)
    print(wsi_path,"完成")

def process_wsi_patch(wsi_dir,save_dir,patch_size= 1024,process=False):
    """_summary_

    Args:
        wsi_dir (_type_): 处理wsi的目录
        save_dir (_type_): 处理成patch后的保存目录
        patch_size (int, optional): patch大小. 默认  1024.
        process (bool, optional): 是否使用进程加快处理. 默认False.
    """
    wsi_path_list = [os.path.join(wsi_dir,wsi_name) for wsi_name in os.listdir(wsi_dir)]
    save_dir = save_dir
    if not process:###不使用多进程
        for wsi_path in wsi_path_list:
            process_onewsi2patch(wsi_path,save_dir,patch_size)
    else:##使用多进程
        from multiprocessing import Pool      
        pool = Pool(processes=4)##4个进程的进程池
        # pool.map(process_onewsi2patch, wsi_path_list)
        for wsi_path in wsi_path_list:
            pool.apply_async(func=process_onewsi2patch,args=(wsi_path,save_dir,patch_size))
        """
        对Pool对象调用join()方法会等待所有子进程执行完毕，
        调用join()之前必须先调用close()，调用close()之后就不能继续添加新的Process了。
        """
        pool.close()
        pool.join()


 
         
# if __name__=="__main__":
#     wsi_dir = "/home/hongzhenlong/hzl_main/data/wsi_dir"
#     save_dir = "/home/hongzhenlong/hzl_main/data/save_path"
#     get_wsi_patch(wsi_dir=wsi_dir,save_dir=save_dir,patch_size=1024,process=True)


