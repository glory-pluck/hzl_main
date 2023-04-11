import os
import torch
from PIL import Image
from torchvision import transforms
from plot_confusion_matrix import plot_maxtrix
from utils import read_split_data
from model import get_model
import numpy as np
from my_dataset import MyDataSet
def verifyModel(data_path,save_fig_dir,weight_path,device=torch.device("cpu")):
    """_summary_
        验证模型 \n
        需要修改函数内部：\n
        1 模型路径,加载模型方式(由于使用timm加载模型框架) \n
        2 data_transform 中img_size大小 \n
        3 classes \n
    Args:
        save_fig_dir (_type_): _description_
        device (_type_, optional): _description_. Defaults to torch.device("cpu").
    """
    ##内部修改的#################################################################
    classes = ["00_NIML", "01_ASC-US", "02_LSIL",  "03_ASC-H","04_HSIL"]
    # weight_path = "/repository03/hongzhenlong_data/hzl_main_data/model_weight/swinv2_train/lmf_model-8-0.862.pth"
    model = get_model(model_name="swin_large_patch4_window7_224",
                        num_classes=5).to(device)
    model.load_state_dict(torch.load(weight_path,map_location=device))
    _, _, val_images_path, val_images_label = read_split_data(data_path,val_rate=1.0,train_flag=False)
    print(len(val_images_path),len(val_images_label))
    img_size = 224
    data_transform = transforms.Compose(
            [transforms.Resize(size=(img_size,img_size)),
            # transforms.CenterCrop(size=(128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ##############################################################################
    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=128,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=16,
                                                collate_fn=val_dataset.collate_fn)

    # 验证集上的表现情况：
    correct = 0
    total = 0
    labelsAll=[]
    predictedAll=[]
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labelsAll+=labels.cpu()
            predictedAll +=predicted.cpu()

    print('Accuracy of the network on the validation images: %.3f %%' % (100 * correct / total))
    acc = "%.3f %%"%(100 * correct / len(predictedAll))
    # 增加混淆矩阵的输出：
    y_true = np.array(labelsAll)
    y_pred = np.array(predictedAll)
    ####可以修改normalize 显示百分比
    plot_maxtrix(classes,y_true,y_pred,save_fig_dir,acc,normalize = False)


if __name__=="__main__":
    data_path = "/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/wsi_small_cnn_data/val"
    save_fig_dir = "/home/hongzhenlong/my_main/classification/cnn/swin"
    weight_path = "/home/hongzhenlong/root/classification/gnn/timm_gnn/weights/last.pth"
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    verifyModel(data_path,save_fig_dir,weight_path,device)
