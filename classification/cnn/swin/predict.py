import os
import torch
from PIL import Image
from torchvision import transforms
from plot_confusion_matrix import plot_maxtrix
from utils import read_split_data
from model import get_model
weight_path = "/home/lichaowei/deeplearning/classification/GNN/timm_gnn/workplace_loss_error/weights/best.pth"
data_path = "/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/wsi_small_cnn_data/val"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = get_model(model_name="swin_large_patch4_window7_224",
                    num_classes=5).to(device)
model.load_state_dict(torch.load(weight_path,map_location=device))

data_transform = transforms.Compose(
                            [
                            transforms.Resize((224,224 )),
                            # transforms.Resize(int(224 * 1.143)),
                            #    transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
_, _, val_images_path, val_images_label = read_split_data(data_path,val_rate=1.0,train_flag=False)
predictedAll=[]
correct=0
for img_path ,label in zip(val_images_path,val_images_label):
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        predict_cla = predict_cla.tolist()
        if predict_cla==label:
            correct+=1
        predictedAll.append(predict_cla)

print(len(predictedAll),len(val_images_label))
acc = "%.3f %%"%(100 * correct / len(predictedAll))
print('Accuracy of the network on the validation images: %.3f %%' % (100 * correct / len(predictedAll)))

classes = ["00_NIML", "01_ASC-US", "02_LSIL", "03_ASC-H", "04_HSIL"]
save_fig_dir = "/home/hongzhenlong/my_main/classification/cnn/swin"
plot_maxtrix(classes,val_images_label,predictedAll,save_fig_dir,acc)