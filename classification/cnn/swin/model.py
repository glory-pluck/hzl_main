import timm
import torch
import os
def get_model(model_name,pretrained= False,checkpoint_path="",num_classes=1000):
    model_names_list = timm.list_models()
    assert (model_name in model_names_list),f"{model_name}不存在"
    model = None
    if checkpoint_path != "":##如果有传入checkpoint_path
        assert (os.path.exists(checkpoint_path)),"%s 路径不存在"%(checkpoint_path)

        model = timm.create_model(model_name=model_name,pretrained=pretrained,checkpoint_path=checkpoint_path)
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    
    else:#没有传入checkpoint_path
        model = timm.create_model(model_name=model_name,pretrained=pretrained) 
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model


# if __name__ =="__main__":
#     model = get_model(model_name  ="swin_large_patch4_window7_224",
#             checkpoint_path = "/home/hongzhenlong/hzl_main/classification/cnn/swin/swin_large_patch4_window7_224_22kto1k.pth",
#             num_classes = 5)
#     model.eval()
#     output = model(torch.ones((2, 3, 224, 224)))
#     print(output)


        
    