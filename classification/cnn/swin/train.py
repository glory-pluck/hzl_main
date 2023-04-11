#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/04/04 16:19:26
@Author  :   hzl 
@Version :   1.0
'''


import math
import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import get_model
from utils import read_split_data, train_one_epoch, evaluate
import torch.optim.lr_scheduler as lr_scheduler

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")

    tb_writer = SummaryWriter(log_dir="/home/hongzhenlong/my_main/classification/cnn/swin/lmf_log")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
    #                                transforms.CenterCrop(img_size),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    data_transforms_wei = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transforms_wei["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transforms_wei["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # model = create_model(num_classes=args.num_classes).to(device)
    model = get_model(model_name="swin_large_patch4_window7_224",
                      checkpoint_path=args.weights,
                      num_classes=args.num_classes).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
                
    ###by wei 需要修改train_one_epoch里的trainer等参考wei的处理
    # 定义优化函数
    trainer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
    # 更新学习率方式 - 余弦退火
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - 0.1) + 0.1  # cosine
    optimizer = lr_scheduler.LambdaLR(trainer, lr_lambda=lf)

    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    
    
    
    max_acc = 0
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if val_acc>max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), "/repository03/hongzhenlong_data/hzl_main_data/model_weight/swinv2_train/lmf_model-{}-{:.3f}.pth".format(epoch,max_acc))


if __name__ == '__main__':
    """_summary_
    1 loss
    2 优化器，学习率
    3 data_transforms
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-path', type=str,
                        default="/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/wsi_small_cnn_data/train")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/repository03/hongzhenlong_data/hzl_main_data/model_weight/swin_large_patch4_window7_224_22kto1k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)


