import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class WeightedFocalLoss(nn.Module):
    """
        单类 FocalLoss
    """
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        cross_loss = nn.CrossEntropyLoss(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-cross_loss)
        F_loss = at*(1-pt)**self.gamma * cross_loss
        return F_loss.mean()

class MultiClassFocalLossWithAlpha(nn.Module):
    """
        多类 FocalLoss
    """
    def __init__(self, alpha=[0.03, 0.07, 0.2, 0.35, 0.35], gamma=2, reduction='mean', device=torch.device("cpu")):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = torch.tensor(gamma).to(device)
        self.reduction = reduction

    def forward(self, pred, target):

        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


  
    

class LDAMLoss(nn.Module):
    ##cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
    def __init__(self, cls_num_list, max_m=0.5, weight=[0.03, 0.07, 0.2, 0.35, 0.35], s=30,device=torch.device("cpu")):
        super(LDAMLoss, self).__init__()
        self.device = device
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.Tensor(cls_num_list)))# nj的四次开方
        m_list = m_list * (max_m / torch.max(m_list))# 常系数 C
        self.m_list = m_list.to(self.device)
        assert s > 0
        self.s = s# 
        if weight is not None:
            self.weight = torch.tensor(weight).to(device)# 和频率相关的 re-weight
        else:
            self.weight = None
        
    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)# 和 x 维度一致全 0 的tensor
        index.scatter_(1, target.data.view(-1, 1), 1)# dim idx input
        ''' 以上的idx指示的应该是一个batch的y_true '''
        index_float = index.type(torch.cuda.FloatTensor)
        index_float.to(self.device)
        self.m_list.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index.float().t()).view((-1, 1))
        batch_m = batch_m.expand((-1, x.size()[1]))
        x_m = x - batch_m# y 的 logit 减去 margin
        output = torch.where(index, x_m, x) # 按照修改位置合并
        if self.weight is not None:
            output = output * self.weight.unsqueeze(0)
        ldamLoss  = F.cross_entropy(self.s * output, target)
        return ldamLoss





class LMFLoss(nn.Module):
    """
    LMFLoss
    """
    def __init__(self,cls_num_list,per_cls_weights=[0.03, 0.07, 0.2, 0.35, 0.35],device=torch.device("cpu")):
        super(LMFLoss,self,).__init__()
        self.device  =device
        self.ldam_loss = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights,device=self.device)
        self.focal_loss = MultiClassFocalLossWithAlpha(alpha=[0.03, 0.07, 0.2, 0.35, 0.35], gamma=2, reduction="sum", device=self.device)
    
    def forward(self, x, target):
        loss_focal = self.focal_loss(x, target)
        loss_ldam = self.ldam_loss(x, target)
        return (loss_focal+loss_ldam)/2
    

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num=5, gamma=2, alpha=[0.03, 0.07, 0.2, 0.35, 0.35], reduction='mean',device=torch.device("cpu")):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1).to(device))
            print(self.alpha )
        else:
            self.alpha = Variable(torch.Tensor(alpha).reshape(class_num, 1).to(device))
            print(self.alpha )
        self.gamma = torch.Tensor(gamma).to(device)
        self.reduction = reduction
        self.class_num =  class_num
        self.device = device
    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        probs = torch.Tensor(probs).to(self.device )
        log_p = torch.Tensor(probs.log()).to(self.device )
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss  
class Focal_Loss(nn.Module):
    def __init__(self, weight=[0.03, 0.07, 0.2, 0.35, 0.35], gamma=2,reduction="mean",device =torch.device("cpu")):
        super(Focal_Loss,self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = torch.Tensor(weight).to(device)        # 是tensor数据格式的列表
        self.device =device

    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        preds = F.softmax(preds,dim=1)
        eps =torch.tensor(1e-7).to(self.device) 

        target = self.one_hot(preds.size(1), labels).to(self.device)
        ce = -1 * torch.log(preds+eps) * target
        floss = torch.pow((1-preds), 2) * ce
        floss = torch.mul(floss, self.weight)

        floss = torch.sum(floss, dim=1)
        if self.reduction == "mean":
            focal_loss =  torch.mean(floss)
        if self.reduction == "sum":
            focal_loss =  torch.sum(floss)
        return focal_loss
        # return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0),num))
        one[range(labels.size(0)),labels] = 1
        return one   



###测试    
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# cls_num_list = [10000,10,10,10,10]

# input_tensor = torch.randn(16, 5)
# target_tensor = torch.randint(low=0, high=5, size=(16,))

# # Compute the loss
# loss_fn = MultiClassFocalLossWithAlpha(reduction ="sum",device=device)
# loss_fn1 = LDAMLoss(cls_num_list=cls_num_list,device=device)
# loss_fn2 =LMFLoss(cls_num_list=cls_num_list,device=device)
# loss_fn3 =Focal_Loss(device=device,reduction="sum")

# loss = loss_fn(input_tensor.to(device), target_tensor.to(device))
# loss1 = loss_fn1(input_tensor.to(device), target_tensor.to(device))
# loss2 = loss_fn2(input_tensor.to(device), target_tensor.to(device))
# loss3 = loss_fn3(input_tensor.to(device), target_tensor.to(device))
# loss_list = [loss,loss1,loss2,loss3]
# # Print the loss
# print(loss_list)  