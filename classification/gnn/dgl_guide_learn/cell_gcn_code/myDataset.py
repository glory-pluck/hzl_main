from dgl.data import DGLDataset
from dgl.data.utils import generate_mask_tensor
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
# from utils import data2graph
from dgl import  load_graphs
class MyDGLDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str 输入数据
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(MyDGLDataset, self).__init__(name='cell_data',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    # def download(self):
    #     # 将原始数据下载到本地磁盘
    #     pass   
    def process(self):
        root = self.raw_dir
        # features = np.load(file="/repository03/hongzhenlong_data/dgl_data/cell_data/train/all_features.npy")
        # labels = np.load(file="/repository03/hongzhenlong_data/dgl_data/cell_data/train/all_labels.npy")
        # g = data2graph(features)
        # 将原始数据处理为图、标签和数据集划分的掩码
        graphs, label_dict = load_graphs(root)
        # print(graphs,len(graphs),type(graphs[0]))
        graphs = graphs[0]
        # import dgl
        # for i in range(len(graphs)):
        #     if not isinstance(graphs[i], dgl.DGLGraph):
        #         graphs[i] = dgl.DGLGraph(graphs[i])
        labels = label_dict['labels']
        label_lenth = len(labels)
        my_list = range(label_lenth)
        idx_train, idx_val = train_test_split(my_list, train_size=0.6, random_state=42)
        idx_val, idx_test = train_test_split(idx_val, train_size=0.5, random_state=42)
        train_mask = generate_mask_tensor(
            _sample_mask(idx_train, label_lenth)
        )
        val_mask = generate_mask_tensor(_sample_mask(idx_val, label_lenth))
        test_mask = generate_mask_tensor(
            _sample_mask(idx_test, label_lenth)
        )
        # 划分掩码
        graphs.ndata['train_mask'] = train_mask
        graphs.ndata['val_mask'] = val_mask
        graphs.ndata['test_mask'] = test_mask
        self._num_classes = 7
        self._labels = labels
        self._g = graphs
        return self._g
        pass

    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        assert idx == 0, "这个数据集里只有一个图"
        return self._g
        pass

    def __len__(self):
        # 数据样本的数量
        return 1
        pass

    # def save(self):
    #     # 将处理后的数据保存至 `self.save_path`
    #     pass

    # def load(self):
    #     # 从 `self.save_path` 导入处理后的数据
    #     pass

    # def has_cache(self):
    #     # 检查在 `self.save_path` 中是否存有处理后的数据
    #     pass

def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels 