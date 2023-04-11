import time
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
from matplotlib import pyplot as plt
def plot_maxtrix(classes:list,y_true:list,y_pred:list,save_fig_dir:str,acc:str,normalize=False):
    # 使用sklearn工具中confusion_matrix方法计算混淆矩阵
    confusion_mat = confusion_matrix(y_true, y_pred)
    # print("confusion_mat.shape : {}".format(confusion_mat.shape))
    # print("confusion_mat : {}".format(confusion_mat))
    #归一化
    if normalize:
        confusion_mat = confusion_mat.astype('float') /  confusion_mat.sum(axis=1)[:, np.newaxis]
    # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    disp.plot(
        include_values=True,            # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,                        # 同上
        xticks_rotation="horizontal",   # 同上
        # values_format="d"               # 显示的数值格式
    )
    # plt.show()
    plt.title("Confusion Matrix acc:%s"%(acc))
    save_path = save_fig_dir+"/save_fig_at_%s.png"%time.strftime("%Y-%m-%d-%H:%M:%S")
    plt.savefig(save_path,dpi=300)
    print("save_fig in %s"%(save_path))

# acc = "%.3f %%"%(100 * 12 / 100)
# classes = ["00_NIML", "01_ASC-US", "02_LSIL", "03_ASC-H", "04_HSIL"]
# plot_maxtrix(classes,[1,2,3,4,5],[1,2,3,4,5],"./",acc)