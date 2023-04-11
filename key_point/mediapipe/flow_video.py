import numpy as np
import cv2 as cv


def compute_optical_flow(video_path):
    video_path = video_path
    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'XVID') 
    out_name = video_path.split("/")[-1][:-4]
    out = cv.VideoWriter('/home/hongzhenlong/my_main/key_point/mediapipe/output_flow_%s.avi'%(out_name), fourcc, fps, (width, height))
    # 用于ShiTomasi角检测的参数
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # lucas kanade光流参数
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    #创建一些随机的颜色
    color = np.random.randint(0, 255, (100, 3))

    # 取第一帧，找出其中的角
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # 创建一个掩码图像用于绘图
    mask = np.zeros_like(old_frame)

    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 计算光流
        ##p1, st, err 
        #nextPtrs--> 输出一个二维点的向量，这个向量可以是用来作为光流算法的输入特征点，也是光流算法在当前帧找到特征点的新位置（浮点数）
        #status--> 标志，在当前帧当中发现的特征点标志status==1，否则为0;
        #err--> 向量中的每个特征对应的错误率.
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 选择好的点
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # 绘制轨道
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        out.write(img)

        # cv.imshow('frame', img)4
        # k = cv.waitKey(30) & 0xff
        # if k == 27:
        #     break

        # 现在更新上一帧和之前的点
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cap.release()
    out.release()

import os
video_data_dir = "/home/hongzhenlong/my_main/key_point/mediapipe/data"
video_path_list = [os.path.join(video_data_dir,video_name) for video_name in os.listdir(video_data_dir)]
for video_path in video_path_list:
    compute_optical_flow(video_path)