import mediapipe as mp
import cv2
import mediapipe as mp
from utils import predict_action,precess_action_list
import os

def pose_detect(video_path,command):
    print(video_path,"processing")
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    results_frame= []
    results_frame_max_legth = int(fps/2)
    current_predict_action_score = -1
    all_predict_action_score_list = []
    previous_predict_action_score = -1
    angle_list =[-1,-1,-1,-1,-1,-1,-1,-1]
    angle_history_dict = {###使用 angle_history_dict 存储四肢主要关节的5次记录；四个角度；5次记录
        "l_arm_angle":[],##列表长为：6，第0位为：1-5的元素 顺序排序的逆序数
        "r_arm_angle":[],
        "l_leg_angle":[],
        "r_leg_angle":[]
    }
    out_name = video_path.split("/")[-1][:-4]
    out = cv2.VideoWriter('/home/hongzhenlong/my_main/key_point/mediapipe/output_%s.avi'%(out_name), fourcc, fps, (width, height))
    with mp_pose.Pose(
        static_image_mode = False,###默认情况下它被初始化为假,即处理视频流
        model_complexity = 1,##姿势地标模型的复杂性
        smooth_landmarks = True,###表示 是否平滑关键点,筛选跨不同输入的地标 图像以减少抖动
        enable_segmentation = False,###表示 是否平滑关键点,除了姿势特征点外，解决方案还会生成 分段掩码
        smooth_segmentation = True,###如果设置为 true，该解决方案会过滤不同输入图像的分割掩码以减少抖动,如果 enable_segmentation 为假或 static_image_mode 为真则忽略。默认为真。
        min_detection_confidence=0.5,##置信度的阈值
        min_tracking_confidence=0.5,###跟踪级别的置信度
        ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                # print(len(results.pose_landmarks.landmark))
                if len(results_frame)==results_frame_max_legth:###控制存储的帧长度
                    """_summary_
                    """
                    current_predict_action_score = predict_action(command =command ,point_list = results_frame)##
                    all_predict_action_score_list.append(current_predict_action_score)
                    results_frame.pop(0)
                temp = []
                for i in range(33):##存储帧数据
                    x = results.pose_landmarks.landmark[i].x
                    y = results.pose_landmarks.landmark[i].y
                    temp.append([x,y])
                results_frame.append(temp)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # print(current_predict_action_score)
            
            cv2.putText(image, "previous frame:"+str(previous_predict_action_score), (5, 50 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "current frame:"+str(current_predict_action_score), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            previous_predict_action_score = current_predict_action_score 
            # angle_list_str = ["l_arm_angle","l_elbow_angle","r_arm_angle","r_elbow_angle","l_leg_angle","l_knee_angle","r_leg_angle","r_knee_angle"]
            # print(angle_list)
            # for i in range(8):
            #     cv2.putText(image, angle_list_str[i]+":"+str(angle_list[i]), (5,150+ i*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  
            out.write(image)
            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            # if cv2.waitKey(5) & 0xFF == 27:
            #   break
    cap.release()
    out.release()
    print(video_path,"processed")
    print("saved in /home/hongzhenlong/my_main/key_point/mediapipe/output_%s.avi"%(out_name))
    print("final score:%s"%precess_action_list(all_predict_action_score_list))
if __name__=="__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    data_path = "/home/hongzhenlong/my_main/key_point/mediapipe/data"
    # video_path_list = [os.path.join(data_path,data_name) for data_name in os.listdir(data_path)]
    video_path_list = [
        "/home/hongzhenlong/my_main/key_point/mediapipe/data/la.mp4",
        "/home/hongzhenlong/my_main/key_point/mediapipe/data/ra.mp4",
        "/home/hongzhenlong/my_main/key_point/mediapipe/data/ll.mp4",
        "/home/hongzhenlong/my_main/key_point/mediapipe/data/rl.mp4"
                       ]
    command_list = ["l_arm_up_test","r_arm_up_test","l_leg_up_test","r_leg_up_test",]
    # pool = Pool(processes=4)
    for video_path,command in zip(video_path_list[2:],command_list[2:]):
        pose_detect(video_path,command)