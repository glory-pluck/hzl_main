#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/04/07 15:04:56
@Author  :   hzl 
@Version :   1.0
'''
import math
import numpy as np

# def mean(limb):
#     x_sum = 0
#     y_sum = 0
#     for x,y in limb:
#         x_sum+=x
#         y_sum+=y
#     limb = [x/len(limb),y/len(limb)]  
#     return limb
# def get_point_change(point_list):
#     """
#     计算list中头尾方向
#     list存有多帧result
#     返回：左臂，右臂；左腿。右腿；的幅度
#     """
#     l_arm = []
#     r_arm = []
#     l_leg = []
#     r_leg = []
#     body_pose = 0
#     if point_list[0][0]>point_list[0][26]:###判断头尾朝向x轴 头——尾 为正
#         body_pose = -1#尾——头
#     else:
#         body_pose = 1#头——尾
#     point_list = np.array(point_list)
#     for i in [12,14,16,18,20,22]:
#         temp = point_list[-1][i]-point_list[0][i]
#         l_arm.append(temp)
#     for i in [11,13,15,17,19,20]:
#         temp = point_list[-1][i]-point_list[0][i]
#         r_arm.append(temp)
#     for i in [24,26,28,30,32]:
#         temp = point_list[-1][i]-point_list[0][i]
#         l_leg.append(temp)
#     for i in [23,25,27,29,31]:
#         temp = point_list[-1][i]-point_list[0][i]
#         r_leg.append(temp)

      
#     all_list = [mean(l_arm),mean(r_arm),mean(l_leg),mean(r_leg)]
#     return all_list,body_pose

# def get_action(all_list,body_pose):####解决近大远小问题：双目摄像
#     actions = {"l_arm":["l_arm_down","l_arm_up","l_arm_keep"],
#               "r_arm":["r_arm_down","r_arm_up","r_arm_keep"],
#               "l_leg":["l_leg_down","l_leg_up","l_leg_keep"],
#               "r_leg":["r_leg_down","r_leg_up","r_leg_keep"],}
#     true_action = ""
#     predict_action = []
#     l_arm,r_arm,l_leg,r_leg = all_list
#     limbs = ["l_arm","r_arm","l_leg","r_leg"]
#     if body_pose==1:
#         for limb,data in zip(limbs,all_list):
            
#             if data[1]<-0.001 and data[0]<-0.001:###up
#                 true_action = actions[limb][1]
#             elif data[1]>0.001 and data[0]>0.001:###dowm
#                 true_action = actions[limb][0]
#             else:
#                 true_action = actions[limb][2]
#             predict_action.append(true_action)
#     else:
#         for limb,data in zip(limbs,all_list):
#             if data[1]<-0.001 and data[0]>0.001:###up
#                 true_action = actions[limb][1]
#             elif data[1]>0.001 and data[0]<-0.001:###dowm
#                 true_action = actions[limb][0]
#             else:
#                 true_action = actions[limb][2]
#             predict_action.append(true_action)
         
#     return predict_action




##############################################################################
"""
action 判断
关键点 12，14，16，24，26，28
    11，13，15，23，25，27
1 关键点流动++角度流动
2 stgcn类似时序视频理解
"""
def compute_angle(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    # 分别计算两个向量的模：
    l_x=np.sqrt(vector1.dot(vector1))
    l_y=np.sqrt(vector2.dot(vector2))
    # 计算两个向量的点积
    dian=vector1.dot(vector2)
    # print('向量的点积=',dian)
    # print('向量的模=',l_x,l_y)
    # 计算夹角的cos值：
    cos_=dian/(l_x*l_y)
    # print('夹角的cos值=',cos_)
    # 求得夹角（弧度制）：
    angle_hu=np.arccos(cos_)
    # print('夹角（弧度制）=',angle_hu)
    # 转换为角度值：
    angle_d=angle_hu*180/np.pi
    # print('夹角=%f°'%angle_d)
    
    return int(angle_d)

def where_limb_up(point_list):
    """_summary_
        基本只有仰卧使用
        输入多帧的point_list通过肢体幅度判断那个肢体处于活动状态
    Args:
        point_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    ##那个肢体活动？ 默认none
    where_movable_limb = None
    body_pose_dict = {"th":0,
                      "ht":0}
    true_body = None
    action_dict = {"l_arm":"keep",
                   "r_arm":"keep",
                   "l_leg":"keep",
                   "r_leg":"keep"}
    ##step 1 计算头尾方向
    for i in range(len(point_list)):
        if point_list[i][0][0]>point_list[i][26][0] or point_list[i][0][0]>point_list[i][25][0]:###判断头尾朝向x轴 头——尾 为正
            body_pose_dict ["th"]+=1#尾——头
        elif point_list[i][0][0]<point_list[i][26][0] or point_list[i][0][0]<point_list[i][25][0]:
            body_pose_dict ["ht"]+=1#头——尾
    true_body = "th" if body_pose_dict["th"]>body_pose_dict ["ht"] else "ht"
    # print(point_list[-1][13],point_list[0][13])
    ##step 2 根据方向判断 哪个limb up
    if true_body=="th":#尾——头
        ##计算前缀差复杂度高,no 简单使用尾帧减头帧
        ##distance_diff 记录 la ra ll rl
        # print(point_list)
        
        distance_diff = [point_list[-1][13][0]-point_list[0][13][0],#la
                         point_list[-1][14][0]-point_list[0][14][0],#ra
                         point_list[-1][25][0]-point_list[0][25][0],#ll
                         point_list[-1][26][0]-point_list[0][26][0],#rl  
                         ]
        limb_up_index =  distance_diff.index(max(distance_diff))  if (max(distance_diff)/min(distance_diff))>10 else False
        if limb_up_index:
            # assert limb_up_index 
            for index, key in enumerate(action_dict.keys()):
                if limb_up_index==index:
                    action_dict[key] = "up"
                    where_movable_limb = "%s_up"%key
                    break
        else:
            where_movable_limb = None
    return where_movable_limb,true_body

def get_one_frame_angle(point_list):
    """_summary_

    Args:
        point_list (_type_): _description_
        一帧的

    Returns:
        _type_: _description_
        一帧的角度
    """
    point_list = np.array(point_list)
    r_arm_angle = compute_angle(point_list[14]-point_list[12],point_list[24]-point_list[12])
    r_elbow_angle = compute_angle(point_list[16]-point_list[14],point_list[12]-point_list[14])
    r_leg_angle = compute_angle(point_list[12]-point_list[24],point_list[26]-point_list[24])
    r_knee_angle = compute_angle(point_list[28]-point_list[26],point_list[24]-point_list[26])
    
    l_arm_angle = compute_angle(point_list[13]-point_list[11],point_list[23]-point_list[11])
    l_elbow_angle = compute_angle(point_list[15]-point_list[13],point_list[11]-point_list[13])
    l_leg_angle = compute_angle(point_list[11]-point_list[23],point_list[25]-point_list[23])
    l_knee_angle = compute_angle(point_list[27]-point_list[25],point_list[23]-point_list[25])
    
    ##上（左右）下（左右）
    angle_list=  [l_arm_angle,l_elbow_angle,r_arm_angle,r_elbow_angle,l_leg_angle,l_knee_angle,r_leg_angle,r_knee_angle] 
    # angle_list =  [l_arm_angle,l_elbow_angle,l_leg_angle,l_knee_angle,r_arm_angle,r_elbow_angle,r_leg_angle,r_knee_angle]
    return angle_list



def get_action1(angle_list):####解决近大远小问题：双目摄像
    """
    判断是坐姿还是平躺
    """
    l_arm_angle,l_elbow_angle,r_arm_angle,r_elbow_angle,l_leg_angle,l_knee_angle,r_leg_angle,r_knee_angle=angle_list
    body_pose = None
    if l_leg_angle>150:
        body_pose = "lie"
    else:
        body_pose = "sit"
        
    actions = {"l_arm":["l_arm_down","l_arm_up","l_arm_keep"],
            "r_arm":["r_arm_down","r_arm_up","r_arm_keep"],
            "l_leg":["l_leg_down","l_leg_up","l_leg_keep"],
            "r_leg":["r_leg_down","r_leg_up","r_leg_keep"],}
    true_action = None
    predict_action = []
    limbs = ["l_arm","r_arm","l_leg","r_leg"]
    body_pose = 'lie'
    if body_pose == 'lie':  
        #判断上肢
        temp_angle_list = [[l_arm_angle,l_elbow_angle],[r_arm_angle,r_elbow_angle]]
        for limb,data in zip(limbs[:2],temp_angle_list):
            if 45<data[0]<130 and data[1]>90:
                true_action = actions[limb][1]##arm_up
            else:
                true_action = actions[limb][0]##arm_down
            predict_action.append(true_action)
        #判断下肢
        temp_angle_list = [[l_leg_angle,l_knee_angle],[r_leg_angle,r_knee_angle]]
        for limb,data in zip(limbs[2:],temp_angle_list):
            if data[0]<150 and data[1]>130:
                true_action = actions[limb][1]##leg_up
            else:
                true_action = actions[limb][0]##leg_down
            predict_action.append(true_action)
    else:

        pass
        # predict_action = "None"
        # assert body_pose != "sit"            
    if not len(angle_list):
        angle_list = [-1,-1,-1,-1,-1,-1,-1,-1]          
    return predict_action,angle_list


def posture_correction(angle_list):
    """_summary_
    第一步
         根据角度判定姿态，sit lay none
         可以添加姿态的距离分布辅助判断
    Args:
        angle_list (_type_): _description_
        angle_list 角度列表，注意是单帧，一但姿态确定，启动对应程序
    Returns:
        _type_: _description_
        返回posture  sit or lay or none
    """
    
    posture = None
    upper_body = angle_list[:4]#l_arm_angle,l_elbow_angle,r_arm_angle,r_elbow_angle
    lower_body = angle_list[4:]#l_leg_angle,l_knee_angle,r_leg_angle,r_knee_angle
    if 60<=lower_body[0]<=130 and 60<=lower_body[2]<=130:# and 130<lower_body[1]>130 and lower_body[3]>130:
        posture = "sit"
    elif 130<lower_body[0] and 130<lower_body[2]: #and lower_body[1]>130 and lower_body[3]>130:
        posture = "lay"
    else:###
        pass 
    return posture

from typing import List
def againt_gravity_test(command:str,posture:str,angle_list,point_list):
    """_summary_
        计算当前指令发出后，对应肢体能否抵抗重力 True or False
        传入是当前指令
        以及角度值列表，注意是多帧列表
    Args:
        angle_list (_type_): _description_
        angle_list=  [
            第一帧的
            [l_arm_angle,l_elbow_angle,r_arm_angle,r_elbow_angle,l_leg_angle,l_knee_angle,r_leg_angle,r_knee_angle],
            第二帧的
            [l_arm_angle,l_elbow_angle,r_arm_angle,r_elbow_angle,l_leg_angle,l_knee_angle,r_leg_angle,r_knee_angle]
            ......
            ]

        posture 当前姿态
        command 指令
        
    """
    ##指令集合
    command_set = ["l_arm_up","r_arm_up","l_leg_up","r_leg_up",]
    person_limb,true_body = where_limb_up(point_list)
    if "test" not in command:
        assert command in command_set,"%s 指令不存在"%(command)
        assert person_limb==command_set,"指令:%s 与动作:%s不一致"%(command_set,person_limb)
    
    cmd_index = command_set.index(command.replace("_test",""))
    ##根据命令获取对应肢体的角度值
    current_cmd_angle_list = [lst[cmd_index*2:2*(cmd_index+1)] for lst in angle_list]
    """
    current_cmd_angle_list [ 多帧数据中对应cmd的角度 ]
    """
    main_angle_list = [angle[0] for angle in  current_cmd_angle_list]##主角度列表
    sub_angle_list = [angle[1] for angle in  current_cmd_angle_list]##副角度列表
    againt_gravity_degree = None   ##抵抗重力?
    max_main_angle,min_main_angle = max(main_angle_list),min(main_angle_list)
    max_sub_angle,min_sub_angle = max(sub_angle_list),min(sub_angle_list)
    ##坐姿只判断上肢抵抗标志
    if posture=="sit":
        againt_gravity_degree = True  if max_main_angle>45 and min_sub_angle>100  else False
    elif posture=="lay":
        if "arm" in command:###上肢判断
            againt_gravity_degree = True if max_main_angle>20 and min_sub_angle>100 else False
        if "leg" in command:###下肢判断
            againt_gravity_degree = True  if min_main_angle<155 and min_sub_angle>145 else False
    return againt_gravity_degree,true_body,main_angle_list,sub_angle_list    



def predict_action(command:str,point_list):
    """_summary_

    Args:
        command (str): _description_
        point_list (List[list]): _description_
        point_list 为多帧数据；默认15帧，根据和摄像机接受帧率数据判断秒数   
    Returns:
        _type_: _description_
        返回当前输入帧流的状态，对输入帧流的评分
    """
    
    action = None
    anglelist = []
    for point in point_list:
        angle = get_one_frame_angle(point)
        anglelist.append(angle)
    ###step1 body_posture 判断
    body_posture = None
    posture_dict={None:0,
                  "sit":0,
                  "lay":0}
    for angle in anglelist:##遍历所有帧的角度，取最大概率的姿态
        posture = posture_correction(angle)
        posture_dict[posture]+=1
    max_post_count = max(posture_dict.values())
    for key,value in zip(posture_dict.keys(),posture_dict.values()):
        if value==max_post_count:
            body_posture=key
            break
    assert posture is not None,"posture is %s"%posture
    ###step2 重力抵抗判断,通过说命令与动作一致
    againt_gravity_flag,true_body,main_angle_list,sub_angle_list = againt_gravity_test(command,body_posture,anglelist,point_list)
    
    if "test" in command:
        againt_gravity_flag = True
    assert againt_gravity_flag is not  None,"% 重力抵抗判断：%s,请重新测试"%againt_gravity_flag
    if "arm" in command:###上肢
        if againt_gravity_flag:####通过重力测试后,再次通过判断角度&漂移等 0，1，2
            ###发出指令,根据指令提取对应肢体特征
            ###对当前输入帧流判断状态（up,down,keep）
            point_feature,angle_feature = feature_extract(command=command,point_list=point_list,anglelist=main_angle_list)
            mean_angle = np.mean(main_angle_list)
            if angle_feature<100 and mean_angle>40:
                action=0
            elif angle_feature>100 and mean_angle>40:
                action=1 
            else:
                action=2  
        else:###没有通过重力测试
            point_feature,angle_feature = feature_extract(command=command,point_list=point_list,anglelist=main_angle_list)
            mean_angle = np.mean(main_angle_list)
            ###标准待定，默认4
            action = 3
    if "leg" in command:###上肢
        if againt_gravity_flag:####通过重力测试后,再次通过判断角度&漂移等 0，1，2
            ###发出指令,根据指令提取对应肢体特征
            ###对当前输入帧流判断状态（up,down,keep）
            point_feature,angle_feature = feature_extract(command=command,point_list=point_list,anglelist=main_angle_list)
            mean_angle = np.mean(main_angle_list)
            if angle_feature<100 and mean_angle<160:
                action=0
            elif angle_feature>100 and mean_angle<160:
                action=1 
            else:
                action=2  
        else:###没有通过重力测试
            point_feature,angle_feature = feature_extract(command=command,point_list=point_list,anglelist=main_angle_list)
            mean_angle = np.mean(main_angle_list)
            ###标准待定，默认4
            action = 3
    
    ###step5 得到action判断数据 打分
     
    return  action

# def limb_pose_predict(true_body,point_feature,angle_feature):
#     """_summary_

#     Args:
#         true_body (_type_): _description_
#         point_list (_type_): _description_
#         anglelist (_type_): _description_

#     Returns:
#         _type_: _description_
#         返回当前输入帧流的状态，up，down，keep
#     """

#     return 0
    

def feature_extract(command,point_list:List[list],anglelist):
    """_summary_
        特征提取
        主要做漂移特征
    Args:
        point_list (List[list]): _description_
        anglelist (List[list]): _description_

    Returns:
        _type_: _description_
        返回特征
    """
    command_set = ["l_arm_up","r_arm_up","l_leg_up","r_leg_up",]
    assert command.replace("_test","") in command_set,"%s not  in %s"%(command,command_set)
    ##根据命令获取对应肢体的角度值&位置值
    cmd_index = command_set.index(command.replace("_test",""))
    #2023年04月08日 14:19:06 传入主角度，减少计算
    # current_cmd_angle_list = [lst[cmd_index*2:2*(cmd_index+1)] for lst in anglelist]
    ###筛选point_list，只保留command_set
    current_cmd_point_list = [[sublist[i] for i in [11,13,15,
                                                    12,14,16,
                                                    23,25,27,
                                                    24,26,28,]] for sublist in point_list]
    current_cmd_point_list = [sublist[cmd_index*3:3*(cmd_index+1)] for sublist in current_cmd_point_list]
    ###初始化 特征=0；计算并赋值
    point_feature = 0##点漂移方差
    angle_feature = 0##角度漂移方差
    point_list = np.array(current_cmd_point_list)
    # anglelist = np.array(current_cmd_angle_list)
    ###计算前缀差,每一帧数据都与第一帧比较，得到类似光流效果
    prefix_point_feature = [point_list[i]-point_list[0] for i in range(len(point_list))]
    prefix_angle_feature = [anglelist[i]-anglelist[0] for i in range(len(anglelist))]
    #axis 
    #参数0代表对每一列求值，
    #参数1代表对每一行求值，
    #无参数则求所有元素的值
    point_feature = np.var(prefix_point_feature,axis=0)##对应各个点的方差
    angle_feature = np.var(prefix_angle_feature)##对应各个关节的方差 
    return point_feature.mean(),angle_feature.mean()
    
         
def precess_action_list(action_list):
    """_summary_
        1 action 是0.5s 的帧数量的判断的action评分
    Args:
        action_list (_type_): _description_
    """
    from collections import Counter
    mode_score = Counter(action_list).most_common(1)[0][0]  # find the mode of the list
    return mode_score
    
    
