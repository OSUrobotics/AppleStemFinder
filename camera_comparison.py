#!/usr/bin/env python3

###############
# Author: Yi Herng Ong, Chang Ju Lee, Nigel Swenson
# Purpose: find the stem orientation based on an image of the apple
#
# ("/home/graspinglab/NearContactStudy/MDP/jaco/jaco.xml")
#
###############


# import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from random import random
import time
import sys
import os
import inspect
import sys
import copy


# import rospy
# import moveit_commander
# import moveit_msgs.msg
# import geometry_msgs.msg
# import sensor_msgs.msg
# import shape_msgs.msg
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import tf2_msgs.msg
# import tf
# from tf.transformations import euler_from_quaternion, quaternion_about_axis
from sklearn.decomposition import PCA


def degreetorad(degree):
    """  converts from degrees to radians
    @param degree - angle in degrees"""
    rad = degree / (180 / math.pi)
    return rad


def get_point(pose_msg):
    """  Extracts the point from a pose message and puts it in a np array
    @param pose_msg - pose message containing a point"""
    point = np.array([pose_msg.point.x, pose_msg.point.y, pose_msg.point.z])
    return point


class image_finder():
    """  Divides the pixels into groups based on their value and saves masks
    @function get_images - check to see if there is a rviz and real image
    saved, then return the images if true
    @function update_real_image - save image from real camera
    @function update_rviz_image - save image from rviz"""
    def __init__(self):
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/image_raw", Image, self.update_real_image, queue_size=1)
        rospy.Subscriber("/camera1/image", Image, self.update_rviz_image, queue_size=1)
        self.real_image = []
        self.rviz_image = []

    def update_real_image(self, image_data):
        self.real_image = self.bridge.imgmsg_to_cv2(image_data, desired_encoding='passthrough')[100:370, 110:498]

    def update_rviz_image(self, image_data):
        self.rviz_image = self.bridge.imgmsg_to_cv2(image_data, desired_encoding='passthrough')[100:370, 110:498]

    def get_images(self):
        if self.real_image != [] and self.rviz_image != []:
            real_im, rviz_im = self.real_image, self.rviz_image
            self.real_image, self.rviz_image = [], []
            return rviz_im, real_im
        else:
            print('real im', np.shape(self.real_image))
            print('fake im', np.shape(self.rviz_image))
            return False


def blur_im(target, fake_img, real_img, save_fl=0):
    """  Dialates the image to reduce the effect of noise for higher efficiency
    in cluster matching
    @param target - folder name where images are saved
    @param fake_img - image from rviz to be blurred
    @param real_img - real image to be blurred
    @param save_fl - Bool to save images or not"""
    img_blur = cv2.blur(fake_img, (7, 7))
    img_r_blur = cv2.blur(real_img, (7, 7))
    if save_fl == 1:
        plt.imsave(target + "/img_dila.png", img_blur)
        plt.imsave(target + "/real_img_dila.png", img_r_blur)
        plt.imsave(target + "/real_image.png", real_img)
        plt.imsave(target + "/rviz_image.png", fake_img)
    print("blur_im_out")
    return img_blur, img_r_blur, real_img


def RGB2YUV(target, mask, real, save_fl=0):
    """  Changes image coordinates from RGB to YUV to improve accuracy
    @param target - folder name where images are saved
    @param mask - image from rviz
    @param real - image from real world
    @param save_fl - Bool to save images or not"""
    mask_out = cv2.cvtColor(mask, cv2.COLOR_BGR2YUV)
    real_out = cv2.cvtColor(real, cv2.COLOR_BGR2YUV)
    if save_fl == 1:
        plt.imsave(target + "/mask_YUV.png", mask_out)
        plt.imsave(target + "/real_YUV.png", real_out)
    return mask_out, real_out


def Kmeanclus(target, mask, real, parK=5, save_fl=0):
    """  Divides the pixels into groups based on their value and saves masks
    @param target - folder name where images are saved
    @param mask - image from rviz
    @param real - image from real world
    @param parK - number of clusters to generate
    @param save_fl - Bool to save images or not"""
    Z_mask = np.float32(mask.reshape((-1, 3)))
    Z_real = np.float32(real.reshape((-1, 3)))
    h, w, _ = real.shape
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = parK
    ret_real, label_real, center_real = cv2.kmeans(Z_real, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    ret_mask, label_mask, center_mask = cv2.kmeans(Z_mask, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label_real_out = label_real.reshape(h, w)
    label_mask_out = label_mask.reshape(h, w)
    if save_fl == 1:
        plt.imsave(target + "/real_K" + str(K) + "_mod.png", label_real_out)
        plt.imsave(target + "/mask_K" + str(K) + ".png", label_mask_out)
    return label_mask_out, label_real_out, center_real, center_mask


def grab_cuting(target, K, sim_img, real, rviz_mask=[], nwmask=[], save_fl=0):
    """  Uses grabCut to separate the image into foreground and background
    using some of the pixels in the mask determined previously as markers of
    which part belongs in the foreground
    @param target - folder name where images are saved
    @param K - int indicating stem (0) or apple (1)
    @param sim_img - image from rviz
    @param real - image from real world
    @param nwmask - mask from real world
    @param rviz_mask - mask from rviz
    @param save_fl - Bool to save images or not"""
    
    TODO Fix this so that it uses some of the pixels from the mask near where 
    we expect the apple and stem as foreground markers for grabcut
    
    img = real
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    img2 = sim_img
    mask2 = np.zeros(img2.shape[:2], np.uint8)
    bgdModel2 = np.zeros((1, 65), np.float64)
    fgdModel2 = np.zeros((1, 65), np.float64)
    mj_seg = [0]
    if K == 0:
        rect = (300, 300, 100, 100)
    else:
        rect = (250, 200, 150, 200)
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_RECT)
    mask2, bgdModel2, fgdModel2 = cv2.grabCut(img2,mask2, rect, bgdModel2, fgdModel2, 7, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask2 = np.where((mask2 == 2) | (mask2 == 0), 0, 1).astype('uint8')
    img = img * mask[:, :, np.newaxis]
    img2 = img2 * mask2[:, :, np.newaxis]
    if save_fl == 1:
        plt.imsave(target + "/Grab" + str(K) + ".png", img)
        plt.imsave(target + "/Grab_rviz" + str(K) + ".png", img2)
        plt.imsave(target + "/Grab_mask" + str(K) + ".png", mask)
        plt.imsave(target + "/Grab_mask_rviz" + str(K) + ".png", mask2)
    return mask, mj_seg, mask2


# Currently this is unused but we could use it in an iterative approach
def cal_score(inmask, insimu):
    """  Calculates the number of pixels that are the same label in the
    sim image and the real image
    @param inmask - processed mask from real image
    @param insimu - processed mask from sim image"""

    h, w = inmask.shape
    score_mask = np.zeros(inmask.shape)
    score_mask[(insimu == 0) & (inmask == 1)] = 1
    score = np.count_nonzero(score_mask)
    sim_score = insimu.size - np.count_nonzero(insimu)
    mask_score = np.count_nonzero(inmask)
    return score


def findCOM(rviz, real, rviz2, real2, camera_matrix, Z):
    """  Finds the center of mass of the stem and apple in the real and rviz
    images in world coordinates
    @param sim_img - image from rviz
    @param rviz - stem mask from rviz
    @param real - stem mask from real world
    @param rviz2 - apple mask from rviz
    @param real2 - apple mask from real world
    @param camera_matrix - camera mtx used by the camera in real world AND RVIZ
    @param Z - Distance from the camera to the stem/apple"""
    hfull = 480
    wfull = 640
    h_adj = 100
    w_adj = 110
    
    #we append j,i not i,j because the image is saved in open cv as [h,w] and we want our pose to be x,y not y,x
    
    rviz_stem_pos = np.argwhere(rviz == 1)
    rviz_stem_pos += [h_adj, w_adj]
    rviz_stem_pos[:, [1, 0]] = rviz_stem_pos[:, [0, 1]]
    rviz_stem_pos = np.average(rviz_stem_pos, axis=0)
    rviz_stem_pos = rviz_stem_pos * 2 / np.array([wfull, hfull]) - 1
    
    rviz_apple_pos = np.argwhere(rviz2 == 1)
    rviz_apple_pos += [h_adj, w_adj]
    rviz_apple_pos[:, [1, 0]] = rviz_apple_pos[:, [0, 1]]
    rviz_apple_pos = np.average(rviz_apple_pos, axis=0)
    rviz_apple_pos = rviz_apple_pos * 2 / np.array([wfull, hfull]) - 1
    
    real_stem_pos = np.argwhere(real == 1)
    real_stem_pos += [h_adj, w_adj]
    real_stem_pos[:, [1, 0]] = real_stem_pos[:, [0, 1]]
    real_stem_pos = np.average(real_stem_pos, axis=0)
    real_stem_pos = real_stem_pos * 2 / np.array([wfull, hfull]) - 1
    
    real_apple_pos = np.argwhere(real2 == 1)
    real_apple_pos += [h_adj, w_adj]
    real_apple_pos[:, [1, 0]] = real_apple_pos[:, [0, 1]]
    real_apple_pos = np.average(real_apple_pos, axis=0)
    real_apple_pos = real_apple_pos * 2 / np.array([wfull, hfull]) - 1
    
    rviz_stem_pos, rviz_apple_pos = rviz_stem_pos * Z, rviz_apple_pos * Z
    real_stem_pos, real_apple_pos = real_stem_pos * Z, real_apple_pos * Z
    
    rviz_stem_pos = np.matmul(camera_matrix, [rviz_stem_pos[0], rviz_stem_pos[1], Z])
    rviz_apple_pos = np.matmul(camera_matrix, [rviz_apple_pos[0], rviz_apple_pos[1], Z])
    real_stem_pos = np.matmul(camera_matrix, [real_stem_pos[0], real_stem_pos[1], Z])
    real_apple_pos = np.matmul(camera_matrix, [real_apple_pos[0], real_apple_pos[1], Z])
    
    return np.array((real_stem_pos[0:2], real_apple_pos[0:2])), np.array((rviz_stem_pos[0:2], rviz_apple_pos[0:2]))


def correctCOM(rviz_poses, real_poses, movement_rate):
    """  Returns new rviz positions for the apple and stem that are closer to
    their real world positions
    @param rviz_poses - pose of stem and apple in rviz
    @param real_poses - pose of the stem and apple in the real world
    @param movement_rate - float between 0 and 1 indicating how drastically to 
    change the position"""
    stem_diff, apple_diff = real_poses[0] - rviz_poses[0], real_poses[1] - rviz_poses[1]
    new_stem_rviz = rviz_poses[0] + stem_diff * movement_rate
    new_apple_rviz = rviz_poses[1] + apple_diff * movement_rate
    return [new_stem_rviz, new_apple_rviz]


def correctOrientation(rviz_poses, real_poses, listener, Z):
    """  Returns new rviz positions and orientations for the apple and stem 
    that are closer to their real world positions
    @param rviz_poses - pose of stem and apple in rviz
    @param real_poses - pose of the stem and apple in the real world
    @param listener - rospy listener that can lookup the transformation of 
    objects in rviz
    @param Z - distance from the camera to the apple/stem"""
    
    TODO fix this so that it correctly calculates the object pose and orientation
    
    stem_point = geometry_msgs.msg.PointStamped()
    apple_point = geometry_msgs.msg.PointStamped()
    tf_point = [0.5, -0.08, 0.45]
      
    new_pos = correctCOM(rviz_poses, real_poses, 0.9)
    stem_point.point.x = new_pos[0][0]
    stem_point.point.y = new_pos[0][1] # need to make sure these are the same. y and x might be negative
    stem_point.point.z = Z
    stem_point.header.frame_id = 'camera1'
    stem_point.header.stamp = rospy.Time(0)# rospy.Time.from_sec(time.time())
    apple_point.point.x = new_pos[1][0]
    apple_point.point.y = new_pos[1][1] # need to make sure these are the same. y and x might be negative
    apple_point.point.z = Z
    apple_point.header.frame_id = 'camera1'
    apple_point.header.stamp = rospy.Time(0)# rospy.Time.from_sec(time.time())
    (trans, rot) = listener.lookupTransform('/world', '/camera1', rospy.Time(0))
    world_stem_com = listener.transformPoint('world', stem_point)
    world_apple_com = listener.transformPoint('world', apple_point)
    world_stem_point = get_point(world_stem_com)
    world_apple_point = get_point(world_apple_com)
    stem_connection = np.array([0.45, 0.09, 0.36]) # fill in with the pos of the stem connector in world corods
    stem_length = 0.075 # fill in with the stem length
    stem_to_com = stem_connection - world_stem_point
    stem_to_com = stem_to_com / np.linalg.norm(stem_to_com) # normalize it
    
    stem_apple_connection = stem_connection - stem_to_com * stem_length
    apple_to_com = stem_to_com - world_stem_point
    apple_to_com = apple_to_com / np.linalg.norm(apple_to_com)
    
    apple_unrotated_axis = [0, 0, 1] # update these to match what is there in rviz
    stem_unrotated_axis = [0, 0, -1] # update these to match what is there in rviz
    
    rotation_axis_stem = np.cross(stem_to_com,stem_unrotated_axis)
    rotation_angle_stem = math.acos(np.dot(stem_to_com,stem_unrotated_axis))
    
    rotation_axis_apple = np.cross(apple_to_com,apple_unrotated_axis)
    rotation_angle_apple = math.acos(np.dot(apple_to_com,apple_unrotated_axis))
    
    # Step 4 - Obtain the quaternion from the single axis rotation
    stem_quaternion = quaternion_about_axis(rotation_angle_stem, rotation_axis_stem)
    apple_quaternion = quaternion_about_axis(rotation_angle_apple, rotation_axis_apple)
    
    # we can then send the quaternion of the stem and the quaternion of the apple to RVIZ in the msg with the coms
    return [world_stem_point, world_apple_point], [stem_quaternion, apple_quaternion]


def get_2D_stats(stem_mask, apple_mask):
    """  Finds the stem angle in the Z direction of the camera
    @param stem_mask - mask containing only the stem
    @param apple_mask - mask containing only the apple"""
    
    TODO finish this function so it accurately gets the stem angle
    
    pca = PCA(n_components=1)
    stem_points = np.argwhere(stem_mask == 1)
    pca.fit(stem_points)
    eigen_vals = pca.components_
    image_stem_vector = [eigen_vals[0, 0], eigen_vals[0, 1]]
    stem_vector = [0, 1]
    
    stem_angle = np.arccos(np.dot(stem_vector,image_stem_vector))
    print('stem angle is,', stem_angle)
    

def check_ids(mask):
    """  Separates a clustered mask into images with only one of the IDs so 
    the user can determine which ID is the stem and which is the apple
    @param mask - mask with multiple different ids"""
    num_ids = np.max(mask)
    for ID in range(num_ids+1):
        mask_copy = np.zeros(mask.shape)
        mask_copy[mask == ID] = 1
        plt.imsave('id_num' + str(ID)+".png", mask_copy)
        print("id out",ID)


def useKmeans(mask_yuv, real_yuv, rviz_params, real_params, save_fl = 0):
    """  Divides the pixels into groups based on their value and saves masks
    @param mask_yuv - rviz image in yuv
    @param real_yuv - real image in yuv
    @param rviz_params - center and std_dev from kmeans clustering for rviz im
    @param real_params - center and std_dev from kmeans clustering for real im
    @param save_fl - Bool to save images or not"""
    rviz_stem_mins = rviz_params[0][0]-2*np.sqrt(rviz_params[0][1]) - 20
    rviz_stem_maxes = rviz_params[0][0]+2*np.sqrt(rviz_params[0][1]) + 20
    rviz_apple_mins = rviz_params[1][0]-2*np.sqrt(rviz_params[1][1]) - 20
    rviz_apple_maxes = rviz_params[1][0]+2*np.sqrt(rviz_params[1][1]) + 20
    real_stem_mins = real_params[0][0]-1.8*np.sqrt(real_params[0][1]) - 10
    real_stem_maxes = real_params[0][0]+1.8*np.sqrt(real_params[0][1]) + 10
    real_apple_mins = real_params[1][0]-1.8*np.sqrt(real_params[1][1]) - 20
    real_apple_maxes = real_params[1][0]+1.8*np.sqrt(real_params[1][1]) + 20

    rviz_stem_maxes = np.reshape(rviz_stem_maxes, (1, 1, 3))
    rviz_stem_mins = np.reshape(rviz_stem_mins, (1, 1, 3))
    rviz_apple_maxes = np.reshape(rviz_apple_maxes, (1, 1, 3))
    rviz_apple_mins = np.reshape(rviz_apple_mins, (1, 1, 3))

    mask_yuv = np.array(mask_yuv)
    real_yuv = np.array(real_yuv)
    rviz_stem_mask = np.all(mask_yuv > rviz_stem_mins, axis=2) * np.all(mask_yuv < rviz_stem_maxes, axis=2) * 1
    rviz_apple_mask = np.all(mask_yuv > rviz_apple_mins, axis=2) * np.all(mask_yuv < rviz_apple_maxes, axis=2) * 2
    real_stem_mask = np.all(real_yuv > real_stem_mins, axis=2) * np.all(real_yuv < real_stem_maxes, axis=2) * 1
    real_apple_mask = np.all(real_yuv > real_apple_mins, axis=2) * np.all(real_yuv < real_apple_maxes, axis=2) * 2
    real_mask = real_apple_mask + real_stem_mask
    rviz_mask = rviz_apple_mask + rviz_stem_mask
    if save_fl:
        plt.imsave('output2/real_segmented.png', real_mask)
        plt.imsave('output2/rviz_segmented.png', rviz_mask)
    return rviz_mask, real_mask


def find_kmeans_params(rviz_image, real_image):
    """  Finds the mean and std dev from kmeans clustering for stem and apple 
    in both real and rviz images. Need user input to find correct ID
    @param rviz_image - rviz image in rgb
    @param real_image - real image in rgb"""
    mask_dia, real_dia, realimg = blur_im('real', rviz_image, real_image, save_fl=1)
    mask_yuv, real_yuv = RGB2YUV(output, mask_dia, real_dia, save_fl=1)
    real_masks = []
    rviz_masks = []
    real_params = []
    rviz_params = []
    for j in range(6):
        mask_l, real_l, temp1, temp2 = Kmeanclus(output, mask_yuv, real_yuv, j+4, save_fl=1)
        real_masks.append(real_l)
        rviz_masks.append(mask_l)
        real_params.append(temp1)
        rviz_params.append(temp2)
        print('kmean', j+4)
        print(real_params)
    a = input('which of the 6 is best for the real images?')
    a = int(a)
    check_ids(real_masks[a-4])
    stem_label = input('which id is the stem?')
    stem_label = int(stem_label)
    apple_label = input('which id is the apple?')
    apple_label = int(apple_label)
    apple_colors = []
    stem_colors = []
    for i in range(real_l.shape[0]):
        for j in range(real_l.shape[1]):
            if real_l[i, j] == apple_label:
                apple_colors.append(real_yuv[i][j])
            elif real_l[i, j] == stem_label:
                stem_colors.append(real_yuv[i][j])
    apple_var = np.var(apple_colors, axis = 0)
    stem_var = np.var(stem_colors, axis = 0)
    real_params = np.array([[real_params[a-4][stem_label], stem_var], [real_params[a-4][apple_label], apple_var]])
    np.save('real_params.npy', real_params)
    a = input('which of the 6 is best for the rviz images?')
    a = int(a)
    check_ids(rviz_masks[a-4])
    stem_label = input('which id is the stem?')
    stem_label = int(stem_label)
    apple_label = input('which id is the apple?')
    apple_label = int(apple_label)
    apple_colors = []
    stem_colors = []
    for i in range(mask_l.shape[0]):
        for j in range(mask_l.shape[1]):
            if mask_l[i, j] == apple_label:
                apple_colors.append(mask_yuv[i][j])
            elif mask_l[i, j] == stem_label:
                stem_colors.append(mask_yuv[i][j])
    apple_var = np.var(apple_colors, axis=0)
    stem_var = np.var(stem_colors, axis=0)
    rviz_params = np.array([[rviz_params[a-4][stem_label], stem_var], [rviz_params[a-4][apple_label], apple_var]])
    print('real and rviz params', real_params, rviz_params)
    np.save('rviz_params.npy', rviz_params)
    return rviz_params, real_params
    
def RViz_main(image_finder, rviz_params = None, real_params = None):
    """  Finds the position of the apple and stem in rviz and the real world
    and continuously updates the rviz position to match
    @param image_finder - image finder instance to pull images from ros
    @param rviz_params - mean and std dev of apple and stem for rviz
    @param real_params - mean and std dev of apple and stem for real"""
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('apple_proxy_experiment', anonymous=True)

    ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
    ## kinematic model and the robot's current joint states
    robot = moveit_commander.RobotCommander()

    ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
    ## for getting, setting, and updating the robot's internal understanding of the
    ## surrounding world:
    scene = moveit_commander.PlanningSceneInterface()

    group_name = "proxy"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
    ## trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    ## Getting Basic Information
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^
    # We can get the name of the reference frame for this robot:
    planning_frame = move_group.get_planning_frame()
    print("============ Planning frame: %s" % planning_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = move_group.get_end_effector_link()
    print("============ End effector link: %s" % eef_link)

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print("============ Available Planning Groups: %s", robot.get_group_names())

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print("============ Printing robot state")
    print(robot.get_current_state())
    print()

    joint_goal = move_group.get_current_joint_values()
    print('current_joint_goals', joint_goal)
    sleep_time = 5

    planning_scene_publisher = rospy.Publisher('/scene_info', moveit_msgs.msg.CollisionObject, queue_size=1)
    
    aspect_ratio = 480.0 / 640.0
    # diagonal fov is the angle between the rays of the corner of the image
    d_fov = 78
    # we use the aspect ratio and the diagonal fov to calculate the horizontal and vertical fovs
    xfov = 2 * (np.arctan(np.tan(d_fov * np.pi / 180) * np.cos(np.arctan(aspect_ratio))))
    yfov = 2 * (np.arctan(np.tan(d_fov * np.pi / 180) * np.sin(np.arctan(aspect_ratio))))
    # using these, we can find the maximum values for x and y given a maximum visible depth, zmax
    zmax = 1
    xmax = np.sin(xfov) * zmax
    ymax = np.sin(yfov) * zmax
    # we can then put these into a scale matrix. Multiplying a point in xyz space by this matrix will get us that point in [-1,1],[-1,1],[0,1]
    scale_mtx = np.array([[1 / xmax, 0, 0], [0, 1 / ymax, 0], [0, 0, 1 / zmax]])
    # we can invert this matrix to get us a matrix that undoes this operation
    unscale_mtx = np.linalg.inv(scale_mtx)
    Z = 0.55
    output = 'output2'
    
    br = tf.TransformBroadcaster()
    listener = tf.TransformListener()   
    
    # use this line when using saved exaple images
    # real_image, rviz_image = cv2.imread('with_light.jpg'), cv2.imread('rviz_with_planner.jpg')

    # use this line when using images from live cameras
    rviz_image, real_image = image_finder.get_images()

    # If we don't have pixel ranges for the stem and apple for our images, we
    # need to perform kmeans clustering once to find them
    if (real_params is None) | (rviz_params is None):
        find_kmeans_params(rviz_image, real_image)
    
    # In here we actually run the process to find the stem and apple pos
    # This should be a while loop when the code is finished
    save_flag = 1
    for i in range(10):
        # get images
        real_image, rviz_image = image_finder.get_images()
        # process images
        mask_blur, real_blur, realimg = blur_im('real', rviz_image, real_image, save_fl=save_flag)
        mask_yuv, real_yuv = RGB2YUV(output, mask_blur, real_blur, save_fl=save_flag)
        mask_l, real_l = useKmeans(mask_yuv, real_yuv, rviz_params, real_params)
        # separate images into two masks, one for stem and one for apple
        stem_grab_mask = np.copy(real_l)
        stem_grab_mask[stem_grab_mask == 2] = 0
        apple_grab_mask = np.copy(real_l)
        apple_grab_mask[apple_grab_mask == 1] = 0
        rviz_stem_grab_mask = np.copy(mask_l)
        rviz_stem_grab_mask[rviz_stem_grab_mask == 2] = 0
        rviz_apple_grab_mask = np.copy(mask_l)
        rviz_apple_grab_mask[rviz_apple_grab_mask == 1] = 0
        # segment out the foreground and background (this in theory will result
        # in smoother segmented parts of the image)
        stem_realim, stem_mj, stem_rviz = grab_cuting(output, 0, rviz_image, realimg, rviz_mask=rviz_stem_grab_mask, nwmask=stem_grab_mask, save_fl=save_flag)
        apple_realim, apple_mj, apple_rviz = grab_cuting(output, 1, rviz_image, realimg, rviz_mask=rviz_apple_grab_mask, nwmask=apple_grab_mask, save_fl=save_flag)
        # find the center of the image based on the grab_cut images
        center_real, center_rviz = findCOM(stem_rviz, stem_realim, apple_rviz, apple_realim, unscale_mtx, Z)
        # use this line to find the com base on the masks themselves
        #center_real, center_rviz = findCOM(rviz_stem_grab_mask, stem_grab_mask, rviz_apple_grab_mask, apple_grab_mask, unscale_mtx, Z)
        
        # find new pos based on com calculations (working)
        new_pos = correctCOM(center_rviz, center_real, 0.9)
        
        # find new pose and orientation based on com calculations (broken)
        #new_poses, new_orientations = correctOrientation(center_rviz, center_real, listener, Z)
        
        # send the new pose and orientation to rviz
#        br.sendTransform(new_poses[0], new_orientations[0], rospy.Time.from_sec(time.time()), 'apple_stem',
#                         'world') # check that the quaternions defined in tf are the same as those used by rviz
#        br.sendTransform(new_poses[1], new_orientations[1], rospy.Time.from_sec(time.time()), 'apple',
#                         'world') # check that the quaternions defined in tf are the same as those used by rviz
        
    

if __name__ == '__main__':
    rospy.init_node('apple_proxy_experiment', anonymous=True)
    joint_state_publisher = rospy.Publisher('/joint_states', sensor_msgs.msg.JointState, queue_size=1)
    tf_publisher = rospy.Publisher('/tf', tf2_msgs.msg.TFMessage, queue_size=1)
    image_finder_1 = image_finder()
    # Use these two lines and comment the images above if loading images
    # rviz_stuff = np.load('rviz_params.npy')
    # real_stuff = np.load('real_params.npy')
    RViz_main(image_finder_1)