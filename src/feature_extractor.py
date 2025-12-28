# -*- coding: utf-8 -*-

"""

Created on Tue Mar 14 21:14:29 2023



@author: Longfei

"""

import csv

import numpy as np

import cv2

import json

import os.path

import datetime




class FeatureExtractor():

    def __init__(self):

        self.reset()   

    

    def reset(self):

        self.one_frame_feature_now = []

        # self.one_frame_feature_names = []

        self.one_frame_feature_all = []

        self.current_bbox = []

        self.all_move_scale = []
        self.all_move_speed = []

        self.rec_flag  = 0
        

    

    # per frame features

    def get_temporal_every_frame_features(self, logger, motionDetector, poseEstimator, live=1, vid_time=None):

        mean_speed, optical_field = self.cal_optical_flow(motionDetector.recent_imgC_seqs, motionDetector.grown_mask)

        # print(mean_speed)
        
        poses = self.get_pose(poseEstimator)

        pose_person_locations_xyxy = self.get_human_bbox(poses)

        self.current_bbox = pose_person_locations_xyxy


        self.one_frame_feature_now = dict()
        self.one_frame_feature_now['f_n'] = logger.f_cnt_now
        self.one_frame_feature_now['t_stamp'] = logger.t_stamp_now
        if live == 1 and vid_time is not None:
            date_time = f"{vid_time:.3f} sec"
        else:
            now = datetime.datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.one_frame_feature_now['t_stamp_local'] = date_time
        self.one_frame_feature_now['total_secs'] = float(logger.duration_now)
        self.one_frame_feature_now['person_f'] = logger.human_flag_now
        self.one_frame_feature_now['mv_f'] = logger.move_flag_now
        self.one_frame_feature_now['mv_sp'] = float(mean_speed)
        self.one_frame_feature_now['nomv_sec'] = float(logger.nomove_sec)
        self.one_frame_feature_now['poses'] = poses

        
        this_ratio_po = self.human_recent_period_ratio(logger)

        if this_ratio_po<0.25:
          self.rec_flag = 0
          pass
        else:
          self.one_frame_feature_all.append(self.one_frame_feature_now)
          self.rec_flag = 1 




    def get_human_bbox(self, poses):  
        pose_person_locations_xyxy = []
        if len(poses[3])>0: # get box from pose first
            pose_person_locations_xyxy = self.get_all_pose_bbox(poses[3])
        return pose_person_locations_xyxy



    def get_all_pose_bbox(self, bounding_boxes):
        # Initialize variables with initial values
        xmin_min = float('inf')
        ymin_min = float('inf')
        xmax_max = float('-inf')
        ymax_max = float('-inf')
        # Iterate over the bounding boxes and update the minimum and maximum values
        for box in bounding_boxes:
            xmin, ymin, w, h = box
            xmax = xmin + w
            ymax = ymin + h
            xmin_min = min(xmin_min, xmin)
            ymin_min = min(ymin_min, ymin)
            xmax_max = max(xmax_max, xmax)
            ymax_max = max(ymax_max, ymax)
        # Calculate the overall bounding box dimensions
        return (xmin_min, ymin_min, xmax_max, ymax_max)


    def human_recent_period_ratio(self, logger):
        stop_window = 20  # seconds
        # case pose: if human or not?
        if len(logger.poses_len_all) > logger.smooth_window*stop_window:
            this_period_flag = logger.poses_len_all[-logger.smooth_window*stop_window:]
        else:
            this_period_flag = logger.poses_len_all
        try:
            this_ratio_po = sum(this_period_flag)/len(this_period_flag)
        except:
            this_ratio_po = 0
        return  this_ratio_po


    def get_pose(self, poseEstimator):
        poses_ids = []
        poses_scores = []
        poses_kpoints = []
        poses_bbox = []
        if len(poseEstimator.current_pose)>0:
            for i in range(0, len(poseEstimator.current_pose)):
                poses_ids.append(poseEstimator.current_pose[i].id)
                poses_scores.append(float(poseEstimator.current_pose[i].confidence))
                poses_kpoints.append(poseEstimator.current_pose[i].keypoints.tolist())
                poses_bbox.append(poseEstimator.current_pose[i].bbox)
        poses = [poses_ids, poses_scores, poses_kpoints, poses_bbox]
        return poses
        

        
    def append_to_json(self, output_file, big_list):
        if os.path.isfile(output_file):
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            #  Append new data to existing data
            existing_data.extend(big_list)
        else:
            existing_data = big_list
        with open(output_file, 'w') as f:
            # Write the big list to the file
            # for i in range(0, len(big_list)):
            json.dump(existing_data, f)
            # Add a new line character to separate the lists
            #   f.write('\n')
        


    def write_to_json(self, file_name, my_list):

        # Open a file for writing

        with open(file_name, 'w') as f:

            # Write the list to the file as JSON

            json.dump(my_list, f)

    

    

    def cal_optical_flow(self, recent_imgC_seqs, grown_mask):

        mean_speed = 0

        A = []

        if np.count_nonzero(grown_mask)<5:    

            return mean_speed, A

        This_imgC = recent_imgC_seqs[-1]

        Last_imgC = recent_imgC_seqs[-2]

        gray_last = cv2.cvtColor(Last_imgC, cv2.COLOR_BGR2GRAY)

        gray_this = cv2.cvtColor(This_imgC, cv2.COLOR_BGR2GRAY)

        gray_last[grown_mask==0] = 0

        gray_this[grown_mask==0] = 0

        gray_last = cv2.resize(gray_last, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        gray_this = cv2.resize(gray_this, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # calculate optical flow using Farneback's algorithm

        flow = cv2.calcOpticalFlowFarneback(gray_last, gray_this, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # convert optical flow to polar coordinates

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        A = mag

        # Find the non-zero and non-inf values

        A[np.isinf(A)] = 0

        non_zero_inf_values = A[np.logical_and(A != 0, ~np.isinf(A))]

        A = cv2.resize(A, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA)

        # Calculate the mean of the non-zero and non-inf values

        mean_speed = np.mean(non_zero_inf_values)

        if mean_speed<0.1:

            mean_speed = 0

        # # create HSV image to visualize optical flow

        # hsv = np.zeros_like(This_imgC)

        # hsv[...,0] = ang * 180 / np.pi / 2

        # hsv[...,1] = 255

        # hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # # convert HSV image to BGR for display

        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # show_image_pair(This_imgC, bgr)

        return mean_speed, A




