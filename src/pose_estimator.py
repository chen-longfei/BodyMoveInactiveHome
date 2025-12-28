# -*- coding: utf-8 -*-
# Adapted from:
# https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Integrated with YOLO detector
# - Added inactivity and speed estimation logic

"""

Created on Tue Mar 14 21:14:29 2023



@author: Longfei

"""

import argparse

import cv2
import numpy as np
import torch

import os

# import sys
# sys.path.append('/modules')
# sys.path.append('/models')

from third_party.lightweight_pose.models.with_mobilenet import PoseEstimationWithMobileNet
from third_party.lightweight_pose.modules.keypoints import extract_keypoints, group_keypoints
from third_party.lightweight_pose.modules.load_state import load_state
from third_party.lightweight_pose.modules.pose import Pose, track_poses
from src.utils import normalize, pad_width



class PoseEstimator():
    def __init__(self, cpu):
        self.height_size = 256; 
        self.track = 1; 
        self.smooth = 1;
            ## load the model
        self.net = PoseEstimationWithMobileNet()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine the path to the checkpoint file (assumed to be in the project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(project_root, 'third_party', 'lightweight_pose', 'checkpoint_iter_370000.pth')
        
        if not os.path.exists(checkpoint_path):
             # Fallback to relative path
             checkpoint_path = os.path.join('third_party', 'lightweight_pose', 'checkpoint_iter_370000.pth')
             
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_state(self.net, checkpoint)
            ## begin to estimate
        self.cpu = cpu; 
        self.net = self.net.eval()
        if not self.cpu:
            self.net = self.net.cuda()
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.reset()



    def get_enlarged_mask(self, frame, pose_rect, margin=20):
        """
        Create a foreground mask from a bounding box, enlarged by a margin in all directions.
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x, y, w, h = pose_rect
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        mask[y1:y2, x1:x2] = 255
        return mask


    def reset(self):
        self.previous_poses = []
        self.current_pose = []  
        self.human_bbox = None
        self.pose_img = None
        self.pose_human_mask = None
   

    def pose_detection(self, frame):
        img_pose = np.asarray( frame )
        heatmaps, pafs, scale, pad = self.infer_fast(img_pose)
        self.extractposepoints(heatmaps, pafs, pad, scale)
        self.trackpose()
        if len(self.current_pose)>0:
            frame_non = np.zeros(frame.shape)
            self.pose_img = self.showpose(frame_non)   
            self.pose_human_mask = self.get_enlarged_mask(frame, self.human_bbox, 10) 
        return self.current_pose, self.human_bbox, self.pose_human_mask, self.pose_img


    def extractposepoints(self, heatmaps, pafs, pad, scale):
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
        self.current_pose = []
        rect = None
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            self.current_pose.append(pose)

            rect = pose.bbox
            self.human_bbox = rect



    def trackpose(self):
        if self.track:
            track_poses(self.previous_poses, self.current_pose, smooth=self.smooth)
            self.previous_poses = self.current_pose


    def get_rect(self, pose, img_size):
        """Get the bounding box for a given pose."""
        min_x = np.min(pose[:, 0][pose[:, 0] != -1])
        min_y = np.min(pose[:, 1][pose[:, 1] != -1])
        max_x = np.max(pose[:, 0][pose[:, 0] != -1])
        max_y = np.max(pose[:, 1][pose[:, 1] != -1])
        
        buffer = 5
        min_x = max(0, min_x - buffer)
        min_y = max(0, min_y - buffer)
        max_x = min(img_size[1], max_x + buffer)
        max_y = min(img_size[0], max_y + buffer)
        return [int(coord) for coord in [min_x, min_y, max_x - min_x, max_y - min_y]]
    

    
    def infer_fast(self, img,
                pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        height, width, _ = img.shape
        scale = self.height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [self.height_size, max(scaled_img.shape[1], self.height_size)]
        padded_img, pad = pad_width(scaled_img, self.stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not self.cpu:
            tensor_img = tensor_img.cuda()

        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad


    def showpose(self, img):
        for pose in self.current_pose:
            pose.draw(img)
            #print(pose.keypoints)
        img = cv2.addWeighted(img, 0.6, img, 0.4, 0)
        for pose in self.current_pose:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if self.track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        return img

