# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:35:05 2023

@author: Longfei
"""

import numpy as np
from scipy.ndimage import median_filter
import cv2

class MotionDetector():
    def __init__(self):
        self.reset()   
    
    def reset(self):
        # last_c_img
        self.img_hei = 480   # default can change
        self.img_wid = 640   # default can change
        self.recentMoveMaskNo = 5
        self.recent_move_mask_seqs = list()
        self.recent_move_mask_seqs.append(np.zeros((self.img_hei, self.img_wid, 1)))
        self.move_flag = 0
        self.grown_mask = np.zeros((self.img_hei, self.img_wid, 1))
        self.recent_imgC_seqs  = []
        
    # def shift_mask(self, mask, para):
    #     non_zeros_eles = np.argwhere(mask)
    #     m,n = mask.shape
    #     II = non_zeros_eles[:,0]
    #     JJ = non_zeros_eles[:,1]
    #     JJ = JJ + para
    #     II = II[JJ<n]
    #     JJ = JJ[JJ<n]
    #     mask[II,JJ] = 1
    #     return mask
    
    
    # def findAreaNew(self, img, min_size):
    #     img = img.astype(np.uint8)
    #     #find all your connected components (white blobs in your image)
    #     nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #     #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #     #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    #     sizes = stats[1:, -1]; nb_components = nb_components - 1
    #     # minimum size of particles we want to keep (number of pixels)
    #     #where, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    #         #your answer imageimg_c_show
    #     num_valid = 0
    #     new_all_labels = np.zeros((output.shape))
    #     #for every component in the image, you keep it only if it's above min_size
    #     for i in range(0, nb_components):
    #         if sizes[i] >= min_size:
    #             new_all_labels[output == i + 1] = 255
    #             num_valid = num_valid + 1
    #     return new_all_labels, num_valid
    
    
    def cal_movement(self, mask_foreground_grown, imgC, recent_imgC_seqs):
      # 5 caculate movement
     #  start_t =  time.time()
      valid_region = mask_foreground_grown.copy()
      # if len(self.Last_imgC)==0:
      #     self.Last_imgC = np.zeros(imgC.shape)
      Last_imgC = recent_imgC_seqs[-1]
      diff_color = np.int16(imgC) - np.int16(Last_imgC)
      move_region = self.filter_small_color(diff_color)
      move_region[valid_region==0]=0
      move_region = median_filter(move_region, size=5)
      # move_region, move_num = self.findAreaNew(move_region, min_size=5)
      move_region[move_region!=0]=1
      return move_region
  
   # 5 caculate movement
    def filter_small_color(self, color_diff_img):
         th = 20;
         m,n,d = color_diff_img.shape
         mask = np.ones((m,n))
         r_diff = color_diff_img[:,:,0]
         mask[r_diff<th]=0
         g_diff = color_diff_img[:,:,1]
         mask[g_diff<th]=0
         b_diff = color_diff_img[:,:,2]
         mask[b_diff<th]=0
         return mask
     
    def renderMovement(self, img_c_show, move_region, mask_foreground_grown):
        # img_c_show[mask_foreground_grown!=0] = 255  
        if len(img_c_show.shape)>2:
            B = img_c_show[:,:,0]
            G = img_c_show[:,:,1]
            R = img_c_show[:,:,2]
            R[move_region==1] = 255
            img_c_show[:,:,0] = B
            img_c_show[:,:,1] = G
            img_c_show[:,:,2] = R 
        else:
            img_c_show0 = np.zeros((img_c_show.shape[0], img_c_show.shape[1], 3))
            B = img_c_show.copy()
            G = img_c_show.copy()
            R = img_c_show.copy()
            R[move_region==1] = 255
            img_c_show0[:,:,0] = B
            img_c_show0[:,:,1] = G
            img_c_show0[:,:,2] = R 
            img_c_show = img_c_show0.copy()
     # ttt = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S') 
     # img_c_show = cv2.putText(img_c_show, str(i) + ' ' + ttt, (25, 35), font, 1, (0, 0, 0), 2,)
     # img_c_show = cv2.putText(img_c_show, str(cnt_i), (25, 35), font, 1, (0, 0, 0), 2,)
        return img_c_show
     
    def update_recent_fore(self, move_mask):
        if len(self.recent_move_mask_seqs)<self.recentMoveMaskNo:
            self.recent_move_mask_seqs.append(move_mask)    
        else:
            self.recent_move_mask_seqs.pop(0)
            self.recent_move_mask_seqs.append(move_mask)
            
    
   # - 4. shift the depth foreground and find human movement by color frames
    def movementDetection(self, detected_person_mask, imgC, img_c_show, recent_imgC_seqs):
        self.recent_imgC_seqs = recent_imgC_seqs
        # mask_foreground_grown = self.shift_mask(detected_person_mask, 35)
        mask_foreground_grown = detected_person_mask.copy()
        self.grown_mask = mask_foreground_grown
        self.This_imgC = imgC
        move_region = self.cal_movement(mask_foreground_grown, imgC, recent_imgC_seqs)
        if np.count_nonzero(move_region)>5:
            self.move_flag = 1
        else: 
            self.move_flag = 0
        self.update_recent_fore(move_region)
     # render movement
        # img_c_show = np.zeros(imgC.shape)
        img_c_show = self.renderMovement(img_c_show, move_region, mask_foreground_grown)
        return img_c_show