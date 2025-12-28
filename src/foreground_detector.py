# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:16:36 2023

@author: Longfei
"""

from scipy.ndimage import median_filter
import cv2
import numpy as np


class ForegroundDetector:
    def __init__(self):
        self.reset()   
    
    def reset(self):
        # background
        self.BG_seqs = []
        self.shot_BGM_g = []
        # recent fore frames
        self.recentForegroundNo = 5
        self.recent_fore_mask_seqs = list()
        # last_c_img
        self.Last_imgC = []
        self.img_hei = 0   # default can change
        self.img_wid = 0   # default can change
        self.foregroundNum = 0
 


    def buildGrayBackground(self, prev_frames):
        """
        Build a background model from a list of grayscale frames.
        prev_frames: list of 2D numpy arrays (grayscale images)
        Returns: median-filtered average background image (float32)
        """
        if not prev_frames:
            raise ValueError("No frames provided for background modeling.")

        # Stack frames and compute the average
        stacked = np.stack(prev_frames, axis=0).astype(np.float32)
        avg_background = np.mean(stacked, axis=0)
        # Apply median filter for smoothing
        avg_background = median_filter(avg_background, size=3)
        return avg_background
                

                
    def readBackground(self, gray_prev_frames):
        self.BG_seqs = gray_prev_frames
        self.img_hei = gray_prev_frames[-1].shape[0]   # default can change
        self.img_wid = gray_prev_frames[-1].shape[1]   # default can change
        grayBackground = self.buildGrayBackground(gray_prev_frames)
        self.shot_BGM_g = grayBackground



    def update_shortterm_bgm(self, img_g, grown_mask):
        para = 0.01
        m,n = grown_mask.shape
        enlarge_size = m/10
        grown_mask = self.make_enlarge_box(grown_mask, enlarge_size)
        grown_mask[grown_mask!=0]=1
        updated_region = grown_mask==0
        self.shot_BGM_g = self.shot_BGM_g*grown_mask + self.shot_BGM_g*updated_region*(1-para) + img_g*updated_region*para;
        
    def update_BG(self):
        self.BG_seqs.pop(0)
        self.BG_seqs.append([self.shot_BGM_g])
        
    def update_recent_fore(self, org_mask):
        if len(self.recent_fore_mask_seqs)<self.recentForegroundNo:
            self.recent_fore_mask_seqs.append(org_mask)    
        else:
            self.recent_fore_mask_seqs.pop(0)
            self.recent_fore_mask_seqs.append(org_mask)
    
    def backgroundUpdate(self, img_g, org_mask):  
      # - 5. update background, and recent foreground list  
      self.update_shortterm_bgm(img_g, org_mask)
      self.update_BG()
      self.update_recent_fore(org_mask)
      

    ######################################## foreground functions ########################################  
    def estimateBGGrayErrorMean(self):
        m, n = self.BG_seqs[0].shape
        diff = np.zeros((m, n), dtype=np.float32)
        for i in range(1, len(self.BG_seqs)):
            g_img = self.BG_seqs[i]
            g_img_prev = self.BG_seqs[i-1]
            diff_temp = np.abs(np.float32(g_img) - np.float32(g_img_prev))
            diff += diff_temp
        M = diff / (len(self.BG_seqs) - 1)
        M = np.mean(M)
        return M
    
    
    def density_estimation_matrix_new(self, matrix_diff, M):
        pi = np.pi
        sigma = M/0.68/np.sqrt(2)
        A = 1/np.sqrt(2*pi*pow(sigma,2)) 
        B1 = -0.5/sigma/sigma
        tempB = np.int32(matrix_diff) 
        B2 = tempB * tempB
        B = B1 * B2
        pr_matrix_log = np.log(A) +  B
        return pr_matrix_log
    
    def make_enlarge_box(self, small_mask, enlarge_size):
       m,n = small_mask.shape
       enlarged_mask = np.zeros((m,n))
       non_zeros_eles = np.argwhere(small_mask)
       box_coordin = [0,0,0,0] 
       if non_zeros_eles.shape[0]!=0:   
           II = non_zeros_eles[:,0]
           JJ = non_zeros_eles[:,1]
           min_x = np.min(II)
           max_x = np.max(II)
           min_y = np.min(JJ)
           max_y = np.max(JJ)
           box_coordin = [max(min_x-enlarge_size,0),min(max_x+enlarge_size,m-1),max(min_y-enlarge_size,0),min(max_y+enlarge_size,n-1)]
           enlarged_mask[int(box_coordin[0]):int(box_coordin[1]), int(box_coordin[2]):int(box_coordin[3])] = 1
       return enlarged_mask
     
    def constrainSearchArea(self, enlarge_size):
       previous_foreground_mask_enlarge = np.ones((self.img_hei, self.img_wid))
       if len(self.recent_fore_mask_seqs)>=2 and self.recent_fore_mask_seqs[-1].sum()!= 0 and self.recent_fore_mask_seqs[-2].sum()!=0:
           temp1 = self.recent_fore_mask_seqs[-1]; temp1[temp1!=0] = 1; 
           temp2 = self.recent_fore_mask_seqs[-2]; temp2[temp2!=0] = 1; 
           combined_mask = np.zeros_like(temp2)
            # Use bitwise OR operation to combine the masks
           combined_mask = cv2.bitwise_or(combined_mask, temp1)
           combined_mask = cv2.bitwise_or(combined_mask, temp2)
           previous_foreground_mask_enlarge = self.make_enlarge_box(combined_mask, enlarge_size)
       return previous_foreground_mask_enlarge
     
    def findAreaNew(self, img, min_size):
        img = img.astype(np.uint8)
        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        # minimum size of particles we want to keep (number of pixels)
        #where, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
            #your answer image
        num_valid = 0
        new_all_labels = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                new_all_labels[output == i + 1] = 255
                num_valid = num_valid + 1
        return new_all_labels, num_valid
    

    


    def foregroundDetection(self, g_img_detection, M, kernel20):
        m, n = g_img_detection.shape
        mask_foreground = np.zeros((m, n), dtype=np.uint8)
        w_frames_cnt = 0
        p_matrix = np.zeros((m, n), dtype=np.float32)
        non_parametric_method_pro_threshold = np.log(0.1 / 100)
        for i in range(len(self.BG_seqs)):
            bgNow = np.int16(self.BG_seqs[i])
            w_frames_diff_temp = np.int16(g_img_detection) - bgNow
            w_frames_cnt += 1
            pr_matrix = self.density_estimation_matrix_new(w_frames_diff_temp, M)
            pr_matrix[g_img_detection == 0] = 100
            pr_matrix[bgNow == 0] = 100
            p_matrix += pr_matrix
        p_matrix /= w_frames_cnt
        mask_foreground[p_matrix < non_parametric_method_pro_threshold] = 1
        gray_noise_error = 3
        mask_foreground[np.abs(w_frames_diff_temp) < gray_noise_error] = 0
        th = m * n / 1000
        mask_foreground, mask_num = self.findAreaNew(mask_foreground, th)
        mask_foreground = cv2.morphologyEx(mask_foreground, cv2.MORPH_CLOSE, kernel20)

        return mask_foreground, mask_num
    



    ######################################## foreground functions ########################################
    
    
    def foregroundDetectFull3Steps(self, imgGray, prev_frames):
        kernel20 = np.ones((20,20),np.uint8)
        # input: background sequences
        # input: current depth image
        # output:  foreground region mask of human
            # image based pixel foreground detection
        gray_prev_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in prev_frames]
        self.readBackground(gray_prev_frames)
 
        g_img_detection1_enlarge = imgGray.copy()
        ## preprocess the depth image
        # - 1. constrain foreground search area
        # input: previous foregrounds, enlarge_box_size
        # output: enlarged foreground search mask
        enlarge_size = int(self.img_hei/2)
        previous_foreground_mask_enlarge = self.constrainSearchArea(enlarge_size)
        g_img_detection1_enlarge[previous_foreground_mask_enlarge==0]=0
        # self.show_image(previous_foreground_mask_enlarge)
        
        # - 2. detect foreground region
        #  input: depth image after 1, backgrouds
        #  output: foreground mask
        M = self.estimateBGGrayErrorMean()
        mask_foreground, self.foregroundNum = self.foregroundDetection(g_img_detection1_enlarge, M, kernel20)
        # self.show_image(mask_foreground)
        
        # -4.update bacgrounds
        self.backgroundUpdate(imgGray, mask_foreground)
        
        return mask_foreground


    def show_image_pair(self, imgC, imgG):
        cv2.namedWindow('captured', cv2.WINDOW_AUTOSIZE)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(imgG, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((imgC, imgG))
        cv2.imshow('captured', images)
        cv2.waitKey(1)
        
    def show_image(self, img):
        cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('img', img)
        cv2.waitKey(1)     


