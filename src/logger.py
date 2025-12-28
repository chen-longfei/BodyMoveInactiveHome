# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:05:09 2023

@author: Longfei
"""

import cv2
import numpy as np
import time
import datetime
from scipy import stats

# Log: movement control, show nomove seconds
# save other logs and features


class Logger():
    def __init__(self):
        self.NO_FOREGROUND = -1.0
        self.FOREGROUND_NO_MOVE = 0.0
        self.FOREGROUND_MOVE = 1.0
        self.smooth_window = 5
        self.reset()

    def reset(self):
        # per frame
        self.f_cnt_now = []
        self.t_stamp_now = []
        self.t_stamp_start = []
        self.t_stamp_start_flag = 0
        self.t_stamp_end = []
        self.move_flag_now = []
        self.human_flag_now = []
        self.duration_now = 0

        # add once per frame
        # self.t_stamp_all = []
        self.move_flag_all = []
        # add once several frames
        self.human_flag_all = []
       # self.object_list_all = []
        self.person_recent_boxes_all = []
        self.poses_len_all = []

        # nomove sec log paras
        self.nomove_Log_cnt = 0
        self.nomove_Logs = []
        self.startNoMoveCnt_flag = 0
        self.nomove_sec = 0
        self.nomove_st = 0
        self.nomove_et = 0
        self.past_several_move_flag = []
        self.logStopCountNoHuman = 0
        self.this_ratio = 0

        self.all_nomove_sec = []

    def get_current_duration(self, live, imgC_name, vid_time=None):
        if live == 1 and vid_time is not None:
            if self.t_stamp_start_flag == 0:
                self.t_stamp_start = vid_time
        elif live == 1:
            if self.t_stamp_start_flag == 0:
                self.t_stamp_start = int(time.time()*1000)
        else:
            if self.t_stamp_start_flag == 0:
                self.t_stamp_start = imgC_name
        self.t_stamp_start_flag = 1
        if live == 1 and self.t_stamp_start_flag == 1:
            if vid_time is not None:
                self.t_stamp_end = vid_time
            else:
                self.t_stamp_end = int(time.time()*1000)
        else:
            self.t_stamp_end = imgC_name
        self.duration_now = (self.t_stamp_end - self.t_stamp_start)/1000

    def get_fcnt_now(self, fcnt):
        self.f_cnt_now = fcnt

    def get_tstampeNow(self, live, imgC_name, vid_time=None):
        if live == 1 and vid_time is not None:
            self.t_stamp_now = vid_time
        elif live == 1:
            self.t_stamp_now = int(time.time()*1000)
        else:
            self.t_stamp_now = imgC_name

    def get_moveFlagNow(self, foregroundNum, move_flag):
        if foregroundNum == 0:
            self.move_flag_now = -1
        else:
            if move_flag == 0:
                self.move_flag_now = 0
            else:
                self.move_flag_now = 1
        self.move_flag_all = np.append(self.move_flag_all, self.move_flag_now)

    def get_humanFlagNow(self, person_locations_xyxy):
        self.human_flag_now = len(person_locations_xyxy)

    def update_human_lists(self, person_exist_flag, person_locations_xyxy):
        self.human_flag_all.append(person_exist_flag)
        if sum(self.human_flag_all[-10:])==0:
            self.human_flag_all = self.human_flag_all[-2:]
        self.person_recent_boxes_all.append(person_locations_xyxy)

    def update_pose_flags(self, current_pose):
        self.poses_len_all.append(len(current_pose))

    def logUpdater(self, live, f_cnt, img_name, foregroundNum, move_flag, person_locations_xyxy, vid_time=None):
        self.get_fcnt_now(f_cnt)
        if len(img_name)>10:
            temp = int(img_name.split('.')[0])
        else:
            temp = 1
        self.get_tstampeNow(live, temp, vid_time)
        self.get_moveFlagNow(foregroundNum, move_flag)
        self.get_current_duration(live, temp, vid_time)
        self.get_humanFlagNow(person_locations_xyxy)
        # print(self.nomove_sec)

    def smooth_flag(self):
        temp = self.move_flag_all[-self.smooth_window:]
        temp1 = stats.mode(temp)[0] * np.ones((1, len(temp)))
        self.move_flag_all[-self.smooth_window:] = temp1
        if len(self.move_flag_all) > 10000:
            self.move_flag_all = self.move_flag_all[-1000:]
        if len(self.human_flag_all) > 10000:
            self.human_flag_all = self.human_flag_all[-1000:]
        # if len(self.object_list_all) > 100:
        #     self.object_list_all = self.object_list_all[-50:]
        if len(self.person_recent_boxes_all) > 20:
            self.person_recent_boxes_all = self.person_recent_boxes_all[-10:]
        if len(self.poses_len_all) > 10000:
            self.poses_len_all = self.poses_len_all[-1000:]

    def reset_nomove_counter(self):
        self.startNoMoveCnt_flag = 0
        self.nomove_st = 0
        self.nomove_et = 0
        self.nomove_sec = 0

    def logStopCountNoHumanOrNot(self):
        stop_window = 10  # seconds
        # case X: if human or not?
        # if self.nomove_sec % (self.Recording_secs)==0:
        if len(self.human_flag_all) > self.smooth_window*stop_window:
            this_period_flag = self.human_flag_all[-self.smooth_window*stop_window:]
        else:
            this_period_flag = self.human_flag_all
        try:
            this_ratio = sum(this_period_flag)/len(this_period_flag)
        except:
            this_ratio = 0
            # print('network is not detecting' + str(self.nomove_sec))
        if this_ratio > 0.3:  # case 3a  is human (do nothing)
            print("this seq human detected, trust score: " + str(this_ratio) +
                  '    now' + str(self.nomove_sec) + 's    ' + str(this_period_flag))
        else:  # case 3b is not human (stop counting, and update to BG)
            print("this seq not a human, stop count and update to backgournd, trust score: " +
                  str(this_ratio) + '    ' + str(self.nomove_sec) + '    ' + str(this_period_flag))
            self.logStopCountNoHuman = 1
            self.reset_nomove_counter()
        self.this_ratio  = this_ratio

    def get_nomove_sec(self, live, log_file_name):
        len_past = self.smooth_window*3
        if len(self.past_several_move_flag) < len_past:
            self.past_several_move_flag = self.move_flag_all[-len_past:]
        else:
            self.past_several_move_flag = np.append(
                self.past_several_move_flag, self.move_flag_all[-1])
            self.past_several_move_flag = self.past_several_move_flag[-len_past:]

        # case 1:  start count no movement
        if self.startNoMoveCnt_flag == 0 and len(self.move_flag_all) > len_past and np.sum(self.move_flag_all[-self.smooth_window:]) == self.FOREGROUND_NO_MOVE:
            # if np.mean(self.past_several_move_flag[-2*self.smooth_window:]) >= self.FOREGROUND_MOVE*0.0: # if real movement exists, start count, even small ones have been smoothed
            self.startNoMoveCnt_flag = 1
            if live == 0:
                self.nomove_st = self.t_stamp_now
            else:
                self.nomove_st = time.time()

        # case 2:   continue count
        # and pause_add == 0:
        if self.startNoMoveCnt_flag == 1 and np.mean(self.move_flag_all[-self.smooth_window:]) == self.FOREGROUND_NO_MOVE:
            # if np.mean(past_several_flag[-smooth_window:])<=FOREGROUND_NO_MOVE:  # in case of small real movement been smoothed and still counting, if only loose one frame still counting
            if live == 0:
                self.nomove_et = self.t_stamp_now
                self.nomove_sec = (self.nomove_et - self.nomove_st)/1000
            else:
                self.nomove_et = time.time()
                self.nomove_sec = (
                    int((self.nomove_et - self.nomove_st)*10)/10)

        # case 3:   no movement stopped
        if self.startNoMoveCnt_flag == 1 and np.sum(self.move_flag_all[-self.smooth_window:]) >= self.FOREGROUND_MOVE:
            # if startNoMoveCnt_flag == 1 and np.mean(move_flag_all[-smooth_window:])!= FOREGROUND_NO_MOVE:
            # if np.mean(past_several_flag[-smooth_window:])>FOREGROUND_NO_MOVE:     # there must be a real movement before disappear?
            if live == 0:
                self.nomove_et = self.t_stamp_now
                self.nomove_sec = (self.nomove_et - self.nomove_st)/1000
            else:
                self.nomove_et = time.time()
                self.nomove_sec = (
                    int((self.nomove_et - self.nomove_st)*10)/10)
                self.all_nomove_sec.append(self.nomove_sec)
            self.logWriter(log_file_name, live=live, vid_time=self.t_stamp_now)
            self.reset_nomove_counter()


    def logWriter(self, log_file_name, live=1, vid_time=None):
        if live == 1 and vid_time is not None:
            ttt = f"{vid_time:.3f} sec"
        elif live == 1 and hasattr(self, 't_stamp_now') and isinstance(self.t_stamp_now, (int, float)):
            ttt = f"{self.t_stamp_now:.3f} sec"
        else:
            ttt = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')
        wt_str = "-- no move for " + \
            str(int((self.nomove_sec*10))/10) + "s" + '  (end at ' + str(ttt) + ')'
        self.nomove_Log_cnt = self.nomove_Log_cnt + 1
        if len(self.human_flag_all) >= 2 and sum(self.human_flag_all[-2:]) > 0:
            with open(log_file_name, 'a') as f:
                Log_temp = [str(self.nomove_Log_cnt), wt_str, str(ttt)]
                f.writelines('           '.join(Log_temp))
                f.write('\n')
                f.write('\n')
