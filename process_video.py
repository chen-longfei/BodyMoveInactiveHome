import cv2
import torch
import os
import glob
from src.pose_estimator import PoseEstimator
from src.motion_detector import MotionDetector
from src.feature_extractor import FeatureExtractor
from src.foreground_detector import ForegroundDetector
from src.logger import Logger
from src.visualization import Visualization
import numpy as np
import collections
from scipy.ndimage import median_filter
import datetime







def frame_generator(source, source_type, cam_index=0):
    """Yields frames from video, images, or webcam."""
    img_name = []
    if source_type == 'video':
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video file {source}")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Get current video time in seconds
            current_vid_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield frame, img_name, current_vid_time
        cap.release()

    elif source_type == 'images':
        images = sorted(glob.glob(os.path.join(source, "*.png")))
        if not images:
            print(f"No PNG files found in {source}")
            return
        for img_path in images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            frame = cv2.imread(img_path)
            if frame is not None:
                yield frame, img_name, None

    elif source_type == 'webcam':
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam (index {cam_index})")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame from webcam.")
                break
            yield frame, img_name, None
        cap.release()

    else:
        print(f"Unsupported source_type: {source_type}")
        return








# --- Configuration Files ---
def create_new(write_fd0, user_now, event_str):
    # create new
    date = datetime.datetime.now().strftime("%Y%m%d")
    now = datetime.datetime.now(); hour = now.hour; minute = now.minute; event_str = str(hour) + '.' + str(minute) + '_'
    
    # Use os.path.join for cross-platform path construction
    write_fd = os.path.join(write_fd0, date, user_now, '')
    
    isExist = os.path.exists(write_fd)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(write_fd)
    
    log_file_name = os.path.join(write_fd, f'{user_now}_no_movement_log.txt')
    datajson_file_name = os.path.join(write_fd, f'{user_now}_data.json')
    cnt = 0

    # # Define the video codec and output parameters
    # Try mp4v first, then fall back to avc1 or others if needed
    try:
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    except:
        codec = cv2.VideoWriter_fourcc(*'avc1')
        
    fps = 15.0  # Set the frames per second (FPS)
    resize_r = 0.5
    frame_width = int((640+60)*resize_r)  # Set the frame width
    frame_height = int(480*resize_r)  # Set the frame height
    # Create a VideoWriter object
    video_file_name = os.path.join(write_fd, f'{user_now}_processedVid.mp4')
    out = cv2.VideoWriter(video_file_name, codec, fps, (frame_width, frame_height))
    return log_file_name, datajson_file_name, cnt, out, write_fd, resize_r, hour







def filter_foregrounds(image_foreground_mask, pose_human_mask):
       # Calculate overlap of two foregrounds
    intersection = np.logical_and(image_foreground_mask != 0, pose_human_mask != 0)
    intersection_area = np.sum(intersection)
    area_fg = np.sum(image_foreground_mask != 0)
    area_pose = np.sum(pose_human_mask != 0)
    min_area = min(area_fg, area_pose) if min(area_fg, area_pose) > 0 else 1  # avoid division by zero
    overlap_ratio = intersection_area / min_area
    combined_mask = (image_foreground_mask != 0) | (pose_human_mask != 0)
    if overlap_ratio < 0.3:
        combined_mask[:] = 0  # set all to zero if not enough overlap
    return combined_mask





def process_source(live,input_path, source_type, cam_index, foreground_detector, poseEstimator, logger, 
               motion_detector, featureExtractor, visualization, record_data_flag, write_fd0, user_now, log_file_name, datajson_file_name, out, resize_r):
    
    
    import sys
    import glob
    maxlen = 10
    skip_n = 60
    total_frames = None
    progress_last = -1
    if source_type == 'video':
        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    elif source_type == 'images':
        # Count PNG files in the folder
        total_frames = len(glob.glob(os.path.join(input_path, "*.png")))
    prev_frames = collections.deque(maxlen=maxlen)
    for idx, (frame, img_name, current_vid_time) in enumerate(frame_generator(input_path, source_type, cam_index)):
        # Print progress for video or images
        if total_frames is not None and total_frames > 0:
            progress = int(100 * idx / total_frames)
            if progress != progress_last and progress % 5 == 0:
                sys.stdout.write(f"\rProcessing {source_type} {user_now}: {progress}% ")
                sys.stdout.flush()
                progress_last = progress
        # Now current_vid_time holds the timestamp (in seconds) for each frame if video, else None
        # Apply detection only after warm-up
        if idx >= maxlen + skip_n:
            # 1. image based pixel foreground detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_foreground_mask = foreground_detector.foregroundDetectFull3Steps(gray_frame, prev_frames)
            if np.sum(image_foreground_mask) == 0:
                # cv2.imshow('Frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                continue


            # 2.pose based foreground estimation
            current_pose, pose_human_box, pose_human_mask, pose_img = poseEstimator.pose_detection(frame)
            if pose_human_mask is None:
                # cv2.imshow('Frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                continue


            # 3. filter foregrounds
            combined_mask = filter_foregrounds(image_foreground_mask, pose_human_mask)
            th = frame.shape[0] * frame.shape[1] / 1000
            combined_mask, combined_mask_num = foreground_detector.findAreaNew(combined_mask, th)
            # Extract bounding boxes (xyxy) from combined_mask
            contours, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            person_locations_xyxy = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                person_locations_xyxy.append([x, y, x + w, y + h])
            logger.update_pose_flags(person_locations_xyxy)

            # Initialize imgC_show with the current frame
            imgC_show = pose_img.copy()

            # 4. Detect motion in the masked region (requires previous frames for optical flow)
            imgC_show = motion_detector.movementDetection(combined_mask, frame, imgC_show, prev_frames)

            # 5. smooth loggers and calculate features
            if idx%logger.smooth_window == 0:
                logger.smooth_flag()

            # update logger
            logger.logUpdater(live, idx, img_name, 
                                combined_mask_num, motion_detector.move_flag, person_locations_xyxy, vid_time=current_vid_time)
            if len(person_locations_xyxy)>0:
                person_exist_flag = 1
            else:
                person_exist_flag = 0
            logger.update_human_lists(person_exist_flag, person_locations_xyxy)
            
            if idx%int(logger.smooth_window/2) == 0:
                logger.get_nomove_sec(live, log_file_name)

            # 6. update features and write features to file
            featureExtractor.get_temporal_every_frame_features(logger, motion_detector, poseEstimator, live=live, vid_time=current_vid_time)
            # print(featureExtractor.one_frame_feature_now)
            if record_data_flag:
                if len(featureExtractor.one_frame_feature_all)>100:
                    featureExtractor.append_to_json(datajson_file_name, featureExtractor.one_frame_feature_all)
                    featureExtractor.one_frame_feature_all = []

            img_c_show1,img_c_show_double  = visualization.render_images(imgC_show, logger, featureExtractor, motion_detector, frame)

            try:
                if record_data_flag:
                    if featureExtractor.rec_flag == 0:
                        pass
                    else:
                        image = np.uint8(img_c_show1)
                        if len(image.shape) >= 2:  # Check if image has at least 2 dimensions
                            new_width = int(image.shape[1] * resize_r)
                            new_height = int(image.shape[0] * resize_r)
                            # # Resize the image
                            image = cv2.resize(image, (new_width, new_height))
                            out.write(image)
            except:
                print('error in writting')


        else:
            img_c_show1 = frame
        prev_frames.append(frame.copy())

        # cv2.imshow('Frame', img_c_show1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


    

    # cv2.destroyAllWindows()
    print(f"Finished processing: {source_type}")
    if record_data_flag:  
      out.release()
    if len(featureExtractor.one_frame_feature_all)>0:
        featureExtractor.append_to_json(datajson_file_name, featureExtractor.one_frame_feature_all)
    if total_frames is not None and total_frames > 0:
        sys.stdout.write(f"\rProcessing {source_type}: 100% \n")
        sys.stdout.flush()







import tkinter as tk
from tkinter import filedialog
from scripts.plot_mv_sp_vs_time import plot_mv_sp_vs_time

# You may need to import these if they are defined elsewhere
# from your_module import process_source, MotionDetector, FeatureExtractor, ForegroundDetector, PoseEstimator, Logger, Visualization, create_new

def main():
    # --- Select Folder ---
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Select Folder Containing MP4 Videos")

    if not folder_path:
        print("No folder selected. Exiting.")
        return

    print("Selected folder:", folder_path)

    # --- Find all .mp4 files ---
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
    if not video_files:
        print("No MP4 files found in the folder.")
        return

    print(f"Found {len(video_files)} video(s):", video_files)

    for video_file in video_files:
        input_path = os.path.join(folder_path, video_file)
        print("\n--- Processing:", input_path)

        # --- Configuration ---
        record_data_flag = 1
        user_now  = os.path.splitext(video_file)[0]
        write_fd0 = os.path.join('.', 'results')
        now = datetime.datetime.now()
        event_str = f"{now.hour}.{now.minute}_"

        if record_data_flag:
            log_file_name, datajson_file_name, cnt, out, write_fd, resize_r, hour1 = create_new(write_fd0, user_now, event_str)
        else:
            log_file_name = []
            cnt = 0
        if not os.path.exists(write_fd0):
            os.makedirs(write_fd0)

        # --- Instantiate Classes ---
        motion_detector = MotionDetector()
        feature_extractor = FeatureExtractor()
        foreground_detector = ForegroundDetector()
        # Auto-detect usage of CPU/GPU
        use_cpu = not torch.cuda.is_available()
        print(f"CUDA available: {torch.cuda.is_available()}. Using CPU: {use_cpu}")
        poseEstimator = PoseEstimator(cpu=use_cpu)
        logger = Logger()
        visualization = Visualization()

        # --- Run Processing ---
        process_source(
            live=1,
            input_path=input_path,
            source_type='video',
            cam_index=0,
            foreground_detector=foreground_detector,
            poseEstimator=poseEstimator,
            logger=logger,
            motion_detector=motion_detector,
            featureExtractor=feature_extractor,
            visualization=visualization,
            record_data_flag=record_data_flag,
            write_fd0=write_fd0,
            user_now=user_now,
            log_file_name=log_file_name,
            datajson_file_name=datajson_file_name,
            out=out,
            resize_r=resize_r
        )

        # --- Optional Plotting ---
        plot_mv_sp_vs_time(datajson_file_name)

if __name__ == "__main__":
    main()
