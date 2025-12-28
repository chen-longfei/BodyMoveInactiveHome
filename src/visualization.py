import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

class Visualization:
    def __init__(self):
        """Initialize the Visualization class"""
        pass

    def bar_plot(self, some_list, color_i):
        """Create a bar plot from a list of data"""
        data = np.array(some_list)
        num_intervals = 20
        data_min = min(data)
        data_max = max(data)
        # Create intervals
        intervals = np.linspace(data_min, data_max, num_intervals + 1)
        # Count the occurrences in each interval
        hist, _ = np.histogram(data, bins=intervals)
        # Plot the histogram as a bar plot
        plt.bar(range(num_intervals), hist, color=color_i)

    def plot_all(self, all_nomove_sec, all_move_speed, name):
        """Create a comprehensive plot with three subplots"""
        plt.figure(figsize=(15,5))
        
        plt.subplot(1,3,1) 
        color_i = 'red'
        self.bar_plot(all_nomove_sec, color_i)
        plt.xlabel('Seconds')
        plt.ylabel('Occurrences')
        plt.title('Inactivity')


        plt.subplot(1,3,3) 
        color_i = 'yellow'
        plt.boxplot(all_move_speed, boxprops=dict(color=color_i, linewidth=4.0), )
        plt.ylabel('pix/f')
        plt.title('Movement Velocity')
        # plt.show()
        plt.savefig(name +'.png', bbox_inches='tight')

    def show_image_pair(self, imgC, imgD):
        """Display a pair of images side by side"""
        cv2.namedWindow('captured', cv2.WINDOW_AUTOSIZE)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(imgD, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = imgD.copy()
        images = np.hstack((imgC, depth_colormap))
        cv2.imshow('captured', images)
        cv2.waitKey(1)
        

    def show_image(self, img):
        """Display a single image"""
        cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('img', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('img',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        m,n,d = img.shape
        resized_up = cv2.resize(img, (n,m), interpolation= cv2.INTER_LINEAR)
        cv2.imshow('img', resized_up)
        cv2.waitKey(1)     

    def draw_clock(self, img, position):
        """Draw a clock on the image at the specified position"""
        # Simple clock drawing - you can enhance this as needed
        x, y = position
        cv2.circle(img, (x, y), 15, (255, 255, 255), 2)
        # Draw clock hands (simplified)
        cv2.line(img, (x, y), (x, y-10), (255, 255, 255), 2)
        cv2.line(img, (x, y), (x+8, y), (255, 255, 255), 2)
        return img

    def render_images(self, img_c_show, logger, featureExtractor, motionDetector, imgC):
        """Render images with various overlays and information"""
        # if len(featureExtractor.current_bbox)>0:
        #     (x1,y1,x2,y2) = featureExtractor.current_bbox
        #     img_c_show = cv2.rectangle(img_c_show, (x1,y1), (x2,y2), color, 2) 
        # plot show images
        # add background by luma
        # if weatherDetector.Luma!=0:
        #     temp = (weatherDetector.Luma/100.0 - 1.0)
        #     # print('luma:',temp)
        #     img_c_show[img_c_show==0] =  max(0, temp)
        #     # add objects and poses                
        # img_c_show = objectDetector.show_detection_img(img_c_show)   
        # img_c_show = poseEstimator.showpose(img_c_show)
        # add text of nomove seconds to show_img

        img_c_show_no_move_only = img_c_show.copy()
        font = cv2.FONT_HERSHEY_DUPLEX  
        if logger.nomove_sec>0:
            [m,n,d] = img_c_show.shape
            img_c_show = cv2.putText(img_c_show, "no move " + str(logger.nomove_sec) + 's', (m-100, n-200), font, 1, (250, 250, 250), 3,)
            img_c_show_no_move_only = cv2.putText(img_c_show_no_move_only, "no move " + str(logger.nomove_sec) + 's', (m-100, n-200), font, 1, (250, 250, 250), 3,)
        
        try:
            # add speed bars
            # Create a black background for the bar plot
            plot = np.zeros((img_c_show.shape[0], 30, 3), dtype=np.uint8)
            # Calculate the height of the bar based on the value
            height = int(featureExtractor.one_frame_feature_now['mv_sp']/2*motionDetector.move_flag * plot.shape[0])
            # Draw the bar
            cv2.rectangle(plot, (0, plot.shape[0]-height), (plot.shape[1], plot.shape[0]), (0, 255, 255), -1)
            text = 'vel' # example text
            cv2.putText(plot, text, (plot.shape[1]-20, plot.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1, cv2.LINE_AA)
            # Add the bar plot to the right side of the image
            img_c_show1 = np.hstack((img_c_show, plot))
            # add size bars
            # Create a black background for the bar plot

            # add inactivity bars
            # Create a black background for the bar plot
            plot = np.zeros((img_c_show1.shape[0], 30, 3), dtype=np.uint8)
            # Calculate the height of the bar based on the value
            height = int(logger.nomove_sec/20 * plot.shape[0])
            # Draw the bar
            cv2.rectangle(plot, (0, plot.shape[0]-height), (plot.shape[1], plot.shape[0]), (0, 0, 255), -1)
            text = 'ina' # example text
            cv2.putText(plot, text, (plot.shape[1]-20, plot.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Add the bar plot to the right side of the image
            img_c_show1 = np.hstack((img_c_show1, plot))

            # img_c_show1 = cv2.putText(img_c_show1,  f'{datetime.datetime.now():%Y%m%d_%H%M}', (1, 20), font, 0.5, (250, 250, 250), 1,)
            # img_c_show1 = self.draw_clock(img_c_show1, (30,30))
            # img_c_show1 = cv2.putText(img_c_show1, 'Lighting ', (70, 30), font, 0.5, (250, 250, 250), 1,)
            # # Define the minimum and maximum values
            # # Calculate the width of the bar based on the current value
            # cv2.rectangle(img_c_show1, (140, 15), (145, 37), (255, 255, 255), 1)
            # # cv2.rectangle(img_c_show1, (140, 15 + int(max(150-weatherDetector.Luma,0)/100*(37-15))), (145, 37), (255, 255, 255), -1)
            # # img_c_show1 = cv2.putText(img_c_show1,  weatherDetector.Type , (170,30), font, 0.6, (250, 250, 250), 1,)
            # # print(event_str)
            img_c_show1 = cv2.putText(img_c_show1, datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S"), (250, 30), font, 0.5, (250, 250, 250), 1,)
            # print('weather:' + weatherDetector.Type), 0.5, (250, 250, 250), 1,)
        except:
            img_c_show1 = imgC.copy()
        # img_c_show_double =  np.hstack((np.uint8(imgC), np.uint8(img_c_show_no_move_only)))
        img_c_show_double =  np.hstack((np.uint8(imgC), np.uint8(img_c_show1)))
        return img_c_show1, img_c_show_double
