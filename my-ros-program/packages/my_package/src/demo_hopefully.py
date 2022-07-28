#!/usr/bin/env python3

import cv2
import time
import psutil
import rospy
import numpy as np
from threading import Thread
import math
import os 
import yaml
from typing import Tuple, cast
from collections import namedtuple
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import WheelsCmdStamped
from dt_class_utils import DTReminder

redLine = 0

#These define a region of interest(ROI) in front of the car, these define a trapezoid which then transforms into the screen
#https://miro.medium.com/max/700/1*smzGB8f7UB-097N24uXC6g.png

#At some point this should be made to be done automaticly 
roipointdict = {
        'modeld' : np.float32([
            (187,250), #Top left
            (0,375),  #Bottom Left
            (640,350),#Bottom right  
            (472,250)  #Top Right
            ]),
        'cooldude' : np.float32([
            (215,245), #Top left
            (0,405), #Bottom Left
            (640,360),#Bottom right  
            (480,248)  #Top Right
            ]),
         'goose' : np.float32([
            (171,260), #Top left
            (0,375),  #Bottom Left
            (640,375),#Bottom right  
            (435,260)  #Top Right
        ])
       }

#Dont actually know, lmao, 
endpointdict = {
            'modeld' : np.float32([
            #modeld
            (0,0), #Top left
            (0,380),  #Bottom Left
            (640,380),#Bottom right  
            (640,0)  #Top Right
            ]),
            'cooldude' : np.float32([
            (0,0), #Top left
            (0,420), #Bottom Left
            (640,375),#Bottom right  
            (640,0)  #Top Right
            ]),
            'goose' : np.float32([
            (0,0), #Top left
            (0,390),  #Bottom Left
            (640,390),#Bottom right  
            (640,0)  #Top Right
        ])
            }


#The main class 
class VideoFilter(DTROS):

    def __init__(self, node_name):
        self.theta = 0
        self.duckie_counter = 0
        self.duckie_window = [0 for x in range(0,4)]
        super(VideoFilter,self).__init__(node_name=node_name,node_type=NodeType.PERCEPTION)
        self.bridge = CvBridge()
        self.vehicle_name = os.environ.get("VEHICLE_NAME")
        
        #Gets the image from the camera 
        self.sub_img= rospy.Subscriber(
            "/{}/camera_node/image/compressed".format(self.vehicle_name), CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )


        #Just a bunch of debug image publishers, should be renamed 
        self.heading_debug = rospy.Publisher(
            "~debug/videoFilter/heading_debug/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        self.step_seven = rospy.Publisher(
            "~debug/videoFilter/step_seven/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.step_six = rospy.Publisher(
            "~debug/videoFilter/step_six/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.step_five = rospy.Publisher(
            "~debug/videoFilter/step_five/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.step_four = rospy.Publisher(
            "~debug/videoFilter/step_four/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.step_three = rospy.Publisher(
            "~debug/videoFilter/step_three/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        
        self.step_two = rospy.Publisher(
            "~debug/videoFilter/step_two/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )
        self.step_one = rospy.Publisher(
            "~debug/videoFilter/step_one/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        self.pub_wheels_cmd = rospy.Publisher(
            "/{}/wheels_driver_node/wheels_cmd".format(self.vehicle_name), WheelsCmdStamped, queue_size=1
        )
        
        rospy.loginfo("Started")


        #This gets the camera calibrations from the calibration file
        camera_config_path = "/data/config/calibrations/camera_intrinsic/{}.yaml".format(self.vehicle_name)
        yaml_config = None
        with open(camera_config_path, 'r') as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        
        
        self.K_matrix = np.array([
            [311.3820210193674,0.0,290.1920371445156],
            [0.0,308.90984124864264,241.32164840244695],
            [0.0,0.0,1.0]
        ])
        self.D_matrix = np.array([[-0.29804495108742585,0.072523846182378,-0.0005492448455570177,-0.001519308275094275]])               
        
        #from the camera calibration this just undistorts the fisheye effect 
        if(not (yaml_config == None)):
            for x in yaml_config:
                self.K_matrix = np.array([
                    yaml_config["camera_matrix"]["data"][0:3],
                    yaml_config["camera_matrix"]["data"][3:6],
                    yaml_config["camera_matrix"]["data"][6:]
                ])
                self.D_Matrix = np.array([yaml_config["distortion_coefficients"]["data"]])
        
        self.roi_pts = roipointdict[self.vehicle_name]


    def detect_duckie(self,img):
        #set values for yellow mask
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yellow_lower = np.array([15, 100, 100])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_upper)

        #extract yellow part of image
        yellow_image = cv2.bitwise_and(img_hsv,img_hsv,mask=yellow_mask)

        #filter the image
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(yellow_image, cv2.MORPH_OPEN, kernel) # change to white mask and yellow mask
        kernel = np.ones((10,10),np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


        dsty = cv2.Canny(closing, 50, 200, None, 3)

        #find and draw contours on original image
        contours,hierarchy = cv2.findContours(dsty,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(0,255,0),2)
        if len(contours)>0:
            for i in contours:
                perimeter = cv2.arcLength(i,True)
                approximation = cv2.approxPolyDP(i,0.04*perimeter, True)
                if len(approximation)==4:
                    (x,y,w,h) = cv2.boundingRect(approximation)
                    cv2.rectangle(yellow_image,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,0),-1)

                    yellow_area = max(contours, key=cv2.contourArea)
                    (x2,y2),radius = cv2.minEnclosingCircle(yellow_area)
                    cv2.circle(yellow_image,(int(x2),int(y2)),int(radius),(0,0,0),-1)


        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(yellow_image, cv2.MORPH_OPEN, kernel) # change to white mask and yellow mask
        kernel = np.ones((10,10),np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        dsty2 = cv2.Canny(closing, 50, 200, None, 3)

        lines = cv2.HoughLinesP(dsty2,1,np.pi / 180, 80,30,10,100)
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    centerx = int((int(x2)+int(x1))/2)
                    centery = int((int(y1)+int(y2))/2)
                    line_radius = 100
                    cv2.circle(yellow_image, (centerx,centery), line_radius, (0,0,0), -1)

        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(yellow_image, cv2.MORPH_OPEN, kernel) # change to white mask and yellow mask
        kernel = np.ones((10,10),np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        dsty3 = cv2.Canny(closing, 50, 200, None, 3)

        lines = cv2.HoughLinesP(dsty3,1,np.pi / 180, 80,30,10,100)
        if lines is not None:
            (x1,y1,x2,y2) = lines[0,0]
            cv2.rectangle(yellow_image, (x1,y1), (x1+100,y1+100), (0,0,0), -1)
            cv2.rectangle(yellow_image, (x2,y2), (x2+100,y2+100), (0,0,0), -1)


        #reprocess the image
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(yellow_image, cv2.MORPH_OPEN, kernel) # change to white mask and yellow mask
        kernel = np.ones((10,10),np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        dsty4 = cv2.Canny(closing, 50, 200, None, 3)
        contours2,hierarchy = cv2.findContours(dsty4,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for i in contours2:
            perim = cv2.arcLength(i,True)
            if perim>500:
                yellow_area = max(contours2, key=cv2.contourArea)
                (x2,y2),radius = cv2.minEnclosingCircle(yellow_area)
                cv2.circle(img,(int(x2),int(y2)),int(radius),(255,0,0),2)
                #print(duckie_counter)
                #print(perim)
                if radius>60:
                    return True

    
    #The main loop, takes the image from the camera and publishes commands to the wheels 
    #This loop can be explained with this writeup we made https://docs.google.com/document/d/1WEe3QA1zJmwkpXnfjBYRxmivsU8whs6uaWPJeWYtbOA/edit?usp=sharing
    def image_cb(self, msg):
        global redLine
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return

        K = self.K_matrix
        D = self.D_matrix

        #Puts a border around the image        
        bordersize = 100
        img = cv2.copyMakeBorder(
            img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        img = cv2.resize(img, (640,480))

        # publish image
        out_msg = self.bridge.cv2_to_compressed_imgmsg(img)
        out_msg.header = msg.header
        self.step_one.publish(out_msg)
    
        



        #Undistorts the image 
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (640,480), cv2.CV_16SC2)
        img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        

        # publish image
        out_msg5 = self.bridge.cv2_to_compressed_imgmsg(img)
        out_msg5.header = msg.header
        self.step_two.publish(out_msg5)


        

        
        #Gets the region of interest
        roi_points = self.roi_pts


        #Unsure what these do
        left_pad = 0 # was 75
        right_pad = 0
        end_points = np.float32([
            (0,0), #Top left
            (0,380),  #Bottom Left
            (640,380),#Bottom right  
            (640-right_pad,0)  #Top Right
        ])
        
        #Original image size
        self.orig_image_size = img.shape[::-1][1:]
        

        #Warps the color 
        transfrom_matrix = cv2.getPerspectiveTransform(roi_points,end_points)
        warped_img = cv2.warpPerspective(img, transfrom_matrix, self.orig_image_size, flags=(cv2.INTER_LINEAR)) 

        ducc = self.detect_duckie(warped_img)
        if ducc:
            rospy.loginfo(f"Sees a duck!")  
            self.duckie_window[self.duckie_counter%4] = 1
            self.duckie_counter += 1
        else:
            rospy.loginfo(f"Does not see a duck!")
            self.duckie_window[self.duckie_counter%4] = 0
            self.duckie_counter += 1  
        rospy.loginfo(f"Duckie window: {self.duckie_window}")         
        if (self.duckie_window.count(1))>=2:
            rospy.loginfo(f"stopping for duckie")  
            wheels_cmd_msg = WheelsCmdStamped()
            wheels_cmd_msg.header.stamp = rospy.Time.now()
            wheels_cmd_msg.vel_left = 0
            wheels_cmd_msg.vel_right = 0
            self.pub_wheels_cmd.publish(wheels_cmd_msg)
            time.sleep(5)
            return
        
        out_msg = self.bridge.cv2_to_compressed_imgmsg(img)
        out_msg.header = msg.header
        self.step_three.publish(out_msg)

        #Converts to HSV image
        warped_imgHSV = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

        #The white parts 
        white_mask = cv2.inRange(warped_imgHSV, (0,0,170), (255,255 ,255))

        #The Yellow Part
        yellow_mask = cv2.inRange(warped_imgHSV, (26, 155, 238), (49, 100, 93))

        #Both white and yellow combined 
        white_yellow_mask = cv2.bitwise_or(white_mask,yellow_mask)

        white_yellow_part = cv2.bitwise_and(warped_img,warped_img,white_yellow_mask)

        gray_image = cv2.cvtColor(white_yellow_part, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5,5), 0)
        image_edges = cv2.Canny(blurred_image, 80, 80)
        lines = cv2.HoughLinesP(image_edges,1,np.pi/180,80,30,10,100)
        for i in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(warped_img,(x1,y1),(x2,y2),(0,255,0),3)
                self.theta=self.theta+math.atan2((y2-y1),(x2-x1))
                rospy.loginfo(self.theta)
                self.motor_command(self.theta)
            self.theta=0

        

        out_msg = self.bridge.cv2_to_compressed_imgmsg(warped_img)
        out_msg.header = msg.header
        self.step_four.publish(out_msg)

        out_msg = self.bridge.cv2_to_compressed_imgmsg(gray_image)
        out_msg.header = msg.header
        self.step_five.publish(out_msg)


        

        #These 4 steps remove noise
        #see https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(white_yellow_mask, cv2.MORPH_OPEN, kernel) # change to white mask and yellow mask
        kernel = np.ones((10,10),np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        #THe red masks, I don't know why theres 2
        upper_rmask = cv2.inRange(warped_imgHSV, (160,100,20), (179,255,255))
        lower_rmask = cv2.inRange(warped_imgHSV, (0,100,20), (10,255,255))
        red_mask = upper_rmask + lower_rmask 

        #Final mask?
        whi_red_yel_mask = cv2.bitwise_or(closing,red_mask) # was white_yellow_mask or red_mask
        
        #As above, removes noise 
        opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((10,10),np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        

        #"res" meaning "result"? unsure 
        res = cv2.bitwise_and(warped_img,warped_img,mask=whi_red_yel_mask)
        

        #This gets the contors for the image 
        if len(contours)>0:
            c_area = max(contours, key=cv2.contourArea)
            (x,y,w,h) = cv2.boundingRect(c_area)
            cv2.rectangle(warped_img, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(res, (x,y),(x+w,y+h),(0,255,0),2)
            # print(x,y,x+w,y+h)
            if (y+h >= 0 and y+h <240):
                redLine = 1
            if (y+h >=475 and redLine == 1):
                redLine = 0
                wheels_cmd_msg = WheelsCmdStamped()
                wheels_cmd_msg.header.stamp = rospy.Time.now()
                wheels_cmd_msg.vel_left = 0
                wheels_cmd_msg.vel_right = 0
                self.pub_wheels_cmd.publish(wheels_cmd_msg)
                time.sleep(3)
                #quit()

    def motor_command(self,angle):
        wheels_cmd_msg = WheelsCmdStamped()
        wheels_cmd_msg.header.stamp = rospy.Time.now()
        wheels_cmd_msg.vel_left = 0.1
        wheels_cmd_msg.vel_right = 0.1
        if angle>0.25:
            wheels_cmd_msg.vel_left-=0.08
            wheels_cmd_msg.vel_right+=0.08
        if angle<-0.25:
            wheels_cmd_msg.vel_left+=0.08
            wheels_cmd_msg.vel_right-=0.08
        
        rospy.loginfo(f"going L{wheels_cmd_msg.vel_left} and R{wheels_cmd_msg.vel_right}")
       
        self.pub_wheels_cmd.publish(wheels_cmd_msg)

    def on_shutdown(self):

        wheels_cmd_msg = WheelsCmdStamped()
        wheels_cmd_msg.header.stamp = rospy.Time.now()
        wheels_cmd_msg.vel_left = 0
        wheels_cmd_msg.vel_right = 0
        self.pub_wheels_cmd.publish(wheels_cmd_msg)

        super(VideoFilter, self).on_shutdown()
        self.pub_wheels_cmd.publish(wheels_cmd_msg)

if __name__ == '__main__':
    node = VideoFilter('VideoFilterNode')
    
    rospy.spin()
