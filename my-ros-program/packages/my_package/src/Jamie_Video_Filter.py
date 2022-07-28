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

#These are the hardcoded HSV ranges for yellow I belive 
low_H = 0
low_S = 12
low_V = 124
high_H = 47
high_S = 255
high_V = 255


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
        super(VideoFilter,self).__init__(node_name=node_name,node_type=NodeType.PERCEPTION)
        self.bridge = CvBridge()
        self.vehicle_name = os.environ.get("VEHICLE_NAME")

        self.veh_name = rospy.get_namespace().strip("/")

        # Set parameters using a robot-specific yaml file if such exists
        self.readParamFromFile()

        # Get static parameters
        self._baseline = rospy.get_param('~baseline')
        self._radius = rospy.get_param('~radius')
        self._k = rospy.get_param('~k')
        # Get editable parameters
        self._gain = DTParam(
            '~gain',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=3.0
        )
        self._trim = DTParam(
            '~trim',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=3.0
        )
        self._limit = DTParam(
            '~limit',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=1.0
        )

        # Wait for the automatic gain control
        # of the camera to settle, before we stop it
        rospy.sleep(2.0)
        rospy.set_param(f"/{self.veh_name}/camera_node/exposure_mode",'off')

        self.log("Initialized")
        
        #Gets the image from the camera 
        self.sub_img= rospy.Subscriber(
            "/{}/camera_node/image/compressed".format(self.vehicle_name), CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )


        #Just a bunch of debug image publishers, should be renamed 
        self.heading_debug = rospy.Publisher(
            "~debug/videoFilter/heading_debug/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
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
        

        
    #The main loop, takes the image from the camera and publishes commands to the wheels a
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


        out_msg4 = self.bridge.cv2_to_compressed_imgmsg(res)
        out_msg4.header = msg.header
        # publish image
        self.step_four.publish(out_msg4)
        

        out_msg = self.bridge.cv2_to_compressed_imgmsg(warped_img)
        out_msg.header = msg.header
        # publish image
        self.step_five.publish(out_msg)       
        
        #min_theta = 3.14/3,max_theta = (2*3.14)/3
        dstw = cv2.Canny(white_mask, 50, 200, None, 3)
        dsty = cv2.Canny(yellow_mask, 50, 200, None, 3)
        



        wlines = cv2.HoughLines(dstw, 1, np.pi / 180, 50, 10, 0, 0,min_theta = -np.pi/3,  max_theta = np.pi/3)
        ylines = cv2.HoughLines(dsty, 1, np.pi / 180, 50, 10, 0, 0,min_theta = -np.pi/3,  max_theta = np.pi/3)
        white_yellow_mask = cv2.cvtColor(white_yellow_mask, cv2.COLOR_GRAY2BGR)


        out_msg2 = self.bridge.cv2_to_compressed_imgmsg(white_yellow_mask)
        out_msg.header = msg.header
        # publish image
        self.step_six.publish(out_msg2)
        
        tot_avg_theta = 0
        num_at = 0
        tot_avg_rho = 0
        num_ar = 0
        #white lines
        avg_theta = 0
        avg_rho = 0
        # Draw the lines
        if wlines is not None:
            for i in range(0, len(wlines)):
                rho = wlines[i][0][0]
                theta = wlines[i][0][1]
                #theta = math.radians(math.degrees(theta) % 180)
                avg_rho += rho
                avg_theta += theta
            avg_theta /= (len(wlines))
            avg_rho /= (len(wlines))
            # avg_theta = avg_theta * (3/4)
            tot_avg_theta += avg_theta
            num_at +=1
            tot_avg_rho += avg_rho
            num_ar +=1
            a = math.cos(avg_theta)
            b = math.sin(avg_theta)
            x0 = a * avg_rho
            y0 = b * avg_rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(white_yellow_mask, pt1, pt2, (int(255*((math.cos(avg_theta)+1)/2)),0,255), 3, cv2.LINE_AA)   
        
        #yellow lines
        avg_theta = 0
        avg_rho = 0
        # Draw the lines
        if ylines is not None:
            for i in range(0, len(ylines)):
                rho = ylines[i][0][0]
                theta = ylines[i][0][1]
                #theta = math.radians(math.degrees(theta) % 180)
                avg_rho += rho
                avg_theta += theta
            avg_theta /= (len(ylines))
            avg_rho /= (len(ylines))
            # avg_theta = avg_theta * (3/4)
            tot_avg_theta += avg_theta
            num_at +=1
            tot_avg_rho += avg_rho
            num_ar +=1
            
            a = math.cos(avg_theta)
            b = math.sin(avg_theta)
            x0 = a * avg_rho
            y0 = b * avg_rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(white_yellow_mask, pt1, pt2, (int(255*((math.cos(avg_theta)+1)/2)),0,255), 3, cv2.LINE_AA)
        if(num_at !=0):
            tot_avg_theta /= num_at
            
            
        # makes green line**
        
        self.motor_command(tot_avg_theta)
        cv2.putText(white_yellow_mask,"{} average theta {}".format(2,tot_avg_theta),(30,30),0,1,(0,255,255))
        a = math.cos(tot_avg_theta)
        b = math.sin(tot_avg_theta)
        x0 = a * tot_avg_rho
        y0 = b * tot_avg_rho
        pt1 = (int(x0 + 20*(-b)), int(y0 + 20*(a)))
        pt2 = (int(x0 - 20*(-b)), int(y0 - 20*(a)))
        cv2.line(white_yellow_mask, (640//2,480//2), pt2, (0,255,0), 3, cv2.LINE_AA)  
        
        gain = .2
        shift = np.pi/2
        k1 = -np.pi/4
        k2 = (5*np.pi)/4       
        cv2.putText(white_yellow_mask,"\nleft: {:.2f} \nright:{:.2f}".format(gain*math.cos(tot_avg_theta+shift+k2),gain*math.cos(tot_avg_theta+shift+k1)),(30,70),0,1,(0,255,255))
        out_msg = self.bridge.cv2_to_compressed_imgmsg(white_yellow_mask)
        out_msg.header = msg.header
        # publish image
        self.heading_debug.publish(out_msg)
        
        

        
    def motor_command(self,theta):
        wheels_cmd_msg = WheelsCmdStamped()
        wheels_cmd_msg.header.stamp = rospy.Time.now()
        gain = .2
        shift = np.pi/2
        k1 = -np.pi/4
        k2 = (5*np.pi)/4
        
        wheels_cmd_msg.vel_left = gain*math.cos(theta+shift+k2) 
        wheels_cmd_msg.vel_right = gain*math.cos(theta+shift+k1)
        rospy.loginfo(f"going L{wheels_cmd_msg.vel_left} and R{wheels_cmd_msg.vel_right}")
       
        self.pub_wheels_cmd.publish(wheels_cmd_msg)
        
    def speedToCmd(self, speed_l, speed_r):
        """Applies the robot-specific gain and trim to the
        output velocities

        Applies the motor constant k to convert the deisred wheel speeds
        to wheel commands. Additionally, applies the gain and trim from
        the robot-specific kinematics configuration.

        Args:
            speed_l (:obj:`float`): Desired speed for the left
                wheel (e.g between 0 and 1)
            speed_r (:obj:`float`): Desired speed for the right
                wheel (e.g between 0 and 1)

        Returns:
            The respective left and right wheel commands that need to be
                packed in a `WheelsCmdStamped` message

        """

        # assuming same motor constants k for both motors
        k_r = self._k
        k_l = self._k

        # adjusting k by gain and trim
        k_r_inv = (self._gain.value + self._trim.value) / k_r
        k_l_inv = (self._gain.value - self._trim.value) / k_l

        # conversion from motor rotation rate to duty cycle
        u_r = speed_r * k_r_inv
        u_l = speed_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = self.trim(u_r,
                                -self._limit.value,
                                self._limit.value)
        u_l_limited = self.trim(u_l,
                                -self._limit.value,
                                self._limit.value)

        return u_l_limited, u_r_limited

    def readParamFromFile(self):
        """
        Reads the saved parameters from
        `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml` or
        uses the default values if the file doesn't exist. Adjsuts
        the ROS paramaters for the node with the new values.

        """
        # Check file existence
        fname = self.getFilePath(self.veh_name)
        # Use the default values from the config folder if a
        # robot-specific file does not exist.
        if not os.path.isfile(fname):
            self.log("Kinematics calibration file %s does not "
                     "exist! Using the default file." % fname, type='warn')
            fname = self.getFilePath('default')

        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

        # Set parameters using value in yaml file
        if yaml_dict is None:
            # Empty yaml file
            return
        for param_name in ["gain", "trim", "baseline", "k", "radius", "limit"]:
            param_value = yaml_dict.get(param_name)
            if param_name is not None:
                rospy.set_param("~"+param_name, param_value)
            else:
                # Skip if not defined, use default value instead.
                pass

    def getFilePath(self, name):
        """
        Returns the path to the robot-specific configuration file,
        i.e. `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`.

        Args:
            name (:obj:`str`): the Duckiebot name

        Returns:
            :obj:`str`: the full path to the robot-specific
                calibration file

        """
        cali_file_folder = '/data/config/calibrations/kinematics/'
        cali_file = cali_file_folder + name + ".yaml"
        return cali_file

    def trim(self, value, low, high):
        """
        Trims a value to be between some bounds.

        Args:
            value: the value to be trimmed
            low: the minimum bound
            high: the maximum bound

        Returns:
            the trimmed value
        """

        return max(min(value, high), low)

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
