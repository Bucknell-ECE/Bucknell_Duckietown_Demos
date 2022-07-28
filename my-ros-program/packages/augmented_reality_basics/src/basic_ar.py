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
from sensor_msgs.msg import Joy
class MyPublisherNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyPublisherNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.bridge = CvBridge()
        self.script_dir = os.path.dirname(__file__)

        self.vehicle_name = rospy.get_param("/veh_name")
        rospy.loginfo(str(self.vehicle_name))
        self.map_name = rospy.get_param("/map_name")
        self.map_path = f"{self.script_dir}/maps/{self.map_name}.yaml"
        rospy.loginfo(str(self.map_name))

        self.map = self.readYamlFile(self.map_path)
        self.homo_matrix = self.load_extrinsics()


        self.sub_img= rospy.Subscriber(
            "/{}/camera_node/image/compressed".format(self.vehicle_name), CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )
        self.pub_img_map = rospy.Publisher(
            f"/{self.vehicle_name}/{node_name}/{self.map_name}/image/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

    def readYamlFile(self,fname):
        """
        Reads the YAML file in the path specified by 'fname'.
        E.G. :
            the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                        %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def image_cb(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return
        map_image = self.render_segments(img)
        img_out = self.bridge.cv2_to_compressed_imgmsg(map_image)
        self.pub_img_map.publish(img_out)
        # img_out = self.bridge.cv2_to_compressed_imgmsg(img)
        # self.pub_img_map.publish(img_out)

    # def process_image(self, img):
    #     map3, map4 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (640,480), cv2.CV_16SC2)
    #     img2 = cv2.remap(img, map3, map4, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #     return img


    def ground2pixel(self, coord):
        x_w = np.array([coord[0],coord[1], 1])
        x_i = np.matmult(self.homo_matrix, x_w)
        return x_i
    

    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        # load intrinsic calibration
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + self.vehicle_name.strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log(
                f"Can't find calibration file: {cali_file}.\n Using default calibration instead.", "warn"
            )
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = "Found no calibration file ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file, "r") as stream:
                calib_data = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError:
            msg = f"Error in parsing calibration file {cali_file} ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        return calib_data["homography"]

# # def load_intrinsics(self):
# #         # load intrinsic calibration
# #         cali_file_folder = "/data/config/calibrations/camera_intrinsics/"
# #         cali_file = cali_file_folder + self.vehicle_name.strip("/") + ".yaml"

# #         # Locate calibration yaml file or use the default otherwise
# #         if not os.path.isfile(cali_file):
# #             self.log(
# #                 f"Can't find calibration file: {cali_file}.\n Using default calibration instead.", "warn"
# #             )
# #             cali_file = os.path.join(cali_file_folder, "default.yaml")

# #         # Shutdown if no calibration file not found
# #         if not os.path.isfile(cali_file):
# #             msg = "Found no calibration file ... aborting"
# #             self.logerr(msg)
# #             rospy.signal_shutdown(msg)

# #         try:
# #             with open(cali_file, "r") as stream:
# #                 calib_data = yaml.load(stream, Loader=yaml.Loader)
# #         except yaml.YAMLError:
# #             msg = f"Error in parsing calibration file {cali_file} ... aborting"
# #             self.logerr(msg)
# #             rospy.signal_shutdown(msg)

# #         return calib_data
    def render_segments(self, img):
        for seg in self.map["segments"]:
            color = seg["color"]
            pt_x = [0,0]
            pt_y = [0,0]

            for i in range(0,2):
                p_name = seg["points"][i]
                point = self.map["points"][p_name]
                p_ref = point[0]
                if p_ref == "axle":
                    pts = self.ground2pixel(point[1])
                    pt_x[i] = pts[0]
                    pt_y[i] = pts[1]
                else:
                    pt_x[i] = point[1][0] * 100
                    pt_y[i] = point[1][1] * 100
            img = self.draw_segment(img, pt_x, pt_y, color)
        return img

    def draw_segment(self, image, pt_x, pt_y, color):
        defined_colors = {
            'red': ['rgb', [1, 0, 0]],
            'green': ['rgb', [0, 1, 0]],
            'blue': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'magenta': ['rgb', [1, 0 , 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'white': ['rgb', [1, 1, 1]],
     
            'black': ['rgb', [0, 0, 0]]}
        _color_type, [r, g, b] = defined_colors[color]
        cv2.line(image, (pt_x[0], pt_y[0]), (pt_x[1], pt_y[1]), (b * 255, g * 255, r * 255), 5)
        return image
if __name__ == '__main__':
    # create the node
    node = MyPublisherNode(node_name='augmented_reality_basics_node')
    # run node
    # node.run()
    # keep spinning
    rospy.spin()