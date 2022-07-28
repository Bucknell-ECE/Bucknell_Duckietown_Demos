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
        
        self.vehicle_name = os.environ.get("VEHICLE_NAME")
        self.joyscribe3 = rospy.Subscriber("/joy", Joy, self.print_joy, queue_size=1)
        self.pub_wheels_cmd = rospy.Publisher(
            f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1
        )
        # construct publisher
        # self.pub = rospy.Publisher('chatter', String, queue_size=10)

    def print_joy(self, joy_msg):
        wheels_cmd_msg = WheelsCmdStamped()
        wheels_cmd_msg.header.stamp = rospy.Time.now()
        R, L = self.convert_to_tank(joy_msg.axes[0:2])
        boost = ((joy_msg.axes[2] + 1) * 2) + .1
        wheels_cmd_msg.vel_left = L * boost
        wheels_cmd_msg.vel_right = R * boost
       
        self.pub_wheels_cmd.publish(wheels_cmd_msg)
        rospy.loginfo(f"left:{L} right:{R} boost:{boost} leftBoost:{L*boost} rightBoost:{R * boost}" )

    def convert_to_tank(self,axes):
        """
        Got this formula from https://home.kendra.com/mauser/joystick.html
        """
        x = axes[0]
        y = axes[1]
        # x = -x
        v = (1.0-abs(x)) * (y) + y
        w = (1.0-abs(y)) * (x) + x
        r = (v+w) / 2
        l = (v-w) / 2
        return r,l


    # def run(self):
    #     # publish message every 1 second
    #     rate = rospy.Rate(1) # 1Hz
    #     while not rospy.is_shutdown():
    #         message = "Hello World!"
    #         rospy.loginfo("Publishing message: '%s'" % message)
    #         self.pub.publish(message)
    #         rate.sleep()

if __name__ == '__main__':
    # create the node
    node = MyPublisherNode(node_name='laptop_pub_node')
    # run node
    # node.run()
    # keep spinning
    rospy.spin()