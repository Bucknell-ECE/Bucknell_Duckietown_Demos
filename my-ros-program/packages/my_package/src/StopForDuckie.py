#!/usr/bin/env python3

import cv2
import time
import psutil
import rospy
import numpy as np
import time
from threading import Thread
from typing import Tuple, cast
from collections import namedtuple
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from dt_class_utils import DTReminder


