#!/usr/bin/env python3
  


from cmath import rect
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
import os
import numpy as np
import yaml
import time

class ImagePipeline():


    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=True)

        self.camera_info = self.get_camera_info()

        self.bridge = CvBridge()

        self.sub_img= rospy.Subscriber(
            f"/{os.environ['DUCKIEBOT_NAME']}/camera_node/image/compressed", CompressedImage, self.image_cb, buff_size=10000000, queue_size=1
        )


        self.debug = rospy.Publisher(f"/debug/{node_name}/debug/compressed", CompressedImage, queue_size=1)

        #This gets the camera calibrations from the calibration file
        camera_config_path = f"config/calibrations/camera_extrinsic/{os.environ['DUCKIEBOT_NAME']}.yaml"
        yaml_config = None
        with open(camera_config_path, 'r') as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.H_matrix = np.reshape(yaml_config["homography"],(3,3))
        print(self.H_matrix)
        self.test = .1



    def image_cb(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return
        
        self.K_matrix = np.reshape(self.camera_info.K, [3,3])
        D_matrix = self.camera_info.D[:-1]
        self.P_matrix = np.reshape(self.camera_info.P, [3,4])
        
        image_size = (self.camera_info.width,self.camera_info.height)


        #Undistorts the image 
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K_matrix, D_matrix, image_size, 1, image_size)
        rectified_img = cv2.undistort(img, self.K_matrix, D_matrix, None, self.P_matrix)



        #Gets the region of interest
        roi_points = np.float32([
            (image_size[0]/2-100,image_size[1]/2), #Top left
            (0,380),  #Bottom Left
            (640,380),#Bottom right  
            (image_size[0]/2+100,image_size[1]/2)  #Top Right
            ])


        #Unsure what these do
        left_pad = 0 # was 75
        right_pad = 0
        end_points = np.float32([
            (0,0), #Top left
            (0,380),  #Bottom Left
            (640,380),#Bottom right  
            (640,0)  #Top Right
        ])
        
        #Original image size
        

        #Warps the color 
        transfrom_matrix = cv2.getPerspectiveTransform(roi_points,end_points)
        warped_img = cv2.warpPerspective(img, transfrom_matrix, image_size, flags=(cv2.INTER_LINEAR)) 

        # l1 = [[0.15, 0.0127, 1], [0.6096, 0.0127, 1]]
        BL2TL = [[0.15, 0.5, 1], [0.6096, 0.5, 1]]
        BR2TR = [[0.15, -0.2794, 1], [0.6096, -0.2794, 1]]
        # l3 = [[0.15, 0.2286, 1], [0.6096, 0.2286, 1]]
        # l4 = [[0.15, -0.0127, 1], [0.6096, -0.0127, 1]]
        lines = [BL2TL, BR2TR]#,l2,l3,l4]
        lined_img = self.line_render(img ,lines)
        self.publish_debug(lined_img)
        # print(self.test)
        # self.test += .5
        # time.sleep(.1)
        

    def ground2pix(self, point):
        "Taken from https://github.com/duckietown/dt-core/blob/4b632e57106126fa538c4500fce2b3e87dd504fd/packages/ground_projection/include/ground_projection/ground_projection_geometry.py"
        image_point = np.linalg.solve(self.H_matrix, point)
        image_point = image_point / image_point[2]
        # print(image_point)
        return image_point
        #  G = np.array([point[0], point[1], point[2], 1.0])
        #  S = self.P_matrix @ G
        #  return [S[0]//S[2],S[1]//S[2]]
    
    def pix2ground(self, pixel):
        uv_raw = np.array([pixel.u, pixel.v])
        uv_raw = np.append(uv_raw, np.array([1]))
        ground_point = np.dot(self.H_matrix, uv_raw)
        point = []
        x = ground_point[0]
        y = ground_point[1]
        z = ground_point[2]
        point[0] = x / z
        point[1] = y / z
        point[2] = 0.0
        return point

    def line_render(self, img, lines):
        new_lines = []
        for l in lines:
            new_l = []
            
            for p in l:
                p = (self.ground2pix(p))
                new_l.append([int(p[0]),int(p[1])])
            # print(new_l)
            cv2.line(img, tuple(new_l[0]), tuple(new_l[1]), (0, 255, 0), 2)
        return img



    def get_camera_info(self):
        return rospy.wait_for_message(f"/{os.environ['DUCKIEBOT_NAME']}/camera_node/camera_info",CameraInfo)

    def publish_debug(self, img):
        out_msg = self.bridge.cv2_to_compressed_imgmsg(img)
        self.debug.publish(out_msg)
  
if __name__ == '__main__':
    node = ImagePipeline("ImagePipeline")
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
