import sys

sys.path.append("/home/gywrc-s1/xfy/xufengyu_BasePerception_0720_sam/src/SAM_seg_head_node")
sys.path.append("/home/gywrc-s1/xfy/xufengyu_BasePerception_0720_sam/src/msgs")

import os

import cv2
import numpy as np
import math
import time
import json
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message

import message_filters
from cv_bridge import CvBridge
from perception_msgs.msg import VitBoundingBoxes
from perception_msgs.msg import PointClusterVec
from perception_msgs.msg import BoundingBoxes

from segment_anything.build_ros_model import build_ros_model
import pycuda.driver as drv
# from pointcloud_cluster_cpp.lib import pointcloud_cluster
from scripts import utils
from collections import deque
from threading import Lock

def compare_ts(ts1, ts2):
    return (ts1.sec-ts2.sec) * 1e9 + (ts1.nanosec-ts2.nanosec)

def calc_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    if x1 > x2 + w2 or x2 > x1 + w1:
        return 0
    if y1 > y2 + h2 or y2 > y1 + h1:
        return 0
    inner_x1 = max(x1, x2)
    inner_x2 = min(x1 + w1, x2 + w2)
    inner_y1 = max(y1, y2)
    inner_y2 = min(y1 + h1, y2 + h2)
    inner_area = (inner_y2 - inner_y1) * (inner_x2 - inner_x1)
    outer_area = w1 * h1 + w2 * h2 - inner_area
    return inner_area / outer_area

class SAMObstacleNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("启动%s node" % name)
        self.bridge = CvBridge()

        self.image_emb_sub = message_filters.Subscriber(self, VitBoundingBoxes, '/vit_bounding_boxes')
        #self.image_sub = message_filters.Subscriber(self, CompressedImage, '/image2/compressed')
        self.lidar_sub = message_filters.Subscriber(self, PointClusterVec, '/sam_point_cluster')
        # self.yolov8_sub = message_filters.Subscriber(self, BoundingBoxes, '/yolov8/bounding_boxes')
        # ts = message_filters.ApproximateTimeSynchronizer([self.image_emb_sub, self.lidar_sub], queue_size=20, slop=0.1)
        # ts.registerCallback(self.callback)
        
        self.device = 0
        self.mask_decoder_model = build_ros_model('mask_decoder', self.device)
        fcc = cv2.VideoWriter_fourcc(*'XVID')
        self.vw=cv2.VideoWriter('/home/gywrc-s1/qianhao/workspace/video/hby_obstacle001.avi', fcc, 10.0, (1920, 1200))
        self.iou_thres = 0.6
        self.lidar_queue = deque()
        self.max_queue_size = 20
        self.img_subscription= self.create_subscription(VitBoundingBoxes, "/vit_bounding_boxes", self.callback, 1)
        self.lidar_subscription= self.create_subscription(PointClusterVec, "/sam_point_cluster", self.lidar_callback, 1)
        print('init done')
        
        # self.lidar_queue_size = 30

    def callback(self, msg_emb):
        image_embedding, check = self.mask_decoder_model.get_buffer(msg_emb.ipc_header, msg_emb.check_header)
        self.get_logger().info('emb_msg received')
        if int(msg_emb.header.stamp.nanosec) == check:
            self.get_logger().info("ipc check success")
        else:
            self.get_logger().info("ipc check failed")
            return

        while True:
            if len(self.lidar_queue) == 0:
                self.get_logger().info('lidar_msg not received')
                return
            msg_lidar = self.lidar_queue[0]
            comp = compare_ts(msg_emb.header.stamp, msg_lidar.header.stamp)
            comp = comp / 1e9
            if abs(comp)<0.1:
                self.get_logger().info('found lidar_msg')
                break
            elif comp < -0.1:
                self.get_logger().info('lidar_msg too late')
                return
            else:
                self.lidar_queue.popleft()
        if msg_lidar.size == 0:
            return
        
        points = np.zeros([msg_lidar.size, 6], dtype=np.float32)
        img_data = self.bridge.compressed_imgmsg_to_cv2(msg_emb.image, 'bgr8')
        for i, lidar_single_msg in enumerate(msg_lidar.vec):
            points[i, 0] = lidar_single_msg.x
            points[i, 1] = lidar_single_msg.y
            points[i, 2] = lidar_single_msg.z
            points[i, 5] = lidar_single_msg.label
        lidar_boxes, coords, res = self.mask_decoder_model.infer(image_embedding, points)
        removed_lidar_boxes = []
        removed_coords = []
        used_lidar_indexes = set()
        for single_res in res:
            used_lidar_indexes.add(single_res["cluster_label"][0])
        for i in range(len(lidar_boxes)):
            lidar_box = lidar_boxes[i]
            coord = coords[i]
            if i not in used_lidar_indexes:
                removed_lidar_boxes.append(lidar_box)
                removed_coords.append(coord)
        yolov8_boxes = msg_emb.data
        n_classes = len(coords)
        for y_box in yolov8_boxes:
            # y_box.x = y_box.x - 0.5 * y_box.w
            # y_box.y = y_box.y - 0.5 * y_box.h
            yolo_bbox = [y_box.x - 0.5 * y_box.w, y_box.y - 0.5 * y_box.h, y_box.w, y_box.h ]
            max_iou = 0
            box_index = -1
            for i, single_res in enumerate(res):
                bbox = single_res['bbox']
                iou = calc_iou(yolo_bbox, bbox)
                if iou > max_iou:
                    max_iou = iou
                    box_index = i
            if max_iou > self.iou_thres:
                res[box_index]['bbox'] = yolo_bbox
            else:
                new_added_result = {}
                new_added_result['bbox'] = yolo_bbox
                new_added_result['segmentation'] = None
                for i, removed_lidar_box in enumerate(removed_lidar_boxes):
                    iou = calc_iou(yolo_bbox, removed_lidar_box)
                    if iou > max_iou:
                        max_iou = iou
                        box_index = i
                if max_iou > self.iou_thres:
                    new_added_result['point_coords'] = removed_coords[box_index]
                    new_added_result['segmentation'] = None
                res.append(new_added_result)
        utils.show_lidar_result(img_data, coords=coords, res=res, show_mask=True, video_writer=self.vw)

        # vit_res_info['timestamp'] = str(msg.header.stamp.sec) + '.' + str(msg.header.stamp.nanosec)
        # feature = json.dumps(vit_res_info)
        # print(type(feature))
        
    def lidar_callback(self, lidar_msg):
        self.lidar_queue.append(lidar_msg)
        if len(self.lidar_queue) > self.max_queue_size:
            self.lidar_queue.popleft()


def main(args=None):
    # 初始化ROS2
    rclpy.init(args=args)
    # 创建节点
    minimal_subscriber = SAMObstacleNode('SAMObstacle')
    # 运行节点
    rclpy.spin(minimal_subscriber)
    # 销毁节点，退出ROS2
    # minimal_subscriber.destroy_node()
    rclpy.shutdown()
