import sys
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
from perception_msgs.msg import VitImageEmbedding, BoundingBoxes
from perception_msgs.msg import PointClusterVec
from perception_msgs.msg import SamResults, OnePoint, OneBox
from geometry_msgs.msg import PoseStamped
from segment_anything.build_ros_model import build_ros_model
from . import infer_utils
from collections import deque
from threading import Lock
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import open3d as o3d
import threading

class Ros2MsgError(Exception):
    def __init__(self,errorInfo):
        Exception.__init__(self)
        self.errorInfo=errorInfo

    def __str__(self):
        return "Ros2MsgError"

class ModelSingleton():
    __model = None
    def __init__(self) -> None:
        return

    def __new__(cls):
        if not cls.__model:
            cls.__model = build_ros_model('mask_decoder', 0)
        return cls.__model

    @classmethod
    def release(cls):
        cls.__model.model = None

def cal_iou(box: np.ndarray, boxes: np.ndarray):
    """ 计算一个边界框和多个边界框的交并比
    Parameters
    ----------
    box: `~np.ndarray` of shape `(4, )`
        边界框
    boxes: `~np.ndarray` of shape `(n, 4)`
        其他边界框
    Returns
    -------
    iou: `~np.ndarray` of shape `(n, )`
        交并比
    """
    # 计算交集
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0]*inter[:, 1]
    # 计算面积
    area_boxes = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0])*(box[3]-box[1])
    return inter/(area_box+area_boxes-inter)


def get_distance(points):

    # 0719
    if type(points) == list:
        points = np.reshape(np.array(points), (1,3))

    if points.shape == (3,):
        xx = points[0]
        yy = points[1]
        zz = points[2]
    else:
        xx = points[:, 0]
        yy = points[:, 1]
        zz = points[:, 2]
    distance = np.sqrt(xx * xx + yy * yy + zz * zz)
    return distance


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
        
        self.device = 0
        # self.mask_decoder_model = build_ros_model('mask_decoder', self.device)
        self.mask_decoder_model = ModelSingleton()
        fcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.vw=cv2.VideoWriter('/home/gywrc-s1/qianhao/workspace/video/hby_obstacle001.avi', fcc, 10.0, (1920, 1080))
        self.vw=None
        self.iou_thres = 0.6
        self.lidar_queue = deque()
        self.max_queue_size = 20
        self.yolo_queue = deque()
        self.yolo_queue_size = 20
        self.img_callback_group = MutuallyExclusiveCallbackGroup()
        self.lidar_callback_group = MutuallyExclusiveCallbackGroup()
        self.yolo_callback_group = MutuallyExclusiveCallbackGroup()
        # lidar_callback_group = None
        self.qos_profile_img = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.img_subscription= self.create_subscription(VitImageEmbedding, "/Image_embedding", self.callback, self.qos_profile_img,  callback_group=self.img_callback_group, )
        self.lidar_subscription= self.create_subscription(PointClusterVec, "/sam_point_cluster", self.lidar_callback, self.qos_profile, callback_group=self.lidar_callback_group)
        self.yolo_subscription = self.create_subscription(BoundingBoxes, "/yolov8/bounding_boxes_mask", self.yolo_callback, self.qos_profile, callback_group=self.yolo_callback_group)
        self.task_num = self.create_subscription(PoseStamped, 'task_num', self.task_callback, self.qos_profile)
        self.sam_obstacle_pub = self.create_publisher(SamResults, "/sam_obstacle", 1)
        self.hang = False
        self.warning_index = 0
        self.result_index = 0
        self.saveimg = False
        print('init done')
        
        # self.lidar_queue_size = 30

    def callback(self, msg_emb):
        image_embedding, check = self.mask_decoder_model.get_buffer(msg_emb.ipc_header, msg_emb.check_header)
        if int(msg_emb.header.stamp.nanosec) != check:
            self.get_logger().info("ipc check failed")
            return
        while True:
            if len(self.lidar_queue) == 0:
                self.get_logger().info('lidar_msg not received')
                self.warning_index += 1
                if self.warning_index >=10:
                    raise Ros2MsgError("lidar_msg not received more than 10 times")
                return
            msg_lidar = self.lidar_queue[0]
            comp = compare_ts(msg_emb.header.stamp, msg_lidar.header.stamp)
            comp = comp / 1e9
            if abs(comp)<0.15:
                self.get_logger().info('found lidar_msg')
                break
            elif comp < -0.15:
                self.get_logger().info('lidar_msg too late')
                return
            else:
                self.lidar_queue.popleft()
        if msg_lidar.size == 0:
            return
        start_t=time.time()
        points = np.zeros([msg_lidar.size, 6], dtype=np.float32)
        img_data = self.bridge.compressed_imgmsg_to_cv2(msg_emb.image, 'bgr8')
        for i, lidar_single_msg in enumerate(msg_lidar.vec):
            points[i, 0] = lidar_single_msg.x
            points[i, 1] = lidar_single_msg.y
            points[i, 2] = lidar_single_msg.z
            points[i, 5] = lidar_single_msg.label
        lidar_boxes, ori_coords, coords, res = self.mask_decoder_model.infer(image_embedding, points)
        removed_lidar_boxes = []
        removed_coords = []
        used_lidar_indexes = set()
        max_iou_dicts = {}
        for i, single_res in enumerate(res):
            cluster_label = single_res['cluster_label'][0]
            iou_preds = single_res['iou_preds']
            if cluster_label not in max_iou_dicts:
                max_iou_dicts[cluster_label]=(iou_preds, i)
            else:
                if iou_preds > max_iou_dicts[cluster_label][0]:
                    max_iou_dicts[cluster_label]=(iou_preds, i)
        res2 = []
        for k, v in max_iou_dicts.items():
            res2.append(res[v[1]])
        res = res2
        for single_res in res:
            used_lidar_indexes.add(single_res["cluster_label"][0])
        for i in range(len(lidar_boxes)):
            lidar_box = lidar_boxes[i]
            coord = coords[i]
            if i not in used_lidar_indexes:
                removed_lidar_boxes.append(lidar_box)
                removed_coords.append(coord)
        # yolov8_boxes = msg_emb.data
        yolov8_boxes = []
        while len(self.yolo_queue) > 0:
            msg_yolo = self.yolo_queue[0]
            comp = compare_ts(msg_emb.header.stamp, msg_yolo.header.stamp)
            comp = comp / 1e9
            if comp < -0.1:
                break
            elif abs(comp) <= 0.1:
                yolov8_boxes = msg_yolo.data
                break
            else:
                self.yolo_queue.popleft()
        n_classes = len(coords)
        for y_box in yolov8_boxes:
            # y_box.x = y_box.x - 0.5 * y_box.w
            # y_box.y = y_box.y - 0.5 * y_box.h
            yolo_bbox = [y_box.x - 0.5 * y_box.w, y_box.y - 0.5 * y_box.h, y_box.w, y_box.h, y_box.object_class]
            max_iou = 0
            box_index = -1
            for i, single_res in enumerate(res):
                bbox = single_res['bbox']
                iou = calc_iou(yolo_bbox[:4], bbox[:4])
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
                    iou = calc_iou(yolo_bbox[:4], removed_lidar_box[:4])
                    if iou > max_iou:
                        max_iou = iou
                        box_index = i
                if max_iou > self.iou_thres:
                    new_added_result['segmentation'] = None
                res.append(new_added_result)
        infer_t=time.time()
        
        boxls = []
        myboxs = []
        if res != []:
            for ans in res:

                bx = OneBox()

                bx.cid = 0

                bbx = [float(a) for a in ans['bbox']]

                if len(bbx) == 5:
                    bbx = [float(a) for a in ans['bbox'][:4]]
                    bx.cid = int(ans['bbox'][4])
                else:
                    bbx = [float(a) for a in ans['bbox']]

                bx.x1 = bbx[0]
                bx.y1 = bbx[1]
                bx.x2 = bbx[0] + bbx[2]
                bx.y2 = bbx[1] + bbx[3]
                
                boxls.append(bx)
                myboxs.append([bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]])

        pointls = []
        if ori_coords != []:
            point_cloud = o3d.geometry.PointCloud()
            for ol, cod in zip(ori_coords, coords):
                point_cloud.points = o3d.utility.Vector3dVector(ol)
                axis_aligned_box = point_cloud.get_axis_aligned_bounding_box()
                mincords, maxcords = axis_aligned_box.get_min_bound(), axis_aligned_box.get_max_bound()
                x1, y1, z1 = mincords
                x2, y2, z2 = maxcords
                # xc, yc, zc = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2
                # pcd_center = np.array([xc, yc, zc])
                # nowdist = float(utils.get_distance(pcd_center))
                pcd_center = np.mean(ol, axis=0)
                nowdist = float(get_distance(pcd_center))

                ct = np.mean(cod, axis=0)

                pt = OnePoint()
                pcd_center = pcd_center.astype(np.float64)
                
                pt.x3d = pcd_center[0]
                pt.y3d = pcd_center[1]
                pt.z3d = pcd_center[2]

                pt.x1 = x1
                pt.y1 = y1
                pt.z1 = z1
                pt.x2 = x2
                pt.y2 = y2
                pt.z2 = z2

                pt.x = ct[0]
                pt.y = ct[1]
                pt.distance = nowdist

                pointls.append(pt)

                xmin, ymin, xmax, ymax = np.min(cod[:,0]), np.min(cod[:,1]), np.max(cod[:,0]), np.max(cod[:,1])

                if len(myboxs) == 0:
                    myboxs.append([xmin, ymin, xmax, ymax])
                    nbx = OneBox()
                    nbx.x1, nbx.y1, nbx.x2, nbx.y2 = float(xmin), float(ymin), float(xmax), float(ymax)
                    boxls.append(nbx)
                    continue

                if len(myboxs) == 1:
                    iounp = cal_iou(np.array([xmin, ymin, xmax, ymax]), np.array(myboxs).reshape((1, 4)))
                else:
                    iounp = cal_iou(np.array([xmin, ymin, xmax, ymax]), np.array(myboxs))

                ioumax = np.amax(iounp)
                ioumax_idx = np.argmax(iounp)

                if ioumax < 0.2:
                    nbx = OneBox()
                    nbx.x1, nbx.y1, nbx.x2, nbx.y2 = float(xmin), float(ymin), float(xmax), float(ymax)
                    boxls.append(nbx)
                    myboxs.append([xmin, ymin, xmax, ymax])
                

        # spoints.coords = [cod.tolist() for cod in coords]
        # spoints.ojblidars = [obl.tolist() for obl in objlidars]

        # sboxs = SamBoxs()
        # spoints = SamPoints()

        # sboxs.boxes = boxls
        # spoints.points = pointls

        # self.publish_samboxs.publish(sboxs)
        # self.publish_sampoints.publish(spoints)


        samres = SamResults()
        samres.boxes = boxls
        samres.points = pointls
        samres.image = msg_emb.image
        samres.header.stamp.sec = msg_emb.header.stamp.sec
        samres.header.stamp.nanosec = msg_emb.header.stamp.nanosec
        # self.sam_obstacle_pub.publish(samres)
        img_data = infer_utils.show_lidar_result(img_data, coords=coords, res=res, show_mask=False, video_writer=self.vw)
        if len(res) > 0 and self.saveimg:
            cv2.imwrite('result_imgs/obstacle_%s.jpg'%(str(self.result_index).rjust(4, '0')), img_data)
            self.result_index += 1
        fps=1/(time.time()-start_t)
        infer_fps=1/(infer_t-start_t)
        print("fps: %2.3f, infer_fps: %2.3f" % (fps, infer_fps))

    def lidar_callback(self, lidar_msg):
        self.warning_index = 0
        print("lidar ts: %d, lidar_queue_size: %d"%(lidar_msg.header.stamp.sec, len(self.lidar_queue)))
        if len(self.lidar_queue) > 0 and self.lidar_queue[-1].header.stamp.sec > lidar_msg.header.stamp.sec:
            self.lidar_queue = deque()
        self.lidar_queue.append(lidar_msg)
        if len(self.lidar_queue) > self.max_queue_size:
            self.lidar_queue.popleft()
            
    def yolo_callback(self, yolo_msg):
        print("yolo ts: %d, yolo_queue_size: %d"%(yolo_msg.header.stamp.sec, len(self.yolo_queue)))
        if len(self.yolo_queue) > 0 and self.yolo_queue[-1].header.stamp.sec > yolo_msg.header.stamp.sec:
            self.yolo_queue = deque()
        self.yolo_queue.append(yolo_msg)
        if len(self.yolo_queue) > self.yolo_queue_size:
            self.yolo_queue.popleft()

    def task_callback(self, task_msg):
        print(task_msg.pose.position.z)
        if int(task_msg.pose.position.z) == 14:
            ModelSingleton.release()
        elif int(task_msg.pose.position.z) in (6,7,10,11):
            self.saveimg = True
        else:
            self.mask_decoder_model = ModelSingleton()
            self.saveimg = False

def try_spin(node):
    try:
        rclpy.spin(node)
    except Ros2MsgError:
        node.destroy_node()
        node = SAMObstacleNode("SAMObstacle")
        try_spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    # 初始化ROS2
    rclpy.init(args=args)
    # 创建节点
    minimal_subscriber = SAMObstacleNode('SAMObstacle')
    # 运行节点
    try_spin(minimal_subscriber)

