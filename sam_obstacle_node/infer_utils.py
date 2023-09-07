import torch
import numpy as np
import sys
sys.path.append("src/SAM_seg_head_node")
from segment_anything_yhl.utils.amg import batch_iterator, MaskData, calculate_stability_score, batched_mask_to_box, box_xyxy_to_xywh
from segment_anything_yhl.utils.transforms import ResizeLongestSide
from segment_anything_yhl.utils.onnx import SamOnnxModel
import cv2
from torchvision.ops.boxes import batched_nms
import random

random_colors = [[random.randint(0,255) for _ in range(3)] for _ in range(100)]

def show_lidar_result(img, coords, res, show_mask=False, video_writer=None):
    import cv2, random, sys
    img = cv2.resize(img, [1920, 1080])
    if coords is not None:
        for i, coord in enumerate(coords):
            color = [255,0,0]
            # color = random_colors[i]
            for pixel in coord:
                pixel = pixel.astype('int32')
                img = cv2.circle(img, pixel, 2, color, 2)
    for i in range(len(res)):
        color = random_colors[i]
        if 'bbox' not in res[i]:
            continue
        bbox=[int(x) for x in res[i]['bbox']]
        x, y, w, h = bbox[:4]
        obj_class = 0
        if len(bbox) == 5:
            obj_class = bbox[4]+1
        # img = cv2.rectangle(img, [x, y], [x+w, y+h], color, 4)
        img = cv2.rectangle(img, [x, y], [x+w, y+h], [0,0,255], 4)
        cls = ['unknown', 'armored_car', 'smog', 'person', 'car', 'cones', 'barrel']
        img = cv2.putText(img, cls[obj_class], [x, y+h+12], cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 2)
        if 'blh' in res[i]:
            blh = res[i]['blh']
            img = cv2.putText(img, blh, [x, y+h+48], cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,255], 2)
        if show_mask:
            mask=res[i]['segmentation']
            if mask is not None:
                img[mask]=color
    cv2.imshow("obstacle", cv2.resize(img, [640, 360]))
    if video_writer:
        video_writer.write(img)
    cv2.waitKey(1)
    return img

def resample_coords(coords, max_point, n_resample):
    import random
    coords_resampled = []
    labels = []
    for i, coord in enumerate(coords):
        n_coord = coord.shape[0]
        if n_coord < max_point:
            coords_resampled.append(coord)
            labels.append(i)
        else:
            indexes=list(range(n_coord))
            random.shuffle(indexes)
            duplicates = max_point * n_resample / len(indexes)
            if duplicates > 0:
                indexes = indexes * (int(duplicates) + 1)
            for j in range(n_resample):
                coords_resampled.append(coord[indexes[j*max_point: (j+1)*max_point]])
                labels.append(i)
    return labels, coords_resampled

def process_data(data, cluster_mode=False, use_lidar=False):
    pred_iou_thresh = 0.88
    mask_threshold = 0.0
    stability_score_thresh = 0.95
    stability_score_offset = 1.0
    # Filter by predicted IoU
    if use_lidar:
        stability_score_thresh = 0.5
        if cluster_mode:
            lidar_iou_thresh = 0.5
            keep_mask = data["lidar_iou"] > lidar_iou_thresh
            data.filter(keep_mask)
            pred_iou_thresh = 0.5
    if pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)
    # Calculate stability score
    data["stability_score"] = calculate_stability_score(
        data["masks"], mask_threshold, stability_score_offset
    )
    keep_mask = data["stability_score"] >= stability_score_thresh
    data.filter(keep_mask)
    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])
    
def get_lidar_iou(bbox1, bbox2):
    b1, _ = bbox1.shape
    b2, n2, _ = bbox2.shape
    bbox1 = torch.tile(bbox1[:,None,:], [1, n2, 1])
    x1_max = torch.maximum(bbox1[:,:,0], bbox2[:,:,0])
    y1_max = torch.maximum(bbox1[:,:,1], bbox2[:,:,1])
    x2_min = torch.minimum(bbox1[:,:,2], bbox2[:,:,2])
    y2_min = torch.minimum(bbox1[:,:,3], bbox2[:,:,3])
    inter_area = torch.maximum(x2_min-x1_max, torch.tensor(0)) * torch.maximum(y2_min-y1_max, torch.tensor(0))
    outer_area = (bbox1[:,:,2]-bbox1[:,:,0]) * (bbox1[:,:,3]-bbox1[:,:,1]) + (bbox2[:,:,2]-bbox2[:,:,0]) * (bbox2[:,:,3]-bbox2[:,:,1]) - inter_area
    iou = inter_area / outer_area
    return iou


class LidarParam:
    def __init__(self) -> None:
        self.camera_matrix = np.array([1358.080518, 0.0, 987.462437,
                              0.0, 1359.770396, 585.756872,
                              0.0, 0.0, 1.0]).reshape((3, 3))

        self.transform = np.array([
            0.02158791, -0.99976086, 0.00349065, -0.24312639,
            -0.01109192, -0.00373076, -0.99993152, -0.22865444,
            0.99970542, 0.02154772, -0.01116981, -0.37689865
        ], dtype=np.float32).reshape(3, 4)

        self.distortion = np.array([[-0.406858, 0.134080, 0.000104, 0.001794, 0.0]], dtype=np.float32)

        # 获取更新后的 旋转和平移
        self.rMat = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
        self.tVec = np.array([0, 0, 0], dtype=np.float32).reshape(1, 3)

class MaskDecoderUtil():
    def __init__(self, model) -> None:
        self.model = SamOnnxModel(model=model, return_single_mask=False)
        self.device = torch.device(0)
        self.max_point = 20
        self.project_max_point = 100
        self.use_lidar = True
        self.cluster_mode = True
        self.origin_image_shape = [1080, 1920]
        self.transf = ResizeLongestSide(1024)
        self.lidar_param = LidarParam()
        self.points_per_batch = 32

        self.mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=self.device)
        self.has_mask_input = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.orig_size = torch.as_tensor(self.origin_image_shape, dtype=torch.int32, device=self.device)

    def _project_by_cluster(self, points):
        labels = points[:,5].astype(int)
        coords = []
        ori_coords = []
        for i in range(labels.max() + 1):
            cluster = points[labels == i]
            n_cluster_point = cluster.shape[0]
            if n_cluster_point > self.project_max_point:
                shuffle = np.random.randint(0, n_cluster_point, size=self.project_max_point)
                cluster = cluster[shuffle]
            cluster = cluster[:, :3].astype(np.float32)
            tmp_point_cloud = np.hstack((cluster, np.ones([len(cluster), 1])))
            cluster = np.dot(tmp_point_cloud, self.lidar_param.transform.T)
            reTransform = cv2.projectPoints(cluster, self.lidar_param.rMat, self.lidar_param.tVec, self.lidar_param.camera_matrix, self.lidar_param.distortion)
            coord = reTransform[0][:, 0].astype(np.int32)
            filter = np.where((coord[:, 0] < self.origin_image_shape[1]) & (coord[:, 1] < self.origin_image_shape[0]) & (coord[:, 0] >= 0) & (coord[:, 1] >= 0))
            coord = coord[filter]
            ori_coord = cluster[filter]
            if coord.shape[0] == 0:
                continue
            coords.append(coord)
            ori_coords.append(ori_coord)
        return ori_coords, coords
    
    def infer(self, *inputs):
        image_embedding, lidar_points = inputs
        ori_coords, coords = self._project_by_cluster(lidar_points)
        n_classes = len(coords)
        print(n_classes)
        orig_lidar_box = np.zeros([n_classes, 4])
        if self.cluster_mode:
            coords_labels, coords_resample = resample_coords(coords, max_point=self.max_point, n_resample=2)
            n_resampled_class = len(coords_resample)
            coord_arr = np.zeros([n_resampled_class, self.max_point, 2], dtype=np.float32)
            label_arr = np.zeros([n_resampled_class, self.max_point], dtype=np.float32)
            cluster_arr = np.zeros([n_resampled_class], dtype=np.int32)
            lidar_boxes = np.zeros([n_resampled_class, 4])
            for i in range(n_classes):
                coord = coords[i]
                orig_lidar_box[i, 0] = np.min(coord[:, 0], axis=0)
                orig_lidar_box[i, 1] = np.min(coord[:, 1], axis=0)
                orig_lidar_box[i, 2] = np.max(coord[:, 0], axis=0)
                orig_lidar_box[i, 3] = np.max(coord[:, 1], axis=0)
            for i in range(n_resampled_class):
                coord = coords_resample[i]
                coord_num = coord.shape[0]
                coord_arr[i, :coord_num, :] = coord
                label_arr[i, :coord_num] = 1
                label_arr[i, coord_num:] = -1
                cluster_arr[i] = coords_labels[i]
                lidar_boxes[i] = orig_lidar_box[coords_labels[i]]
            coords_input = torch.from_numpy(coord_arr).to(self.device)
            labels_input = torch.from_numpy(label_arr).to(self.device)
            lidar_boxes = torch.from_numpy(lidar_boxes).to(self.device)
        else:
            coord = np.concatenate(coords, axis=0)
            n_coord = coord.shape[0]
            coords_input = torch.from_numpy(coord[:,None,:])
            labels_input = torch.ones([n_coord, 1], dtype=torch.float32, device=self.device)
            cluster_arr = np.zeros([n_coord], dtype=np.int32)
        ort_inputs = {
            "image_embeddings": image_embedding,
            "mask_input": self.mask_input,
            "has_mask_input": self.has_mask_input,
            "orig_im_size": self.orig_size
        }
        res = []
        mask_data = MaskData()
        for (coord_input, label_input, lidar_box, cluster_input) in batch_iterator(self.points_per_batch, coords_input, labels_input, lidar_boxes, cluster_arr):
            coord_input = self.transf.apply_coords(coord_input, self.origin_image_shape)
            ort_inputs["point_coords"] = coord_input
            ort_inputs["point_labels"] = label_input
            # _, iou_preds, masks = self.model.torch_inference(ort_inputs)
            masks, iou_preds, _ = self.model(image_embedding, coord_input, label_input, self.mask_input, self.has_mask_input, self.orig_size)
            if self.cluster_mode:
                sam_box = batched_mask_to_box(masks>0)
                lidar_iou=get_lidar_iou(lidar_box, sam_box)
                batch_data = MaskData(
                   masks=masks.flatten(0, 1),
                   iou_preds=iou_preds.flatten(0, 1),
                   lidar_iou=lidar_iou.flatten(0, 1),
                   points=torch.as_tensor(coord_input.repeat([masks.shape[1],1,1])),
                   cluster_label=torch.as_tensor(cluster_input.repeat([masks.shape[1]]))
                #    iou_token_out=iou_token_out.flatten(0, 1)
                )
            else:
                batch_data = MaskData(
                   masks=masks.flatten(0, 1),
                   iou_preds=iou_preds.flatten(0, 1),
                   points=torch.as_tensor(coord_input.repeat([masks.shape[1],1,1])),
                #    iou_token_out=iou_token_out.flatten(0, 1)
                )
            process_data(batch_data, self.use_lidar, self.cluster_mode)
            mask_data.cat(batch_data)
        if len(mask_data.items()) == 0:
            return orig_lidar_box, ori_coords, coords, res
        if self.cluster_mode:
            iou_threshold = 0.5
        else:
            iou_threshold = 0.77
        keep_by_nms = batched_nms(
            mask_data["boxes"].float(),
            mask_data["iou_preds"],
            torch.zeros_like(mask_data["boxes"][:, 0]),  # categories
            iou_threshold=iou_threshold,
        )
        mask_data.filter(keep_by_nms)
        mask_data["segmentations"] = mask_data["masks"]
        mask_data.to_numpy()
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "cluster_label": [mask_data["cluster_label"][idx]]
                # "iou_token_out": mask_data["iou_token_out"][idx],
            }
            res.append(ann)
        return orig_lidar_box, ori_coords, coords, res

