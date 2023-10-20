# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys
import numpy as np
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from eval_det import eval_det_cls, eval_det_multiprocessing
from eval_det import get_iou_obb
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from box_util import get_3d_box, extract_pc_in_box3d, get_3d_box_depth
from model_util import class2size, class2angle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


def parse_prediction_to_pseudo_bboxes(end_points, config_dict, point_clouds):
    """ Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        end_points: dict
            {center, heading_scores, heading_residuals, size_residuals, sem_cls_scores} with batch_size=1
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}
        point_clouds: numpy array with shape (N,3+pc_attr), sample point cloud
        instance_bboxes: numpy array with shape (*, 8) include [cx,cy,cz, dx,dy,dz, heading angle, class_ind],
                       the instances w.r.t. novel classes in the sample point cloud

    Returns:
        pred_bboxes: numpy array with shape (num_valid_detections, 8), each colomn is with (x,y,z,l,w,h,heading angle, class_ind)
    """
    # print('parse_prediction_to_pseudo_bboxes')
    pred_center = end_points['center'].squeeze(0)  # num_proposal,3
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # 1, num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1)).squeeze()  # num_proposal
    pred_heading_class.squeeze_(0)
    pred_sem_cls = end_points['sem_cls'].squeeze(0) # num_proposal
    pred_sem_cls_probs = torch.softmax(end_points['sem_cls_scores'], dim=2)  # B, num_proposal, num_class
    # print('pred_sem_cls_probs.shape', pred_sem_cls_probs.shape)

    # pred_sem_cls_probs = torch.max(pred_sem_cls_probs.squeeze(0), dim=1)[0]
    choose=0
    if choose==0:#set threshold=0
        pred_sem_cls_probs = torch.max(pred_sem_cls_probs.squeeze(0), dim=1)[0]
    if choose ==1:  #flexmatch
        pred_sem_cls_probs, pred_sem_cls_max_indices = torch.max(pred_sem_cls_probs.squeeze(0), dim=1)
        # print("pred_sem_cls_max_values.shape", pred_sem_cls_probs)
        # print("pred_sem_cls_max_indices.shape", pred_sem_cls_max_indices)
        counts = torch.zeros(14, dtype=torch.int32)

        indices_array = pred_sem_cls_max_indices.cpu().numpy()
        txt_file_path = "threshold_data.txt"

        for index in indices_array:
            if 0 <= index < 14:
                counts[index] += 1
        # 指定日志文件保存的目录
        log_dir = "./logs_flexmatch"

        # 创建SummaryWriter对象
        writer = SummaryWriter(log_dir)

        max_count = torch.max(counts).item()
        normalized_counts = counts.float() / max_count
        with open(txt_file_path, 'a') as txt_file:
            for i in range(14):
                threshold = config_dict['obj_conf_thresh']*normalized_counts[i].item()
                writer.add_scalar(f"class_{i}/threshold", threshold)
                txt_file.write(f"class_{i}/threshold: {threshold}\n")
                # writer.add_scalar(f"class {i} threshold: {config_dict['obj_conf_thresh']*normalized_counts[i].item()}")
                # print(f"class {i} threshold: {config_dict['obj_conf_thresh']*normalized_counts[i].item()}")

                # print('pred_sem_cls_probs.shape', pred_sem_cls_probs.shape)
    elif choose ==2:  #freematch
        if config_dict['obj_conf_thresh']==0.95:
            config_dict['obj_conf_thresh'] = np.full(15, 1/14, dtype=np.float32)
            
        pred_sem_cls_probs, pred_sem_cls_max_indices = torch.max(pred_sem_cls_probs.squeeze(0), dim=1)
        max_pred_sem_cls_probs = torch.max(pred_sem_cls_probs).item()
        config_dict['obj_conf_thresh'][0]=max_pred_sem_cls_probs*0.3+config_dict['obj_conf_thresh'][0]*0.7#怎么保证不大于1的
        num_classes = 14  
        class_means = []

        for class_idx in range(num_classes):
            # find the matching index with the current category
            matching_indices = (pred_sem_cls_max_indices == class_idx).nonzero()
            
            if len(matching_indices) > 0:
                matching_indices = matching_indices.view(-1)
                # use these indexes to get the corresponding probability values
                matching_probs = pred_sem_cls_probs[matching_indices]
                # compute the mean
                class_mean = matching_probs.mean().item()
                class_means.append(class_mean)
            else:
                # if there is no matching index, the mean is 0
                class_means.append(0.0)

        class_means = torch.tensor(class_means, device='cuda:0')
        max_value = max(class_means)
        class_means_normalized = [x / max_value for x in class_means]
        print(class_means_normalized)
        for i in range(14):
            config_dict['obj_conf_thresh'][i + 1] = config_dict['obj_conf_thresh'][i + 1]*0.7+class_means_normalized[i]*0.3
        # create a breakpoint here
        import pdb;pdb.set_trace()
    elif choose ==3:  #softmatch
        pred_sem_cls_probs, pred_sem_cls_max_indices = torch.max(pred_sem_cls_probs.squeeze(0), dim=1)
    pred_size_residual = end_points['size_residuals'].squeeze(0)  # num_proposal,3

    pred_center = pred_center.detach().cpu().numpy()
    pred_heading_class = pred_heading_class.detach().cpu().numpy()
    pred_heading_residual = pred_heading_residual.detach().cpu().numpy()
    pred_sem_cls = pred_sem_cls.detach().cpu().numpy()
    pred_sem_cls_probs = pred_sem_cls_probs.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.detach().cpu().numpy()

    num_proposal = pred_center.shape[0]
    K = num_proposal
    # Since we operate in upright_depth coord for points, while util functions
    # assume depth coord.
    pred_corners_3d_depth = np.zeros((K, 8, 3))
    pred_heading_angle = np.zeros((K))
    pred_size = np.zeros((num_proposal,3))
    for i in range(num_proposal):
        heading_angle = class2angle(pred_heading_class[i],pred_heading_residual[i], config_dict['dataset_config'])
        box_size = class2size(int(pred_sem_cls[i]), pred_size_residual[i], config_dict['dataset_config'])
        corners_3d = get_3d_box_depth(box_size, heading_angle, pred_center[i])
        pred_corners_3d_depth[i] = corners_3d
        pred_heading_angle[i] = heading_angle
        pred_size[i] = box_size

    nonempty_box_mask = np.ones((K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        pc = point_clouds[:, 0:3]  # N,3
        for i in range(num_proposal):
            box3d = pred_corners_3d_depth[i, :, :]  # (8,3)
            pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
            if len(pc_in_box) < 5:
                nonempty_box_mask[i] = 0
    # -------------------------------------

    obj_logits = end_points['objectness_scores'].squeeze().detach().cpu().numpy() # num_proposal,2
    obj_prob = softmax(obj_logits)[:, 1]  # (num_proposal)
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (K,7) -----------
        pred_mask = np.zeros((K))
        boxes_2d_with_prob = np.zeros((K,5))
        for i in range(K):
            boxes_2d_with_prob[i,0] = np.min(pred_corners_3d_depth[i, :, 0])
            boxes_2d_with_prob[i,1] = np.min(pred_corners_3d_depth[i, :, 1])
            boxes_2d_with_prob[i,2] = np.max(pred_corners_3d_depth[i, :, 0])
            boxes_2d_with_prob[i,3] = np.max(pred_corners_3d_depth[i, :, 1])
            boxes_2d_with_prob[i,4] = obj_prob[i]
        nonempty_box_inds = np.where(nonempty_box_mask == 1)[0]
        pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask == 1,:],
                            config_dict['nms_iou'], config_dict['use_old_type_nms'])
        assert (len(pick) > 0)
        pred_mask[nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (K,7) -----------
        pred_mask = np.zeros((K))
        boxes_3d_with_prob = np.zeros((K, 7))
        for i in range(K):
            boxes_3d_with_prob[i, 0] = np.min(pred_corners_3d_depth[i, :, 0])
            boxes_3d_with_prob[i, 1] = np.min(pred_corners_3d_depth[i, :, 1])
            boxes_3d_with_prob[i, 2] = np.min(pred_corners_3d_depth[i, :, 2])
            boxes_3d_with_prob[i, 3] = np.max(pred_corners_3d_depth[i, :, 0])
            boxes_3d_with_prob[i, 4] = np.max(pred_corners_3d_depth[i, :, 1])
            boxes_3d_with_prob[i, 5] = np.max(pred_corners_3d_depth[i, :, 2])
            boxes_3d_with_prob[i, 6] = obj_prob[i]
        nonempty_box_inds = np.where(nonempty_box_mask == 1)[0]
        pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask == 1, :],
                             config_dict['nms_iou'], config_dict['use_old_type_nms'])
        assert (len(pick) > 0)
        pred_mask[nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (K,8) -----------
        pred_mask = np.zeros((K))
        boxes_3d_with_prob = np.zeros((K, 8))
        for i in range(K):
            boxes_3d_with_prob[i, 0] = np.min(pred_corners_3d_depth[i, :, 0])
            boxes_3d_with_prob[i, 1] = np.min(pred_corners_3d_depth[i, :, 1])
            boxes_3d_with_prob[i, 2] = np.min(pred_corners_3d_depth[i, :, 2])
            boxes_3d_with_prob[i, 3] = np.max(pred_corners_3d_depth[i, :, 0])
            boxes_3d_with_prob[i, 4] = np.max(pred_corners_3d_depth[i, :, 1])
            boxes_3d_with_prob[i, 5] = np.max(pred_corners_3d_depth[i, :, 2])
            boxes_3d_with_prob[i, 6] = obj_prob[i]
            boxes_3d_with_prob[i, 7] = pred_sem_cls[i]  # only suppress if the two boxes are of the same class!!
        nonempty_box_inds = np.where(nonempty_box_mask == 1)[0]
        pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask == 1, :],
                                     config_dict['nms_iou'], config_dict['use_old_type_nms'])
        assert (len(pick) > 0)
        pred_mask[nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (K) -----------

    pred_bboxes = []  # a list (len: num of predictions) of pred_box_params
    conf_scores = []
    for i in range(K):
        # print("thresh",config_dict['obj_conf_thresh']*normalized_counts[pred_sem_cls_max_indices[i]])
        if choose==0:
            threshold_new=config_dict['obj_conf_thresh']
        elif choose == 1:
            threshold_new=config_dict['obj_conf_thresh']*normalized_counts[pred_sem_cls_max_indices[i]]
        elif choose == 2:
            threshold_new = config_dict['obj_conf_thresh'][pred_sem_cls_max_indices[i]+1]
        if pred_mask[i] == 1 and obj_prob[i] > threshold_new and \
           pred_sem_cls_probs[i] > config_dict['cls_conf_thresh']:
            bbox_param = np.zeros((8))
            bbox_param[0:3] = pred_center[i]
            bbox_param[3:6] = pred_size[i]
            bbox_param[6] = pred_heading_angle[i]
            bbox_param[7] = pred_sem_cls[i]
            pred_bboxes.append(bbox_param)
            conf_scores.append(obj_prob[i]*pred_sem_cls_probs[i])

    if len(pred_bboxes) == 0:
        return None
    else:
        idx = np.argsort(-1 * np.array(conf_scores))
        pred_bboxes = np.stack(pred_bboxes, axis=0)
        pred_bboxes = pred_bboxes[idx]
        # print('pred_bboxes', pred_bboxes)
        return pred_bboxes


def parse_predictions(end_points, config_dict):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points['center'] # B,num_proposal,3
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2,
        pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points['sem_cls_scores'], -1) # B,num_proposal
    pred_size_residual = end_points['size_residuals']  # B,num_proposal,3
    sem_cls_probs = softmax(end_points['sem_cls_scores'].detach().cpu().numpy()) # B,num_proposal,10

    num_proposal = pred_center.shape[1] 
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = class2angle(pred_heading_class[i,j].detach().cpu().numpy(),
                                        pred_heading_residual[i,j].detach().cpu().numpy(),
                                        config_dict['dataset_config'])
            box_size = class2size(int(pred_sem_cls[i,j].detach().cpu().numpy()),
                                  pred_size_residual[i,j].detach().cpu().numpy(),
                                  config_dict['dataset_config'])
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i,j,:])
            pred_corners_3d_upright_camera[i,j] = corners_3d_upright_camera

    K = pred_center.shape[1] # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'].cpu().numpy()[:,:,0:3] # B,N,3
        for i in range(bsize):
            pc = batch_pc[i,:,:] # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i,j,:,:] # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box,inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i,j] = 0
        # -------------------------------------

    obj_logits = end_points['objectness_scores'].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:,:,1] # (B,K)
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K,5))
            for j in range(K):
                boxes_2d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_2d_with_prob[j,2] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_2d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_2d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_2d_with_prob[j,4] = obj_prob[i,j]
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K,7))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K,8))
            for j in range(K):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d_upright_camera[i,j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d_upright_camera[i,j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d_upright_camera[i,j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob[i,j]
                boxes_3d_with_prob[j,7] = pred_sem_cls[i,j] # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i,:]==1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i,:]==1,:],
                config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = [] # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class_final):
                cur_list += [(ii, pred_corners_3d_upright_camera[i,j], sem_cls_probs[i,j,ii]*obj_prob[i,j]) \
                    for j in range(pred_center.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']]
            # for ii in range(config_dict['dataset_config'].num_class_final):
            #     for j in range(pred_center.shape[1]):
            #         if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']:
            #             cur_list += [(ii, pred_corners_3d_upright_camera[i,j], sem_cls_probs[i,j,ii]*obj_prob[i,j])]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i,j].item(), pred_corners_3d_upright_camera[i,j], obj_prob[i,j]) \
                for j in range(pred_center.shape[1]) if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']])
            # for j in range(pred_center.shape[1]):
            #     if pred_mask[i,j]==1 and obj_prob[i,j]>config_dict['conf_thresh']:
            #         batch_pred_map_cls.append([(pred_sem_cls[i,j].item(), pred_corners_3d_upright_camera[i,j], obj_prob[i,j])])
    end_points['batch_pred_map_cls'] = batch_pred_map_cls

    return batch_pred_map_cls

def parse_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label=sem_cls_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    # size_class_label = end_points['size_class_label']
    size_residual_label = end_points['size_residual_label']
    box_label_mask = end_points['box_label_mask']
    sem_cls_label = end_points['sem_cls_label']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1] # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:,:,0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i,j] == 0: continue
            heading_angle = class2angle(heading_class_label[i,j].detach().cpu().numpy(),
                                        heading_residual_label[i,j].detach().cpu().numpy(),
                                        config_dict['dataset_config'])
            box_size = class2size(int(sem_cls_label[i,j].detach().cpu().numpy()),
                                  size_residual_label[i,j].detach().cpu().numpy(),
                                  config_dict['dataset_config'])
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i,j,:])
            gt_corners_3d_upright_camera[i,j] = corners_3d_upright_camera

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(sem_cls_label[i,j].item(), gt_corners_3d_upright_camera[i,j]) for j in range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i,j]==1])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls

class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        print(bsize)
        print(len(batch_gt_map_cls))
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        ret_dict = {} 
        # 36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34
        # self.class2type_map={36: 'bathtub', 4: 'bed', 10: 'bookshelf', 3: 'cabinet', 5: 'chair', 12: 'counter', 16: 'curtain', 14: 'desk', 8: 'door', 39: 'otherfurniture', 11: 'picture', 24: 'refrigerator', 28: 'showercurtain', 34: 'sink', 6: 'sofa' , 7: 'table', 33: 'toilet', 9: 'window'}
        for key in sorted(ap.keys()):
            # print(self.class2type_map)
            if key in self.class2type_map:
                clsname = self.class2type_map[key]
            else:
                clsname = str(key)
            # clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision'%(clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            if key in self.class2type_map:
                clsname = self.class2type_map[key]
            else:
                clsname = str(key)
            # clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall'%(clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
