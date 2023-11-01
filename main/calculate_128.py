import os
import sys
import random
import numpy as np
import argparse
import torch
import vtkmodules.all as vtk
import csv
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from model import create_detection_model, load_detection_model
import vis_utils as vis
from ap_helper import parse_predictions, flip_axis_to_depth, parse_groundtruths,APCalculator,flip_axis_to_camera
from box_util import get_3d_box_depth


def detect_3d_objects_base(net, point_cloud, eval_config_dict):
    net.eval()
    with torch.no_grad():
        end_points = net(point_cloud)
    end_points['point_clouds'] = point_cloud
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print(pred_map_cls)
    num_pred_obj = len(pred_map_cls[0])
    print('Finished detection. %d object detected.' % (num_pred_obj))
    pred_box3d_list = []
    for i in range(num_pred_obj):
        pred_box3d_list.append(flip_axis_to_depth(pred_map_cls[0][i][1]))
    # print(pred_box3d_list)
    return end_points, pred_box3d_list,pred_map_cls

def visualize(args, split, scene_name=None,object_threshold=0.95):

    # =========================== Init Dataset ===========================
    if args.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        import sunrgbd_utils
        from sunrgbd import SunrgbdBaseDatasetConfig, SunrgbdNovelDatasetConfig, SunrgbdDataset
        from sunrgbd_cfg import cfg
        base_model_config = SunrgbdBaseDatasetConfig()
        DATASET = SunrgbdDataset(num_points=args.num_point, use_color=args.use_color,
                                 use_height=(not args.no_height), augment=False)
        DATA_PATH = os.path.join(ROOT_DIR, 'sunrgbd',
                                 'sunrgbd_%s_pc_bbox_50k_%s' % ('v1' if cfg.USE_V1 else 'v2', split))
        ALL_SCENE_NAMES = list(set([os.path.basename(x)[0:6] for x in os.listdir(DATA_PATH)]))
    elif args.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        import scannet_utils
        from scannet import ScannetBaseDatasetConfig, ScannetNovelDatasetConfig, ScannetDataset
        from scannet_cfg import cfg
        base_model_config = ScannetBaseDatasetConfig()
        DATASET = ScannetDataset(num_points=args.num_point, use_color=args.use_color,
                                 use_height=(not args.no_height), augment=False)
        DATA_PATH = os.path.join(ROOT_DIR, 'scannet', 'scannet_%s_detection_data' % split)

        ALL_SCENE_NAMES = list(set([os.path.basename(x)[0:12] for x in os.listdir(DATA_PATH) if x.startswith('scene')]))
    else:
        print('Unknown dataset %s. Exiting...' % (args.dataset))
        exit(-1)

    if scene_name is None:
        scene_name = random.choice(ALL_SCENE_NAMES)
    print('{0} | {1} | {2}'.format(args.dataset, split, scene_name))

    ##  ==================================== mAp ====================================
    if args.dataset == 'sunrgbd':
        # original_point_cloud = np.load(os.path.join(DATA_PATH, scene_name) + '_pc.npz')['pc']  # Nx6
        depth_file = os.path.join(
            '/data2/wufang/sunrgbd_data/sunrgbd/sunrgbd_trainval/depth',
            '%06d.mat' % int(scene_name))
        original_point_cloud = sunrgbd_utils.load_depth_points_mat(depth_file)
        original_point_cloud[:, 3:6] *= 255.

        gt_bboxes = np.load(os.path.join(DATA_PATH, '%06d_bbox.npy' % int(scene_name)))
        bbox_mask_base = np.in1d(gt_bboxes[:, -1], cfg.BASE_CLASSES)
        gt_bboxes_base = gt_bboxes[bbox_mask_base, :]
    else:
        if os.path.exists(os.path.join(DATA_PATH, scene_name) + '_bbox.npy'):
            train=0
        else:
            return None
        gt_bboxes = np.load(os.path.join(DATA_PATH, scene_name) + '_bbox.npy')
        # print(gt_bboxes)
        bbox_mask_base = np.in1d(gt_bboxes[:, -1], cfg.BASE_NYUIDS)
        # print(bbox_mask_base)
        gt_bboxes_base = gt_bboxes[bbox_mask_base, :]
    print("base")
    gt_base_box3d_list = []
    for i in range(gt_bboxes_base.shape[0]):
        box = gt_bboxes_base[i]
        # gt_base_box3d_list.append(box[6])
        if args.dataset == 'sunrgbd':
            box3d = get_3d_box_depth(box[3:6]*2, box[6], box[0:3])
        else:
            box3d = get_3d_box_depth(box[3:6], 0, box[0:3])
        # gt_base_box3d_list.append(box3d)
        element_to_append = (box[6], box3d)
        gt_base_box3d_list.append(element_to_append)
    print("base bbox",gt_bboxes_base.shape[0])
    # print(gt_base_box3d_list)
    from scannet_val import ScannetValDataset
    test_dataset = ScannetValDataset(all_classes=False,
                                         num_points=40000,
                                         use_color=args.use_color,
                                         use_height=(not args.no_height),
                                         augment=False)
    dataset_config=test_dataset.dataset_config
    ap_calculator = APCalculator(args.ap_iou_threshold, dataset_config.class2type)
    new_gt_base_box3d_list = [gt_base_box3d_list]
    gt_base_box3d_list_all = new_gt_base_box3d_list
    csv_filename = '/data2/wufang/SDCoT/scannet/psu/'+scene_name+'.csv'
    base_pred_map_cls = []
    import ast
    try:
        with open(csv_filename, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)                
            next(csv_reader, None)
            for row in csv_reader:
                class_val = float(row[0])
                data_list = row[1].replace("[", "").replace("]", "").split()
                data_array = np.array(data_list, dtype=float)
                bbox_val = data_array.reshape(-1, 3)
                prob_val = float(row[2])
                # print(prob_val)
                base_pred_map_cls.append((class_val, bbox_val, prob_val))
    except Exception as e:
        print(f"csv'{csv_filename}' faise：{str(e)}")
    base_pred_map_cls=[base_pred_map_cls]




    base_pred_map_cls_thresh = [item for item in base_pred_map_cls[0] if item[-1] > object_threshold]
    base_pred_map_cls_thresh=[base_pred_map_cls_thresh]
    # print(base_pred_map_cls_thresh)
    num_pred_obj = len(base_pred_map_cls_thresh[0])
    print('Finished detection. %d object detected.' % (num_pred_obj))
    pred_box3d_list = []
    pred_box3d_list_draw=[]
    for i in range(num_pred_obj):
        
        concat=(int(base_pred_map_cls_thresh[0][i][0]),base_pred_map_cls_thresh[0][i][1],base_pred_map_cls_thresh[0][i][2])
        pred_box3d_list_draw.append(flip_axis_to_depth(base_pred_map_cls_thresh[0][i][1]))
        # concat=(base_pred_map_cls_thresh[0][i][0],flip_axis_to_depth(base_pred_map_cls_thresh[0][i][1]),base_pred_map_cls_thresh[0][i][2])

        # pred_box3d_list.append(flip_axis_to_depth(base_pred_map_cls_thresh[0][i][1]))
        pred_box3d_list.append(concat)
    # print(gt_base_box3d_list_all[0])
    mapping = {36: 0, 4: 1, 10: 2, 3: 3, 5: 4, 12: 5, 16: 6, 14: 7,
            8: 8, 39: 9, 11: 10, 24:11 ,28:12 ,34:13}

    gt_base_box3d_list = [(mapping[x] if x in mapping else x,y) for x,y in gt_base_box3d_list_all[0]]
    gt_base_box3d_list=[gt_base_box3d_list]
    # print(gt_base_box3d_list)
    pred_box3d_list=[pred_box3d_list]

    gt_box3d_list_draw=[]
    # converse gt 8*3 bbox order
    for i in range(len(gt_base_box3d_list[0])):
        convers=flip_axis_to_camera(gt_base_box3d_list[0][i][1])
        a = convers[:4]
        b = convers[4:]
        con = np.concatenate((b, a), axis=0)
        concat=(gt_base_box3d_list[0][i][0],con)
        gt_box3d_list_draw.append(concat)
    gt_box3d_list_draw=[gt_box3d_list_draw]
    
    ap_calculator.step(pred_box3d_list, gt_box3d_list_draw)

    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        print('eval %s: %f' % (key, metrics_dict[key]))
    # print(pred_map_cls)
    return metrics_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='scannet', help='Dataset name: scannet|sunrgbd')
    parser.add_argument('--base_model_checkpoint_path', default='/data2/wufang/SDCoT/log_scannet/log_basetrain_20230911-02:22',
                        help='Detection model checkpoint path [default: None]') #log_sunrgbd/log_basetrain_20230912-12:31
    parser.add_argument('--novel_model_checkpoint_path', default='/data2/wufang/SDCoT/log_scannet/log_SDCoT_20230913-20:41',
                        help='Detection model checkpoint path [default: None]') #log_sunrgbd/log_SDCoT_20230912-22:59
    parser.add_argument('--num_target', type=int, default=128, help='Proposal number [default: 256]')
    parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
    parser.add_argument('--cluster_sampling', default='vote_fps',
                        help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')

    parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 20000]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')

    parser.add_argument('--pseudo_obj_conf_thresh', type=float, default=0.95,
                        help='Confidence score threshold w.r.t. objectness prediction for hard selection of psuedo bboxes')
    parser.add_argument('--pseudo_cls_conf_thresh', type=float, default=0.9,
                        help='Confidence score threshold w.r.t. class prediction for hard selection of psuedo bboxes')
    parser.add_argument('--ap_iou_threshold', default='0.25', help='AP IoU thresholds')
    args = parser.parse_args()

    args.num_input_channel = int(args.use_color) * 3 + int(not args.no_height) * 1

    DATA_DIR = '/data2/wufang/SDCoT/scannet/psu'

    scene_names = [name.replace('.csv', '') for name in os.listdir(DATA_DIR) if name.endswith('.csv')]
    print(scene_names)
    # threshold from 0.9 to 0.99, interval 0.1
    object_names={'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink'}
    metrics = ['Average Precision', 'Recall']
    metric_list = [
        f"{object_name} {metric}" for object_name in object_names for metric in metrics] + ['mAP', 'AR']
    object_results = {object_name: [] for object_name in metric_list}


    # from threshold 0.9 to 0.99, interval 0.1
    for threshold in range(90, 100):
        object_threshold = threshold / 100
        print("object_threshold", object_threshold)
        
        # initialize a dictionary to store the results for each object
        object_results_for_threshold = {object_name: [] for object_name in metric_list}
        
        for scene_name in scene_names:
            csv_filename = '/data2/wufang/SDCoT/scannet/scans/' + scene_name + '_vh_clean.ply'
            metric = visualize(args, split='train', scene_name=scene_name, object_threshold=object_threshold)
            if metric is None:
                continue
            
            # detect the nan value and remove it
            for object_name, value in metric.items():
                if not np.isnan(value):
                    object_results_for_threshold[object_name].append(value)
        
        # compute the average precision for each object
        for object_name, values in object_results_for_threshold.items():
            if values:
                object_results[object_name].append((object_threshold, np.mean(values)))

        # write the results to a file
        with open('result.txt', 'w') as file:
            # print the average precision for each object
            for object_name, avg_values in object_results.items():
                file.write(f"{object_name} Average Precision for Different Thresholds:\n")
                for threshold, avg_value in avg_values:
                    file.write(f"Threshold {threshold}: {avg_value:.2f}\n")

        print("Results have been saved to 'result.txt'")



        
    # visualize(args, split='train', scene_name='scene0000_00')
