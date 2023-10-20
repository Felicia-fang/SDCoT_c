
import os
import sys
import random
import numpy as np
import argparse
import torch
import vtkmodules.all as vtk

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from model import create_detection_model, load_detection_model
import vis_utils as vis
from ap_helper import parse_predictions, flip_axis_to_depth
from box_util import get_3d_box_depth


def detect_3d_objects(net, point_cloud, eval_config_dict):
    net.eval()
    with torch.no_grad():
        end_points = net(point_cloud)
    end_points['point_clouds'] = point_cloud
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    num_pred_obj = len(pred_map_cls[0])
    print('Finished detection. %d object detected.' % (num_pred_obj))
    pred_box3d_list = []
    for i in range(num_pred_obj):
        pred_box3d_list.append(flip_axis_to_depth(pred_map_cls[0][i][1]))
    # print(pred_box3d_list)
    return end_points, pred_box3d_list


def visualize(args, split, scene_name=None):

    # =========================== Init Dataset ===========================
    if args.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        import sunrgbd_utils
        from sunrgbd import SunrgbdBaseDatasetConfig, SunrgbdNovelDatasetConfig, SunrgbdDataset
        from sunrgbd_cfg import cfg
        base_model_config = SunrgbdBaseDatasetConfig()
        novel_model_config = SunrgbdNovelDatasetConfig(cfg.NUM_NOVEL_CLASSES)
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
        novel_model_config = ScannetNovelDatasetConfig(cfg.NUM_NOVEL_CLASSES)
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

    # ==================================== Init Model ====================================
    base_model = create_detection_model(args, base_model_config)
    novel_model = create_detection_model(args, novel_model_config)

    # initialize detection model with checkpoint from first-stage training
    if args.base_model_checkpoint_path is not None:
    # if args.base_model_checkpoint_path is not None and args.novel_model_checkpoint_path is not None:
        base_model = load_detection_model(base_model, args.base_model_checkpoint_path)
        # novel_model = load_detection_model(novel_model, args.novel_model_checkpoint_path)
    else:
        raise ValueError('Base and Novel detection model checkpoint path must be given!')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    novel_model.to(device)

    ##  ==================================== Visualize ====================================
    if args.dataset == 'sunrgbd':
        # original_point_cloud = np.load(os.path.join(DATA_PATH, scene_name) + '_pc.npz')['pc']  # Nx6
        depth_file = os.path.join(
            '/data2/wufang/sunrgbd_data/sunrgbd/sunrgbd_trainval/depth',
            '%06d.mat' % int(scene_name))
        original_point_cloud = sunrgbd_utils.load_depth_points_mat(depth_file)
        original_point_cloud[:, 3:6] *= 255.

        gt_bboxes = np.load(os.path.join(DATA_PATH, '%06d_bbox.npy' % int(scene_name)))
        # bbox_mask_novel = np.in1d(gt_bboxes[:, -1], cfg.NOVEL_CLASSES)
        bbox_mask_base = np.in1d(gt_bboxes[:, -1], cfg.BASE_CLASSES)
        # gt_bboxes_novel = gt_bboxes[bbox_mask_novel, :]
        gt_bboxes_base = gt_bboxes[bbox_mask_base, :]
    else:
        # original_point_cloud = np.load(os.path.join(DATA_PATH, scene_name) + '_vert.npy')

        data_path = '/data2/wufang/SDCoT/scannet/scans/'
        mesh_file = os.path.join(data_path, scene_name, scene_name + '_vh_clean.ply') #dense
        data_path = '/data2/wufang/SDCoT/scannet/scans/'
        # mesh_file = os.path.join(data_path, scene_name, scene_name + '_vh_clean_2.ply')
        meta_file = os.path.join(data_path, scene_name, scene_name + '.txt')
        mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)
        original_point_cloud = scannet_utils.align_pointcloud_to_axis(meta_file, mesh_vertices)

        gt_bboxes = np.load(os.path.join(DATA_PATH, scene_name) + '_bbox.npy')
        # bbox_mask_novel = np.in1d(gt_bboxes[:, -1], cfg.NOVEL_NYUIDS)
        bbox_mask_base = np.in1d(gt_bboxes[:, -1], cfg.BASE_NYUIDS)
        # gt_bboxes_novel = gt_bboxes[bbox_mask_novel, :]
        # print(gt_bboxes_novel)
        gt_bboxes_base = gt_bboxes[bbox_mask_base, :]
        # print(gt_bboxes_base)

    point_cloud = DATASET._process_pointcloud(original_point_cloud)
    point_cloud, _ = DATASET._sample_pointcloud(point_cloud)

    input_point_cloud = torch.from_numpy(point_cloud).to(device).unsqueeze(0)
    base_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
                        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
                        'conf_thresh': args.pseudo_obj_conf_thresh, 'dataset_config': base_model_config}


    base_end_points, pred_base_box3d_list = detect_3d_objects(base_model, input_point_cloud, base_config_dict)
    # novel_end_points, pred_novel_box3d_list = detect_3d_objects(novel_model, input_point_cloud, novel_config_dict)
    print(pred_base_box3d_list)
    # print(pred_base_box3d_list)
    print(11111111111111111111111111111111)

    pred_base_votes = base_end_points['vote_xyz'].squeeze(0).detach().cpu().numpy()  # (num_vote, 3)
    # pred_novel_votes = novel_end_points['vote_xyz'].squeeze(0).detach().cpu().numpy()  # (num_vote, 3)

    # gt_novel_box3d_list = []
    # for i in range(gt_bboxes_novel.shape[0]):
    #     box = gt_bboxes_novel[i]
    #     if args.dataset == 'sunrgbd':
    #         box3d = get_3d_box_depth(box[3:6]*2, box[6], box[0:3])
    #     else:
    #         box3d = get_3d_box_depth(box[3:6], 0, box[0:3])
    #     gt_novel_box3d_list.append(box3d)

    gt_base_box3d_list = []
    for i in range(gt_bboxes_base.shape[0]):
        box = gt_bboxes_base[i]
        if args.dataset == 'sunrgbd':
            box3d = get_3d_box_depth(box[3:6]*2, box[6], box[0:3])
        else:
            box3d = get_3d_box_depth(box[3:6], 0, box[0:3])
        gt_base_box3d_list.append(box3d)
    print(gt_bboxes_base.shape[0])
    print(gt_base_box3d_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='scannet', help='Dataset name: scannet|sunrgbd')
    parser.add_argument('--base_model_checkpoint_path', default='/data2/wufang/SDCoT/log_scannet/log_basetrain_20231017-15:37',
                        help='Detection model checkpoint path [default: None]') #log_sunrgbd/log_basetrain_20230912-12:31
    # parser.add_argument('--novel_model_checkpoint_path', default='/data2/wufang/SDCoT/log_scannet/log_SDCoT_20230913-20:41',
    #                     help='Detection model checkpoint path [default: None]') #log_sunrgbd/log_SDCoT_20230912-22:59
    # parser.add_argument('--base_model_checkpoint_path', default='/data2/wufang/SDCoT/log_sunrgbd/log_basetrain_20230912-12:31',
    #                     help='Detection model checkpoint path [default: None]') #log_sunrgbd/log_basetrain_20230912-12:31
    # parser.add_argument('--novel_model_checkpoint_path', default='/data2/wufang/SDCoT/log_sunrgbd/log_SDCoT_20230912-22:59',
    #                     help='Detection model checkpoint path [default: None]') #log_sunrgbd/log_SDCoT_20230912-22:59
    parser.add_argument('--num_target', type=int, default=128, help='Proposal number [default: 256]')
    parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
    parser.add_argument('--cluster_sampling', default='vote_fps',
                        help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')

    parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 20000]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')

    parser.add_argument('--pseudo_obj_conf_thresh', type=float, default=0.0,
                        help='Confidence score threshold w.r.t. objectness prediction for hard selection of psuedo bboxes')
    parser.add_argument('--pseudo_cls_conf_thresh', type=float, default=0.0,
                        help='Confidence score threshold w.r.t. class prediction for hard selection of psuedo bboxes')
    args = parser.parse_args()

    args.num_input_channel = int(args.use_color) * 3 + int(not args.no_height) * 1

    # visualize(args, split='train', scene_name='005051')
    visualize(args, split='train', scene_name='scene0000_00')