import os
import numpy as np
import pickle
from easydict import EasyDict
import random
random.seed(10)
from collections import defaultdict
__C = EasyDict()
cfg = __C

__C.NUM_CLASSES = 10
__C.TYPE_WHITELIST = ['bathtub', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'night_stand', 'sofa', 'table', 'toilet']
__C.CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
__C.MAX_NUM_POINT = 50000
__C.USE_V1 = True
__C.SKIP_EMPTY_SCENE = True

__C.TYPE_MEAN_SIZE = {'bathtub': np.array([0.765840,1.398258,0.472728]),
                      'bed': np.array([2.114256,1.620300,0.927272]),
                      'bookshelf': np.array([0.404671,1.071108,1.688889]),
                      'chair': np.array([0.591958,0.552978,0.827272]),
                      'desk': np.array([0.695190,1.346299,0.736364]),
                      'dresser': np.array([0.528526,1.002642,1.172878]),
                      'night_stand': np.array([0.500618,0.632163,0.683424]),
                      'sofa': np.array([0.923508,1.867419,0.845495]),
                      'table': np.array([0.791118,1.279516,0.718182]),
                      'toilet': np.array([0.699104,0.454178,0.756250])
                     }

__C.NUM_HEADING_BIN = 12

__C.MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
__C.MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # sunrgbd color is in 0~1

__C.NUM_BASE_CLASSES = 7
__C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'night_stand']
__C.BASE_CLASSES = [0, 1, 2, 3, 4, 5, 6]

__C.NUM_NOVEL_CLASSES = 10
__C.NOVEL_TYPES = [ 'bathtub', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'night_stand','sofa', 'table', 'toilet']
__C.NOVEL_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def transform_data(data):
    new_data = {}
    for key, values in data.items():
        for value in values:
            if value in new_data:
                new_data[value][key] = 1
            else:
                new_data[value] = {k: 0 for k in data.keys()}
                new_data[value][key] = 1
    return new_data
def select_random_samples(data, percent_to_select):
    all_samples = list(data.keys())
    selected_samples = random.sample(all_samples, int(len(all_samples) * percent_to_select))
    return {sample: data[sample] for sample in selected_samples}

# 函数：计算每个键对应的值的总和
def calculate_totals(selected_data):
    totals = defaultdict(int)
    for sample_data in selected_data.values():
        for key, value in sample_data.items():
            totals[key] += value
    return totals
def get_class2scans(data_path, split='train'):
    '''Generate a mapping dictionary whose key is the class name and the values are the corresponding scan names
       containing objects of this class
    '''
    index_data_path = os.path.join(data_path, 'index_data')
    class2scans_file = os.path.join(index_data_path, '%s_class2scans.pkl' %split)
    if not os.path.exists(index_data_path): os.mkdir(index_data_path)

    if os.path.exists(class2scans_file):
        with open(class2scans_file, 'rb') as f:
            class2scans = pickle.load(f)
            # print(class2scans)
            # # # 4895
            # # print(len(class2scans['bathtub'])+len(class2scans['bed'])+len(class2scans['bookshelf'])+len(class2scans['chair'])+len(class2scans['desk'])+len(class2scans['dresser'])+len(class2scans['night_stand']))
            # # # 4485
            # # print(len(class2scans['bathtub'])+len(class2scans['bed'])+len(class2scans['bookshelf'])+len(class2scans['chair'])+len(class2scans['desk']))
            # # for x in class2scans:
            # #     print(x)
            # #     print(x.value)
            # #选场景，读取csv

            # #选物体，读取csv
            # keys_to_process = ['bathtub', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'night_stand']
            # new_data = {}
            # for key in keys_to_process:
            #     original_list = class2scans[key]
            #     sample_size = int(0.05 * len(original_list))#参数化比率
            #     random_sample = random.sample(original_list, sample_size)#固定随机种子
            #     new_data[key] = random_sample
            # for key in new_data:
            #     class2scans[key] = new_data[key]
            # # print(len(class2scans['bathtub'])+len(class2scans['bed'])+len(class2scans['bookshelf'])+len(class2scans['chair'])+len(class2scans['desk'])+len(class2scans['dresser'])+len(class2scans['night_stand']))
            # # print(len(class2scans['sofa'  ])+len(class2scans['table'])+len(class2scans['toilet']))
            # # print(class2scans)
            selected_data = {}
            while True:
                # 步骤1：将数据转换成新字典的格式
                new_data = transform_data(class2scans)
                
                # 清空选择的数据
                selected_data = {}
                
                # 步骤2：随机选择5%的数据
                selected_data = select_random_samples(new_data, 0.1)
                
                # 步骤3：计算每个键对应的值的总和
                totals = calculate_totals(selected_data)
                
                # 步骤4：检查是否满足条件（每个键对应的值都大于0）
                if all(value > 0 for value in totals.values()):
                    break

            # 打印每个键对应的值的总和
            print(selected_data)
            # 初始化一个空的新字典，用于存储转换后的数据
            converted_data = {key: [] for key in class2scans.keys()}

            # 遍历最终生成的字典
            for sample, values in selected_data.items():
                # 遍历每个键
                for key, value in values.items():
                    # 如果值为1，将该样本添加到相应的键中
                    if value == 1:
                        converted_data[key].append(sample)

            # 打印转换后的数据
            print(converted_data)        
            with open("0.1sunrgbd_scene", 'wb') as f:
                pickle.dump(converted_data, f, pickle.HIGHEST_PROTOCOL)
            for class_name in __C.TYPE_WHITELIST:
                print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(converted_data[class_name])))


    else:
        class2scans = {c: [] for c in __C.TYPE_WHITELIST}
        scan_data_path = os.path.join(data_path, 'sunrgbd_%s_pc_bbox_50k_%s' %('v1' if __C.USE_V1 else 'v2', split))
        print(scan_data_path)
        all_scan_names = list(set([os.path.basename(x)[0:6] for x in os.listdir(scan_data_path)]))
        for scan_name in all_scan_names:
            bboxes = np.load(os.path.join(scan_data_path, scan_name)+'_bbox.npy')
            label_ids = bboxes[:,-1]
            unique_label_ids = np.unique(label_ids)
            for label_id in unique_label_ids:
                class_name = __C.TYPE_WHITELIST[int(label_id)]
                class2scans[class_name].append(scan_name)

        print('==== Split: %s | class to scans mapping is done ====' %split)
        for class_name in __C.TYPE_WHITELIST:
            print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))

        #save class2scans to file...
        with open(class2scans_file, 'wb') as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
    return class2scans


if __name__ == '__main__':
    get_class2scans('../sunrgbd/', split='val')
