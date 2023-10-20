# ''' Global configurations of ScanNet dataset.

# Author: Zhao Na
# Data: September, 2020
# '''

# import os
# import numpy as np
# import pickle
# from easydict import EasyDict

# __C = EasyDict()
# cfg = __C

# __C.NUM_CLASSES = 18
# __C.DONOTCARE_CLASS_IDS = np.array([])
# __C.NYU40IDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34, 6, 7, 33, 9]) #the corresponding NYU40 ids of interested object class
# __C.TYPE_WHITELIST = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'otherfurniture',
#                       'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']
# __C.MAX_NUM_POINT = 50000

# __C.TYPE_MEAN_SIZE =  {'cabinet': np.array([0.76966726, 0.81160211, 0.92573741]),
#                        'bed': np.array([1.876858, 1.84255952, 1.19315654]),
#                        'chair': np.array([0.61327999, 0.61486087, 0.71827014]),
#                        'sofa': np.array([1.39550063, 1.51215451, 0.83443565]),
#                        'table': np.array([0.97949596, 1.06751485, 0.63296875]),
#                        'door': np.array([0.53166301, 0.59555772, 1.75001483]),
#                        'window': np.array([0.96247056, 0.72462326, 1.14818682]),
#                        'bookshelf': np.array([0.83221924, 1.04909355, 1.68756634]),
#                        'picture': np.array([0.21132214, 0.4206159 , 0.53728459]),
#                        'counter': np.array([1.44400728, 1.89708334, 0.26985747]),
#                        'desk': np.array([1.02942616, 1.40407966, 0.87554322]),
#                        'curtain': np.array([1.37664116, 0.65521793, 1.68131292]),
#                        'refrigerator': np.array([0.66508189, 0.71111926, 1.29885307]),
#                        'showercurtain': np.array([0.41999174, 0.37906947, 1.75139715]),
#                        'toilet': np.array([0.59359559, 0.59124924, 0.73919014]),
#                        'sink': np.array([0.50867595, 0.50656087, 0.30136236]),
#                        'bathtub': np.array([1.15115265, 1.0546296 , 0.49706794]),
#                        'otherfurniture': np.array([0.47535286, 0.49249493, 0.58021168])
#                         }

# __C.NUM_HEADING_BIN = 1 #Object bboxes are alix-aligned in ScanNet

# __C.MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
# __C.MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# __C.NUM_BASE_CLASSES = 14
# __C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door','otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink']
# __C.BASE_NYUIDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34,])

# __C.NUM_NOVEL_CLASSES = 4
# __C.NOVEL_TYPES = [ 'sofa', 'table','toilet', 'window']
# __C.NOVEL_NYUIDS = np.array([ 33, 9,6,7])

# # __C.NUM_BASE_CLASSES = 9
# # __C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door']
# # __C.BASE_NYUIDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8])

# # __C.NUM_NOVEL_CLASSES = 9
# # __C.NOVEL_TYPES = ['otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']
# # __C.NOVEL_NYUIDS = np.array([39, 11, 24, 28, 34, 6, 7, 33, 9])

# def get_class2scans(data_path, split='train'):
#     '''Generate a mapping dictionary whose key is the class name and the values are the corresponding scan names
#        containing objects of this class
#     '''
#     index_data_path = os.path.join(data_path, 'index_data')
#     class2scans_file = os.path.join(index_data_path, '%s_class2scans.pkl' %split)
#     if not os.path.exists(index_data_path): os.mkdir(index_data_path)

#     if os.path.exists(class2scans_file):
#         with open(class2scans_file, 'rb') as f:
#             class2scans = pickle.load(f)
#     else:
#         class2scans = {c: [] for c in __C.TYPE_WHITELIST}
#         all_scan_names = list(set([os.path.basename(x)[0:12] \
#                                    for x in os.listdir(os.path.join(data_path, 'scannet_%s_detection_data' %split)) \
#                                    if x.startswith('scene')]))
#         for scan_name in all_scan_names:
#             bboxes = np.load(os.path.join(data_path, 'scannet_%s_detection_data' %split, scan_name)+'_bbox.npy')
#             label_ids = bboxes[:,-1]
#             unique_label_ids = np.unique(label_ids)
#             for nyuid in unique_label_ids:
#                 nyuid = int(nyuid)
#                 if nyuid in __C.NYU40IDS:
#                     class_name = __C.TYPE_WHITELIST[np.where(__C.NYU40IDS==nyuid)[0][0]]
#                     class2scans[class_name].append(scan_name)

#         print('==== Split: %s | class to scans mapping is done ====' %split)
#         for class_name in __C.TYPE_WHITELIST:
#             print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))

#         #save class2scans to file...
#         with open(class2scans_file, 'wb') as f:
#             pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
#     return class2scans

# if __name__ == '__main__':
#     get_class2scans('../scannet/', split='val')
import os
import numpy as np
import pickle
from easydict import EasyDict
import random
random.seed(10 )
import csv
__C = EasyDict()
cfg = __C

__C.NUM_CLASSES = 18
__C.DONOTCARE_CLASS_IDS = np.array([])
__C.NYU40IDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34, 6, 7, 33, 9]) #the corresponding NYU40 ids of interested object class
__C.TYPE_WHITELIST = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'otherfurniture',
                      'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']
__C.MAX_NUM_POINT = 50000

__C.TYPE_MEAN_SIZE =  {'cabinet': np.array([0.76966726, 0.81160211, 0.92573741]),
                       'bed': np.array([1.876858, 1.84255952, 1.19315654]),
                       'chair': np.array([0.61327999, 0.61486087, 0.71827014]),
                       'sofa': np.array([1.39550063, 1.51215451, 0.83443565]),
                       'table': np.array([0.97949596, 1.06751485, 0.63296875]),
                       'door': np.array([0.53166301, 0.59555772, 1.75001483]),
                       'window': np.array([0.96247056, 0.72462326, 1.14818682]),
                       'bookshelf': np.array([0.83221924, 1.04909355, 1.68756634]),
                       'picture': np.array([0.21132214, 0.4206159 , 0.53728459]),
                       'counter': np.array([1.44400728, 1.89708334, 0.26985747]),
                       'desk': np.array([1.02942616, 1.40407966, 0.87554322]),
                       'curtain': np.array([1.37664116, 0.65521793, 1.68131292]),
                       'refrigerator': np.array([0.66508189, 0.71111926, 1.29885307]),
                       'showercurtain': np.array([0.41999174, 0.37906947, 1.75139715]),
                       'toilet': np.array([0.59359559, 0.59124924, 0.73919014]),
                       'sink': np.array([0.50867595, 0.50656087, 0.30136236]),
                       'bathtub': np.array([1.15115265, 1.0546296 , 0.49706794]),
                       'otherfurniture': np.array([0.47535286, 0.49249493, 0.58021168])
                        }

__C.NUM_HEADING_BIN = 1 #Object bboxes are alix-aligned in ScanNet

__C.MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
__C.MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

__C.NUM_BASE_CLASSES = 14
__C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door','otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink']
__C.BASE_NYUIDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34,])
# __C.NUM_BASE_CLASSES = 18
# __C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door','otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink','sofa', 'table', 'toilet', 'window']
# __C.BASE_NYUIDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34,6, 7, 33, 9])
__C.NUM_NOVEL_CLASSES = 18
__C.NOVEL_TYPES = [ 'sofa', 'table','toilet', 'window','bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door','otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink']
__C.NOVEL_NYUIDS = np.array([ 33, 9,6,7,36, 4, 10, 3, 5, 12, 16, 14, 8, 39, 11, 24, 28, 34])

# __C.NUM_BASE_CLASSES = 9
# __C.BASE_TYPES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door']
# __C.BASE_NYUIDS = np.array([36, 4, 10, 3, 5, 12, 16, 14, 8])

# __C.NUM_NOVEL_CLASSES = 9
# __C.NOVEL_TYPES = ['otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']
# __C.NOVEL_NYUIDS = np.array([39, 11, 24, 28, 34, 6, 7, 33, 9])

def get_class2scans(data_path, split='train'):
    '''Generate a mapping dictionary whose key is the class name and the values are the corresponding scan names
       containing objects of this class
    '''
    index_data_path = os.path.join(data_path, 'index_data')
    class2scans_file = os.path.join(index_data_path, '%s_class2scans.pkl' %split)
    if not os.path.exists(index_data_path): os.mkdir(index_data_path)
    if os.path.exists("0.05sacannet_scene"):
        print("scene")
        with open("0.05scannet_scene", 'rb') as f:
            class2scans = pickle.load(f)
    elif os.path.exists(class2scans_file):
        print("class")
        with open(class2scans_file, 'rb') as f:
            class2scans = pickle.load(f)
            # print(class2scans)
            keys_to_process = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door','otherfurniture', 'picture', 'refrigerator', 'showercurtain', 'sink']
            new_data = {}
            for key in keys_to_process:
                original_list = class2scans[key]
                sample_size = int(0.05 * len(original_list))
                random_sample = random.sample(original_list, sample_size)
                new_data[key] = random_sample
            for key in new_data:
                class2scans[key] = new_data[key]
            for class_name in __C.TYPE_WHITELIST:
                print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))
            csv_filename = '0.05scannet2.csv'
            with open(csv_filename, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
    
                header = class2scans.keys()
                writer.writerow(header)
    
                max_data_length = max(len(class2scans[key]) for key in class2scans.keys())
                for i in range(max_data_length):
                    row = [class2scans[key][i] if i < len(class2scans[key]) else '' for key in header]
                    writer.writerow(row)
            # print(class2scans)
    else:
        class2scans = {c: [] for c in __C.TYPE_WHITELIST}
        all_scan_names = list(set([os.path.basename(x)[0:12] \
                                   for x in os.listdir(os.path.join(data_path, 'scannet_%s_detection_data' %split)) \
                                   if x.startswith('scene')]))
        for scan_name in all_scan_names:
            bboxes = np.load(os.path.join(data_path, 'scannet_%s_detection_data' %split, scan_name)+'_bbox.npy')
            label_ids = bboxes[:,-1]
            unique_label_ids = np.unique(label_ids)
            for nyuid in unique_label_ids:
                nyuid = int(nyuid)
                if nyuid in __C.NYU40IDS:
                    class_name = __C.TYPE_WHITELIST[np.where(__C.NYU40IDS==nyuid)[0][0]]
                    class2scans[class_name].append(scan_name)

        print('==== Split: %s | class to scans mapping is done ====' %split)
        for class_name in __C.TYPE_WHITELIST:
            print('\t class_name: {0} | num of scans: {1}'.format(class_name, len(class2scans[class_name])))

        #save class2scans to file...
        with open(class2scans_file, 'wb') as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
    return class2scans

if __name__ == '__main__':
    get_class2scans('../scannet/', split='val')
