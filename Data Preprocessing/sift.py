import os
from os import path
import glob
import shutil
import argparse
import json
import cv2

def sift_extract(data_file_path = None, _nfeatures = 300, _nOctaveLayers = 3, _contrastThreshold = 0.04, _edgeThreshold = 10, _sigma = 1.6, target_path = None):
    if data_file_path == None:
        print("data_file_path is required")
    elif target_path == None:
        print("target_path is required")
    
    else:
        path = target_path.format(f"_{_nfeatures}_{_nOctaveLayers}_{_contrastThreshold}_{_edgeThreshold}_{_sigma}")
        main_dict = {
            "name" : path,
            "nfeatures" : _nfeatures,
            "nOctaveLayers" : _nOctaveLayers,
            "contrastThreshold" : _contrastThreshold,
            "edgeThreshold" : _edgeThreshold, 
            "Group 1 - Normal" : [],
            "Group 2 - Abnormal" : []
        }
        sift = cv2.SIFT_create(nfeatures = _nfeatures,
                                nOctaveLayers = _nOctaveLayers,
                                contrastThreshold = _contrastThreshold,
                                edgeThreshold = _edgeThreshold,
                                sigma = _sigma)
        for x in data_file_path:
            split = x.split("\\")
            if "ROI" not in split[-1].split(".")[0].split("_"):
                img = cv2.imread(x)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = sift.detectAndCompute(gray, None)
                main_dict[split[3]] = {
                    split[-1] : des.tolist()
                }
        out_file = open(path, "w")
        json.dump(main_dict, out_file, indent = 6)
        out_file.close()
                

def test(img_paths):
    sum = 0
    for x in img_paths:
        split = x.split("\\")
        if "ROI" not in split[-1].split(".")[0].split("_"):
            img = cv2.imread(x)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
            # print(kp[0])
            # print(des)
            # print(len(kp))
            # print(len(des))
            # print(len(des[0])) # 128
            # print(type(des)) # np.array
            sum += len(kp)
    return sum/len(img_paths)

if __name__ == '__main__':
    print(os.getcwd())
    main_data_dir = os.getcwd() + "/Data set"
    
    origin_data_dir = main_data_dir + "/Original Form"
    train_origin_data_dir = origin_data_dir + "/Train"
    test_origin_data_dir = origin_data_dir + "/Test"
    train_origin_data_files = glob.glob(train_origin_data_dir + "/*/*")
    test_origin_data_files = glob.glob(test_origin_data_dir + "/*/*")
    
    train_json_output = main_data_dir + "/sift_train{}.json"
    test_json_output = main_data_dir + "/sift_test{}.json"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ori_train_files', type=list, default=train_origin_data_files, help='img and mask in train set')
    
    parser.add_argument('--ori_test_files', type=list, default=test_origin_data_files, help='img and mask in test set')
    
    parser.add_argument('--target_sift_json_train_file', type=str, default=train_json_output, help='output surf json train file')
    
    parser.add_argument('--target_sift_json_test_file', type=str, default=test_json_output, help='output surf json test file')

    opt = parser.parse_args()
    
    print("Example of original training img path: {}".format(opt.ori_train_files[0]))
    print("Example of original testing img path: {}".format(opt.ori_test_files[1]))
    print("SIFT Json Train file: {}".format(opt.target_sift_json_train_file))
    print("SIFT Json Test file: {}".format(opt.target_sift_json_test_file))

    # print(test(opt.ori_train_files[:100])) # 230
    # print(test(opt.ori_train_files + opt.ori_test_files)) # 291
    # sift_extract(data_file_path=opt.ori_train_files, target_path=opt.target_sift_json_train_file)
    # sift_extract(data_file_path=opt.ori_test_files, target_path=opt.target_sift_json_test_file)