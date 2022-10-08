import os
from os import path
import glob
import argparse
import json
import cv2

def kaze_extract(data_file_path = None, extended = True, upright = False, threshold = 0.001, nOctaves = 4, 
                    nOctaveLayers = 4, target_path = None):
    if data_file_path == None:
        print("data_file_path is required")
    elif target_path == None:
        print("target_path is required")
    
    else:
        path = target_path.format(f"_{extended}_{upright}_{threshold}_{nOctaves}_{nOctaveLayers}")
        main_dict = {
            "name" : path,
            "extended" : extended,
            "upright" : upright,
            "threshold" : threshold,
            "nOctaves" : nOctaves,
            "nOctaveLayers" : nOctaveLayers,
            "Group 1 - Normal" : [],
            "Group 2 - Abnormal" : []
        }
        kaze = cv2.KAZE_create(extended = extended,
                                upright = upright,
                                threshold = threshold,
                                nOctaves = nOctaves,
                                nOctaveLayers = nOctaveLayers)
        for x in data_file_path:
            split = x.split("\\")
            split_check = split[-1].split(".")[0].split("_")
            if "ROI" not in split_check and "018" not in split_check:
                print(split[-1])
                img = cv2.imread(x)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = kaze.detectAndCompute(gray, None)
                main_dict[split[3]].append({split[-1] : des.tolist()})
        out_file = open(path, "w")
        json.dump(main_dict, out_file, indent = 6)
        out_file.close()
                

def test(img_paths):
    sum = 0
    max = 0
    for x in img_paths:
        split = x.split("\\")
        if "ROI" not in split[-1].split(".")[0].split("_"):
            img = cv2.imread(x)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            kaze = cv2.KAZE_create(nOctaves = 10)
            kp, des = kaze.detectAndCompute(gray, None)
            print(split, kp, des)
            if len(kp) > max:
                max = len(kp) 
            # print(len(des[0])) # 128
            # print(type(kp)) # cv.KeyPoints
            # print(type(des)) # np.array
            sum += len(kp)
    return (sum/len(img_paths), max)

if __name__ == '__main__':
    print(os.getcwd())
    os.chdir("..")
    main_data_dir = os.getcwd() + "/Data set"
    
    origin_data_dir = main_data_dir + "/Original Form"
    train_origin_data_dir = origin_data_dir + "/Train"
    test_origin_data_dir = origin_data_dir + "/Test"
    train_origin_data_files = glob.glob(train_origin_data_dir + "/*/*")
    test_origin_data_files = glob.glob(test_origin_data_dir + "/*/*")
    
    train_json_output = main_data_dir + "/kaze_train{}.json"
    test_json_output = main_data_dir + "/kaze_test{}.json"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ori_train_files', type=list, default=train_origin_data_files, help='img and mask in train set')
    
    parser.add_argument('--ori_test_files', type=list, default=test_origin_data_files, help='img and mask in test set')
    
    parser.add_argument('--target_kaze_json_train_file', type=str, default=train_json_output, help='output kaze json train file')
    
    parser.add_argument('--target_kaze_json_test_file', type=str, default=test_json_output, help='output kaze json test file')

    opt = parser.parse_args()
    
    print("Example of original training img path: {}".format(opt.ori_train_files[0]))
    print("Example of original testing img path: {}".format(opt.ori_test_files[1]))
    print("KAZE Json Train file: {}".format(opt.target_kaze_json_train_file))
    print("KAZE Json Test file: {}".format(opt.target_kaze_json_test_file))

    # print(test(opt.ori_train_files[:100])) # (201.29, 2257)
    # print(test(opt.ori_train_files + opt.ori_test_files)) # (137.4488636363636, 2582)
    # print(test([opt.ori_train_files[12]]))

    kaze_extract(data_file_path=opt.ori_train_files, target_path=opt.target_kaze_json_train_file)
    kaze_extract(data_file_path=opt.ori_test_files, target_path=opt.target_kaze_json_test_file)