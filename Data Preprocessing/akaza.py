import os
import glob
import argparse
import json
import cv2

def akaze_extract(data_file_path = None, descriptor_size = 0, descriptor_channels = 3, threshold = 0.001, nOctaves = 4, 
                    nOctaveLayers = 4, target_path = None):
    if data_file_path == None:
        print("data_file_path is required")
    elif target_path == None:
        print("target_path is required")
    
    else:
        path = target_path.format(f"_{descriptor_size}_{descriptor_channels}_{threshold}_{nOctaves}_{nOctaveLayers}")
        main_dict = {
            "name" : path,
            "descriptor_size" : descriptor_size,
            "descriptor_channels" : descriptor_channels,
            "threshold" : threshold,
            "nOctaves" : nOctaves,
            "nOctaveLayers" : nOctaveLayers,
            "Group 1 - Normal" : [],
            "Group 2 - Abnormal" : []
        }
        akaze = cv2.AKAZE_create(descriptor_size = descriptor_size,
                                descriptor_channels = descriptor_channels,
                                threshold = threshold,
                                nOctaves = nOctaves,
                                nOctaveLayers = nOctaveLayers)
        for x in data_file_path:
            split = x.split("\\")
            split_check = split[-1].split(".")[0].split("_")
            if "ROI" not in split_check and "018" not in split_check and "043" not in split_check and "044" not in split_check and "071" not in split_check:
                print(split[-1])
                img = cv2.imread(x)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = akaze.detectAndCompute(gray, None)
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
            akaze = cv2.AKAZE_create(nOctaves = 10)
            kp, des = akaze.detectAndCompute(gray, None)
            # print(split, kp, des)
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
    
    train_json_output = main_data_dir + "/akaze/akaze_train{}.json"
    test_json_output = main_data_dir + "/akaze/akaze_test{}.json"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ori_train_files', type=list, default=train_origin_data_files, help='img and mask in train set')
    
    parser.add_argument('--ori_test_files', type=list, default=test_origin_data_files, help='img and mask in test set')
    
    parser.add_argument('--target_akaze_json_train_file', type=str, default=train_json_output, help='output akaze json train file')

    parser.add_argument('--target_akaze_json_test_file', type=str, default=test_json_output, help='output akaze json test file')

    opt = parser.parse_args()
    
    print("Example of original training img path: {}".format(opt.ori_train_files[0]))
    print("Example of original testing img path: {}".format(opt.ori_test_files[1]))
    print("AKAZE Json Train file: {}".format(opt.target_akaze_json_train_file))
    print("AKAZE Json Test file: {}".format(opt.target_akaze_json_test_file))

    # print(test(opt.ori_train_files[:100])) # (201.29, 2257)
    # print(test(opt.ori_train_files + opt.ori_test_files)) # (110.4488636363636, 1247)
    # print(test([opt.ori_train_files[12]]))

    akaze_extract(data_file_path=opt.ori_train_files, target_path=opt.target_akaze_json_train_file)
    akaze_extract(data_file_path=opt.ori_test_files, target_path=opt.target_akaze_json_test_file)

    # img = cv2.imread(opt.ori_train_files[100])
    # # print(opt.ori_train_files[14])
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # akaze = cv2.AKAZE_create()
    # kp, des = akaze.detectAndCompute(gray,None)
    # # print(len(kp), len(des))
    # img_keys=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('Data Preprocessing/akaze_invariant_check/akaze_keypoints.jpg',img_keys)

    # scale_percent = 60 
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
  
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # gray_resized = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    # gray_resized_kp, gray_resized_des = akaze.detectAndCompute(gray_resized,None)
    # resized_img_keys = cv2.drawKeypoints(gray_resized,gray_resized_kp,resized,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('Data Preprocessing/akaze_invariant_check/resized_akaze_keypoints.jpg',resized_img_keys)

    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des,gray_resized_des,k=2)

    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])
    # scaled_match = cv2.drawMatchesKnn(gray,kp,gray_resized,gray_resized_kp,good,None,flags=cv2.DrawMatchesFlags_DEFAULT)
    # cv2.imwrite('Data Preprocessing/akaze_invariant_check/scaledmatch_akaze.jpg',scaled_match)

    # (h, w) = img.shape[:2]
    # (cX, cY) = (w // 2, h // 2)
    
    # M = cv2.getRotationMatrix2D((cX, cY), -160, 1.0)
    # rotated = cv2.warpAffine(img, M, (w, h))
    # rotated_gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    # gray_rotate_kp, gray_rotate_des = akaze.detectAndCompute(rotated_gray,None)
    # matches = bf.knnMatch(des,gray_rotate_des,k=2)

    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])
    # rotate_match = cv2.drawMatchesKnn(gray,kp,rotated_gray,gray_rotate_kp,good,None,flags=cv2.DrawMatchesFlags_DEFAULT)
    # cv2.imwrite('Data Preprocessing/akaze_invariant_check/rotatematch_akaze.jpg',rotate_match)

    # gray_cropped_image = gray[50:320, 110:370]
    # # cv2.imshow("cropped", cropped_image)
    # # cv2.waitKey(0)
    # crop_kp, crop_des = akaze.detectAndCompute(gray_cropped_image, None)

    # matches = bf.knnMatch(des,crop_des,k=2)

    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])
    # cropped_match = cv2.drawMatchesKnn(gray,kp,gray_cropped_image,crop_kp,good,None,flags=cv2.DrawMatchesFlags_DEFAULT)
    # cv2.imwrite('Data Preprocessing/akaze_invariant_check/croppedmatch_akaze.jpg',cropped_match)