import os 
import glob
import json 
import numpy as np
import pandas as pd

# print(os.getcwd())
os.chdir("..")
os.chdir("..")
# print(os.getcwd())

main_data_dir = os.getcwd() + "\\Data set"
sift_data = main_data_dir + "\\sift"
akaze_data = main_data_dir + "\\akaze"
kaze_data = main_data_dir + "\\kaze"
orb_data = main_data_dir + "\\orb"
surf_data = main_data_dir + "\\surf"

# print(sift_data)
# print(akaze_data)
# print(kaze_data)
# print(orb_data)

data_train = [i for i in glob.glob(sift_data + '\\*') if 'train' in i]
data_test = [i for i in glob.glob(sift_data + '\\*') if 'test' in i]

# print(sift_data_train)
# print(sift_data_test)

def load_json(path):
    f = open(path)
    return json.load(f)

train_json = load_json(data_train[0])
test_json = load_json(data_test[0])


def create_kmean_dataset(json_data, path = None):
    normal_dict = json_data["Group 1 - Normal"]
    abnormal_dict = json_data["Group 2 - Abnormal"]

    feature_col = [f"feature_{i+1}" for i in range(128)]

    concat_lst = []

    for img_dict in normal_dict:
        for keys in img_dict:
            img_vector_lst = img_dict[keys]
            img_vector_normal = np.array(img_vector_lst)
            concat_lst.append(img_vector_normal)
    
    for img_dict in abnormal_dict:
        for keys in img_dict:
            img_vector_lst = img_dict[keys]
            img_vector_abnormal = np.array(img_vector_lst)
            concat_lst.append(img_vector_abnormal)

    # concat_data = np.concatenate((img_vector_normal, img_vector_abnormal), axis=0)
    
    base = concat_lst[0]
    for x in range(1, len(concat_lst)):
        concat_data = np.concatenate((base, concat_lst[x]), axis=0)
        base = concat_data

    if path:
        df = pd.DataFrame(concat_data, columns=feature_col)
        df.to_csv(path)



save_path = main_data_dir + "\\kmean_dataset\\" + train_json[0].split("\\")[-1].replace("json", "csv")
create_kmean_dataset(train_json, save_path)
