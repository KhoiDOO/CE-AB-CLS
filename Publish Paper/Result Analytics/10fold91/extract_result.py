import json 
import numpy as np
import os
import shutil
from glob import glob

# print(os.getcwd())
for i in range(3):
    os.chdir("..")
# print(os.getcwd())

exper_dir = os.getcwd() + "\\Experiment"
km10fold91_dir = exper_dir + "\\Approach"
print(os.listdir(km10fold91_dir))

exper_files = [i for i in glob(km10fold91_dir + '\\*\\*.ipynb') if '.ipynb_checkpoints' not in i and 'all_model_analysis' not in i and 'HARDMSEG' not in i]

save_dict = {
    "SVM" : {
        "sift" : [],
        'surf' : [],
        'surf100' : [],
        'surf50' : [],
        'surf0' : [],
        'orb' : [],
        'brisk' : [],
        'kaze' : [],
        'akaze' : [],
    },
    "DecisionTree" : {
        "sift" : [],
        'surf' : [],
        'surf100' : [],
        'surf50' : [],
        'surf0' : [],
        'orb' : [],
        'brisk' : [],
        'kaze' : [],
        'akaze' : [],
    },
    "ExtraTreeClassifier" : {
        "sift" : [],
        'surf' : [],
        'surf100' : [],
        'surf50' : [],
        'surf0' : [],
        'orb' : [],
        'brisk' : [],
        'kaze' : [],
        'akaze' : [],
    },
    "LogisticRegression" : {
        "sift" : [],
        'surf' : [],
        'surf100' : [],
        'surf50' : [],
        'surf0' : [],
        'orb' : [],
        'brisk' : [],
        'kaze' : [],
        'akaze' : [],
    },
    "RandomForest" : {
        "sift" : [],
        'surf' : [],
        'surf100' : [],
        'surf50' : [],
        'surf0' : [],
        'orb' : [],
        'brisk' : [],
        'kaze' : [],
        'akaze' : [],
    }
}

save_path = os.getcwd() + "\\Publish Paper\\Result Analytics\\10fold91\\{}.json"

def extract(savepath = save_path, filename = None):
    if os.path.exists(savepath.format(filename)):
        os.remove(savepath.format(filename))
    for x in exper_files:
        with open(x, mode = 'r', encoding = 'utf-8') as f:
            base = json.load(f)
        x_split = x.split("\\")
        # print(x_split)
        method = x_split[5]
        name_split = x_split[-1].split(".")[0].split("_")
        feature_des_type = name_split[2]
        k_clu_num = int(name_split[-1])
        # print(k_clu_num, feature_des_type, method)
        write_result = []
        cells = base['cells']
        for i, cell in enumerate(cells):
            if 'result_max' in cell['source']:
                if len(cell['outputs']) == 0:
                    print(x)
                    continue
                result = cell['outputs'][0]['data']['text/plain']
                save_result = [float(res.split(" ")[-1][:-2]) for res in result[1:-1]]
                write_result = [k_clu_num] + np.ones((len(save_result))).tolist() + save_result
            if 'cl = 0.95\n' in cell['source'] and "dis_type = 'g'" in cell['source'][2]:
                if len(cell['outputs']) == 0:
                    print(x)
                    continue
                base_result = cell['outputs'][0]['text']
                combine_result = []
                for base in base_result:
                    extract = base.replace(")", "(").split("(")[1]
                    combine_result += [float(k) for k in extract.split(", ")]
                write_result += combine_result
            if len(write_result) == 25 and i == len(cells) - 1:
                save_dict[method][feature_des_type].append(write_result)
        with open(save_path.format(filename), "w") as save_file:
            json.dump(save_dict, save_file)

extract(filename = 're1')
    