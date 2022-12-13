import os
import json
import shutil
import pandas as pd

col_names = ["Method", "Feature Descriptor",
    "k-cluster", "Max Accuracy", "Max Precision", "Max Sensitivity", "Max Specificity",
    "Max F1-score", "Max AUC", "Mean Accuracy", "Mean Precision", "Mean Sensitivity",
    "Mean Specificity", "Mean F1-score", "Mean AUC", "CI UP Accuracy", "CI Down Accuracy",
    "CI UP Precision", "CI Down Precision", "CI UP Sensitivity", "CI Down Sensitivity",
    "CI UP Specificity", "CI Down Specificity", "CI UP F1-score", "CI Down F1-score",
    "CI UP AUC", "CI Down AUC"]

# print(os.getcwd())
def extract(filename = None, col_names = col_names):
    base_dict = {i : [] for i in range(len(col_names))}
    # print(base_dict)
    print('Extracting')
    js_path = os.getcwd() + "\\{}.json".format(filename)
    csv_path = os.getcwd() + "\\{}.csv".format(filename)

    with open(js_path, mode='r', encoding='utf-8') as f:
        output_file = json.load(f)

    for method in output_file:
        # print(method)
        for feature_des in output_file[method]:
            # print(feature_des)
            for experiment in output_file[method][feature_des]:
                # print(experiment)
                base_dict[0].append(method)
                # print(base_dict[0])
                base_dict[1].append(feature_des)
                for x in range(len(experiment)):
                    base_dict[x+2].append(experiment[x])
    # print(base_dict)
    col_change = {j : name for j, name in enumerate(col_names)}
    df = pd.DataFrame(data = base_dict).rename(col_change, axis=1)
    df.to_csv(csv_path)

extract(filename='re1')
