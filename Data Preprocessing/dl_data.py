import os
import glob
import shutil

# print(os.getcwd())

os.chdir("..")

# print(os.getcwd())

main_data_dir = os.getcwd() + "\\Data set"
# print(main_data_dir)

ori_data_dir = main_data_dir + "\\Original Form"

full_data = glob.glob(ori_data_dir + "\\*\\*\\*")
dl_data = [x for x in full_data if "ROI" not in x]
print(len(dl_data))

# print(dl_data[1])

target_dir = main_data_dir + "\\DL_data"
target_ab_dir = main_data_dir + "\\DL_data\\Abnormal"
target_no_dir = main_data_dir + "\\DL_data\\Normal"

if not os.path.exists(target_dir):
    os.mkdir(main_data_dir + "\\DL_data")

if not os.path.exists(target_ab_dir):
    os.mkdir(main_data_dir + "\\DL_data\\Abnormal")

if not os.path.exists(target_no_dir):
    os.mkdir(main_data_dir + "\\DL_data\\Normal")

for x in dl_data:
    filename = x.split("\\")[-1]
    if "Normal" in x:
        target = target_no_dir + "\\{}".format(filename)
        if not os.path.exists(target):
            shutil.copyfile(x, target)
    else:
        target = target_ab_dir + "\\{}".format(filename)
        if not os.path.exists(target):
            shutil.copyfile(x, target)

chk_ab_data = glob.glob(target_ab_dir + "\\*")
chk_no_data = glob.glob(target_no_dir + "\\*")
print(len(chk_ab_data), len(chk_no_data))

for i in chk_ab_data:
    if "Abnormal" not in i:
        print("Incorrect Position: {}".format(x.split("\\")[-1]))

for i in chk_no_data:
    if "Normal" not in i:
        print("Incorrect Position: {}".format(x.split("\\")[-1]))
