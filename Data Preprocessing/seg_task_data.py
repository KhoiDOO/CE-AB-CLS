import os
from os import path
import glob
import shutil
import argparse

def validation(_main_data_dir):
	main_seg = _main_data_dir + "/Seg_Task_Data"
	if path.exists(main_seg):
		print("Seg_Task_Data Found")
		train_seg = main_seg + "/Train"
		test_seg = main_seg + "/Test"
		if path.exists(train_seg) and path.exists(test_seg):
			print("Seg_Task_Data/Train and Seg_Task_Data/Test Found")
			for x in [train_seg + "/Img", train_seg + "/Mask", test_seg + "/Img", test_seg + "/Mask"]:
				if path.exists(x):
					print("{} Found".format(x.split('/')[-2:]))
				else:
					print("{} Not Found".format(x.split('/')[-2:]))
					os.mkdir(x)
					print("{} Created".format(x.split('/')[-2:]))
		elif not path.exists(train_seg):
			print("Seg_Task_Data/Train Missing")
			os.mkdir(train_seg)
			os.mkdir(train_seg + "/Img")
			os.mkdir(train_seg + "/Mask")
			print("Seg_Task_Data/Train Created")
		elif not path.exists(test_seg):
			print("Seg_Task_Data/Test Missing")
			os.mkdir(test_seg)
			os.mkdir(test_seg + "/Img")
			os.mkdir(test_seg + "/Mask")
			print("Seg_Task_Data/Test Created")


class SegTaskDataset:
	def __init__(self, ori_train_files, 
						ori_test_files, 
						target_train_img_dir, 
						target_train_mask_dir, 
						target_test_img_dir, 
						target_test_mask_dir):
		self.ori_train_files = ori_train_files
		self.ori_test_files = ori_test_files
		self.target_train_img_dir = target_train_img_dir
		self.target_train_mask_dir = target_train_mask_dir
		self.target_test_img_dir = target_test_img_dir
		self.target_test_mask_dir = target_test_mask_dir

	def create_dataset(self):
		for x in self.ori_train_files:
			filename = x.split('\\')[-1]
			if "ROI" in x:
				shutil.copy(x, self.target_train_mask_dir + "/" + filename)
			else:
				shutil.copy(x, self.target_train_img_dir + "/" + filename)

		for x in self.ori_test_files:
			filename = x.split('\\')[-1]
			if "ROI" in x:
				shutil.copy(x, self.target_test_mask_dir + "/" + filename)
			else:
				shutil.copy(x, self.target_test_img_dir + "/" + filename)
		print("Finish Creating Data Set")

if __name__ == '__main__':
	print(os.getcwd())
	os.chdir("..")
	print(os.getcwd())

	main_data_dir = os.getcwd() + "/Data set"

	# Validation for Folder Structure
	validation(main_data_dir)

	origin_data_dir = main_data_dir + "/Original Form"
	train_origin_data_dir = origin_data_dir + "/Train"
	test_origin_data_dir = origin_data_dir + "/Test"
	train_origin_data_files = glob.glob(train_origin_data_dir + "/*/*")
	test_origin_data_files = glob.glob(test_origin_data_dir + "/*/*")

	seg_task_data_dir = main_data_dir + "/Seg_Task_Data"
	train_seg_task_data_dir = seg_task_data_dir + "/Train"
	test_seg_task_data_dir = seg_task_data_dir + "/Test"

	train_seg_task_img_dir = train_seg_task_data_dir + "/Img"
	train_seg_task_mask_dir = train_seg_task_data_dir + "/Mask"

	test_seg_task_img_dir = test_seg_task_data_dir + "/Img"
	test_seg_task_mask_dir = test_seg_task_data_dir + "/Mask"

	parser = argparse.ArgumentParser()

	parser.add_argument('--ori_train_files', type=list,
                        default=train_origin_data_files, help='img and mask in train set')

	parser.add_argument('--ori_test_files', type=list,
                        default=test_origin_data_files , help='img and mask in test set')

	parser.add_argument('--target_train_img_dir', type=str,
                        default=train_seg_task_img_dir, help='target directory for training image')

	parser.add_argument('--target_train_mask_dir', type=str,
                        default=train_seg_task_mask_dir , help='target directory for training mask')

	parser.add_argument('--target_test_img_dir', type=str,
                        default=test_seg_task_img_dir, help='target directory for testing image')

	parser.add_argument('--target_test_mask_dir', type=str,
                        default=test_seg_task_mask_dir , help='target directory for testing image')

	opt = parser.parse_args()

	print("Example of original training img path: {}".format(opt.ori_train_files[0]))
	print("Example of original testing img path: {}".format(opt.ori_test_files[1]))
	print("Target Dir for Training Image: {}".format(opt.target_train_img_dir))
	print("Target Dir for Training Mask: {}".format(opt.target_train_mask_dir))
	print("Target Dir for Testing Image: {}".format(opt.target_test_img_dir))
	print("Target Dir for Testing Mask: {}".format(opt.target_test_mask_dir))


	segtaskdataset = SegTaskDataset(ori_train_files = opt.ori_train_files, 
						ori_test_files = opt.ori_test_files, 
						target_train_img_dir = opt.target_train_img_dir, 
						target_train_mask_dir = opt.target_train_mask_dir, 
						target_test_img_dir = opt.target_test_img_dir, 
						target_test_mask_dir = opt.target_test_mask_dir)

	segtaskdataset.create_dataset()