def show_seg(img = None, mask = None, current_channel = "last"):
	if img == None:
		raise Exception("img argument is required")
	if mask == None:
		raise Exception("mask argument is require")
	if current_channel == 'last':
		img[:, :, 0] = img[:, :, 0]*mask
    	img[:, :, 1] = img[:, :, 1]*mask
    	img[:, :, 2] = img[:, :, 2]*mask
   	 	return img
   	elif current_channel == 'first':
   		img[0, :, :] = img[0, :, :]*mask
    	img[1, :, :] = img[1, :, :]*mask
    	img[2, :, :] = img[2, :, :]*mask
    	return img
    else:
    	raise Exception("current_channel has to first or last")