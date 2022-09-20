def show_seg(img = None, mask = None, current_channel = "last"):
	"""show_seg extract the segmented part from image by multiplying by mask

	Keyword Arguments:
		img (numpy.array) -- _description_ (default: {None})
		mask (numpy.array) -- _description_ (default: {None})
		current_channel (str) -- last --> img has shape(width, height, channel) 
								 first --> img has shape(channel, wigth, height) (default: {"last"})

	Raises:
		Exception: img can not be None
		Exception: mask can not be None
		Exception: current_channel has to first or last

	Returns:
		return segmented image whose segmented part is presented while others are 0
	"""
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