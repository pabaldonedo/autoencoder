import os
import numpy as np
import cv2

def image_to_patches(fname, num_pixels):
	'''
	Convert an input image to patches. The number of patches per image
	is calculated using the number of pixels per patch.
	'''
	img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	height, width = img.shape
	n = np.sqrt(num_pixels)
	num_patches = height * width / num_pixels
	res = np.empty(shape=(num_patches, num_pixels))

	k = 0
	for i in xrange(0, height / int(n)):
		for j in xrange(0, width / int(n)):
			x0 = i * n
			x1 = (i + 1) * n
			y0 = j * n
			y1 = (j + 1) * n
			res[k,:] = img[x0:x1, y0:y1].flatten()

			k = k + 1

	return res