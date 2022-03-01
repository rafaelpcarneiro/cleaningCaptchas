#!/usr/bin/env python3
# vim: set ts=4 foldmethod=marker:

import numpy as np
from PIL import Image, ImageFilter
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dx, dy = 7, 7

### AUXILIARY FUNCTIONS
#{{{1
def makeImageBinary(img):
	return 255 * (img >= 255/2)

# counting how many black pixels on the direction d0 = 1 * (cos 0, sin 0)
# from the position (pos_x, pos_y)
def d0(img, pos_x, pos_y):
	step = 0
	pos_y  += 1
	while (pos_y < img.shape[1]) and (img[pos_x, pos_y] == 0):
		step   += 1
		pos_y  += 1
	
	return step

# counting how many black pixels on the direction d45 = 1 * (cos 45, sin 45)
# from the position (pos_x, pos_y)
def d45(img, pos_x, pos_y):
	step = 0
	pos_y  += 1
	pos_x -= 1
	while (pos_y < img.shape[1]) and (pos_x >= 0) and (img[pos_x, pos_y] == 0):
		step   += 1
		pos_y  += 1
		pos_x -= 1
	
	return step

# counting how many black pixels on the direction d90 = 1 * (cos 90, sin 90)
# from the position (pos_x, pos_y)
def d90(img, pos_x, pos_y):
	step = 0
	pos_x -= 1
	while (pos_x >= 0) and (img[pos_x, pos_y] == 0):
		step  += 1
		pos_x -= 1
	
	return step

# counting how many black pixels on the direction d135 = 1 * (cos 135, sin 135)
# from the position (pos_x, pos_y)
def d135(img, pos_x, pos_y):
	step = 0
	pos_x -= 1
	pos_y -= 1
	while (pos_x >= 0) and (pos_y >= 0) and (img[pos_x, pos_y] == 0):
		step  += 1
		pos_x -= 1
		pos_y -= 1
	
	return step

# counting how many black pixels on the direction d180 = 1 * (cos 180, sin 180)
# from the position (pos_x, pos_y)
def d180(img, pos_x, pos_y):
	step = 0
	pos_y -= 1
	while (pos_y >= 0) and (img[pos_x, pos_y] == 0):
		step  += 1
		pos_y -= 1
	
	return step

# counting how many black pixels on the direction d225 = 1 * (cos 225, sin 225)
# from the position (pos_x, pos_y)
def d225(img, pos_x, pos_y):
	step = 0
	pos_x += 1
	pos_y -= 1
	while (pos_x < img.shape[0]) and (pos_y >= 0) and (img[pos_x, pos_y] == 0):
		step  += 1
		pos_x += 1
		pos_y -= 1
	
	return step

# counting how many black pixels on the direction d270 = 1 * (cos 270, sin 270)
# from the position (pos_x, pos_y)
def d270(img, pos_x, pos_y):
	step = 0
	pos_x += 1
	while (pos_x < img.shape[0]) and (img[pos_x, pos_y] == 0):
		step  += 1
		pos_x += 1
	
	return step

# counting how many black pixels on the direction d315 = 1 * (cos 315, sin 315)
# from the position (pos_x, pos_y)
def d315(img, pos_x, pos_y):
	step = 0
	pos_x += 1
	pos_y += 1
	while (pos_x < img.shape[0]) and (pos_y < img.shape[1])  and (img[pos_x, pos_y] == 0):
		step  += 1
		pos_x += 1
		pos_y += 1
	
	return step


def whites_rect0(img, pos_x, pos_y):

	dy_plus  = d0(img, pos_x, pos_y)
	dy_minus = d180(img, pos_x, pos_y)

	dx_plus  = d270(img, pos_x, pos_y)
	dx_minus = d90(img, pos_x, pos_y)

	r0 = 1 * (img[(pos_x - dx_minus) : (dx_plus + pos_x + 1), \
	              (pos_y - dy_minus) : (dy_plus + pos_y + 1)] == 255)

	return r0.sum() / ( (dx_plus + 1 + dx_minus) * (dy_plus + 1 + dy_minus))


def whites_rect1(img, pos_x, pos_y):

	dy_plus  = d45(img, pos_x, pos_y)
	dy_minus = d45(img, pos_x, pos_y)

	dx_plus  = d225(img, pos_x, pos_y)
	dx_minus = d225(img, pos_x, pos_y)

	r1 = 1 * (img[(pos_x - dx_minus) : (dx_plus + pos_x + 1), \
	              (pos_y - dy_minus) : (dy_plus + pos_y + 1)] == 255)

	return r1.sum() / ( (dx_plus + 1 + dx_minus) * (dy_plus + 1 + dy_minus))


def whites_rect2(img, pos_x, pos_y):

	dy_plus  = d135(img, pos_x, pos_y)
	dy_minus = d135(img, pos_x, pos_y)

	dx_plus  = d315(img, pos_x, pos_y)
	dx_minus = d315(img, pos_x, pos_y)

	r2 = 1 * (img[(pos_x - dx_minus) : (dx_plus + pos_x + 1), \
	              (pos_y - dy_minus) : (dy_plus + pos_y + 1)] == 255)

	return r2.sum() / ((dx_plus + 1 + dx_minus) * (dy_plus + 1 + dy_minus))


def max_ball(img, pos_x, pos_y):
	r = 1
	while (0 <= pos_x - r) and (pos_x + r <= img.shape[0]) and \
		  (0 <= pos_y - r) and (pos_y + r <= img.shape[1]):

		ball = (img[(pos_x - r):(pos_x + r + 1), \
					(pos_y - r):(pos_y + r + 1)] == 255).sum()

		if ball > 0:
			break
		else :
			r += 1

	
	return r - 1

#1}}}

if __name__ == '__main__':

	# training files
	#{{{1
	trainingSampleFiles = ['20220222182742.jpg', 
			 			   '20220222182808.jpg', 
			 			   '20220222182821.jpg',
			 			   '20220222182837.jpg',
			 			   '20220222182850.jpg',
			 			   '20220222182905.jpg',
			 			   '20220222182921.jpg',
			 			   '20220222182933.jpg',
			 			   '20220222182948.jpg',
			 			   '20220222183002.jpg',
			 			   '20220222183015.jpg',
			 			   '20220222183028.jpg',
			 			   '20220222183042.jpg',
			 			   '20220222183055.jpg',
			 			   '20220222183110.jpg',
			 			   '20220222183123.jpg',
			 			   '20220222183139.jpg',
			 			   '20220222183207.jpg',
			 			   '20220222183220.jpg',
			 			   '20220222183235.jpg',
			 			   '20220222183252.jpg',
			 			   '20220222183307.jpg',
			 			   '20220222183321.jpg',
			 			   '20220222183334.jpg',
			 			   '20220222183347.jpg',
			 			   '20220222183406.jpg',
			 			   '20220222183427.jpg',
			 			   '20220222183444.jpg',
			 			   '20220222183458.jpg',
			 			   '20220222183529.jpg',
			 			   '20220222183542.jpg',
			 			   '20220222183604.jpg',
			 			   '20220222183614.jpg',
			 			   '20220222183627.jpg',
			 			   '20220222183639.jpg',
			 			   '20220222183651.jpg',
			 			   '20220222183705.jpg',
			 			   '20220222183717.jpg',
			 			   '20220222183743.jpg',
			 			   '20220222183756.jpg',
			 			   '20220222183808.jpg',
			 			   '20220222183820.jpg',
			 			   '20220222183831.jpg',
			 			   '20220222183843.jpg',
			 			   '20220222183853.jpg',
			 			   '20220222183906.jpg',
			 			   '20220222183921.jpg',
			 			   '20220222183932.jpg',
			 			   '20220222183943.jpg',
			 			   '20220222183959.jpg',
			 			   '20220222184013.jpg',
			 			   '20220222184048.jpg',
			 			   '20220222184058.jpg',
			 			   '20220222184121.jpg',
			 			   '20220222184139.jpg',
			 			   '20220222184152.jpg',
			 			   '20220222184202.jpg',
			 			   '20220222184214.jpg',
			 			   '20220222184234.jpg',
			 			   '20220222184248.jpg',
			 			   '20220222184301.jpg',
			 			   '20220222184324.jpg',
			 			   '20220222184335.jpg',
			 			   '20220222184400.jpg',
			 			   '20220222184416.jpg',
			 			   '20220222184429.jpg',
			 			   '20220222184449.jpg',
			 			   '20220222184511.jpg',
			 			   '20220222184520.jpg',
			 			   '20220222184530.jpg',
			 			   '20220222184541.jpg',
			 			   '20220222184554.jpg',
			 			   '20220222184606.jpg',
			 			   '20220222184623.jpg',
			 			   '20220222184635.jpg',
			 			   '20220222184646.jpg',
			 			   '20220222184703.jpg',
			 			   '20220222184719.jpg',
			 			   '20220222184730.jpg',
			 			   '20220222184745.jpg',
			 			   '20220222185402.jpg',
			 			   '20220222185423.jpg',
			 			   '20220222185449.jpg',
			 			   '20220222185503.jpg',
			 			   '20220222185533.jpg',
			 			   '20220222185545.jpg',
			 			   '20220222185632.jpg',
			 			   '20220222185642.jpg',
			 			   '20220222185651.jpg',
			 			   '20220222185700.jpg',
			 			   '20220222185713.jpg',
			 			   '20220222185726.jpg',
			 			   '20220222185739.jpg',
			 			   '20220222185747.jpg',
			 			   '20220222185757.jpg',
			 			   '20220222185806.jpg',
			 			   '20220222185822.jpg',
			 			   '20220222185838.jpg',
			 			   '20220222185854.jpg']
	#1}}}


	# Loading the sample
	sample = np.loadtxt('sample.txt', delimiter=';', usecols=range(13))
	target = np.loadtxt('sample.txt', delimiter=';', usecols=13)

	# Adjusting the model

	myModel = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
	#myModel = LogisticRegression(max_iter=500)
	myModel.fit(sample[:800,], target[:800])

	#myModel2 = SVC(max_iter=-1)
	#myModel2.fit(sample[:800,], target[:800])
	myModel2 = make_pipeline(StandardScaler(), SVC())
	myModel2.fit(sample[:800,], target[:800])

	print('logistic -> ', myModel.score(sample[800:,], target[800:]) )
	print('SVM -> ', myModel2.score(sample[800:,], target[800:]) )

	# Cleaning Images
	#for file in trainingSampleFiles:
	#	time.sleep(1)

	#	# Opennig images
	#	# gray image
	#	img        = Image.open('data/' + file)
	#	img        = img.convert('L')
	#	#img        = img.filter(ImageFilter.BoxBlur(3))
	#	img_array  = np.asarray(img) 
	#	img_array  = makeImageBinary(img_array)

	#	img_array2 = img_array.copy()

	#	for x in range(dx, img_array.shape[0] - dx):
	#		for y in range(dy, img_array.shape[1] - dy):

	#			if img_array[x,y] == 0:
	#				pixel_features = np.array( [[x,                            
	#											y,                            
	#											d0(img_array, x, y),          
	#											d45(img_array, x, y),         
	#											d90(img_array, x, y),         
	#											d180(img_array, x, y),        
	#											d225(img_array, x, y),        
	#											d270(img_array, x, y),        
	#											d315(img_array, x, y),        
	#											max_ball(img_array, x, y),    
	#											whites_rect0(img_array, x, y),
	#											whites_rect1(img_array, x, y),
	#											whites_rect2(img_array, x, y)]])

	#				#prob = myModel2.predict_proba(pixel_features)
	#				#if prob[0,0] >= 0.6:
	#				#	img_array2[x,y] = 255 
	#				#print( prob_not_a_word * K)


	#	#print ('Image ' + file + ' --> DONE')

	#	
	#	img2 = Image.fromarray(np.uint8(img_array2), mode='L')
	#	#img2 = img2.filter(ImageFilter.MedianFilter(3))
	#	#img_gray = img_gray.filter(ImageFilter.BoxBlur(3))
	#	#img2 = img2.filter(ImageFilter.MaxFilter(3))
	#	img2.save('tmp.jpg')


