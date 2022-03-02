#!/usr/bin/env python3
# vim: set ts=4 foldmethod=marker:

import numpy as np
from PIL import Image, ImageFilter
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

dx, dy = 7, 7
dy_dx = 3

# vector directions (images have the following coordinate system)
#   0--------->y
#   |
#   |
#   |
#   V
#   x
d0   = np.array([ 0,  1])
d45  = np.array([-1,  1])
d90  = np.array([-1,  0])
d135 = np.array([-1, -1])
d180 = np.array([ 0, -1])
d225 = np.array([ 1, -1])
d270 = np.array([ 1,  0])
d315 = np.array([ 1,  1])

### AUXILIARY FUNCTIONS
#{{{1
def makeImageBinary(img):
	return 255 * (img >= 255/2)

def is_pixel_inside_img(img, pixel):
	return ((0 <= pixel[0] < img.shape[0]) and (0 <= pixel[1] < img.shape[1]))

def is_pixel_black(img, pixel):
	if is_pixel_inside_img(img, pixel):
		return (img[pixel[0], pixel[1]] == 0)
	else:
		False

# counting how many black pixels on the direction d0 = 1 * (cos 0, sin 0)
# from the position (pixel[0], pixel_y)
def d0_direction(img, pixel_at):
	step  = 0
	pixel = pixel_at + d0
	while is_pixel_inside_img(img, pixel) and is_pixel_black(img, pixel):
		step  += 1
		pixel = pixel + d0
	
	return step

def d0_thickness(img, pixel_at):
	length  = d0_direction(img, pixel_at)

	pixel   = pixel_at.copy()
	dir_vec = d0
	ort_vec = d90

	if length == 0:
		return 0
	else:
		thickness = np.zeros(length + 1)

		for i in range(length + 1):
			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec + step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec + step * ort_vec):

				  step += 1

			thickness[i] = step - 1

			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec - step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec - step * ort_vec):

				  step += 1

			thickness[i] += step - 1

		return thickness.mean()


# counting how many black pixels on the direction d45 = 1 * (cos 45, sin 45)
# from the position (pixel[0], pixel[1])
def d45_direction(img, pixel_at):
	step  = 0
	pixel = pixel_at + d45
	while is_pixel_inside_img(img, pixel) and is_pixel_black(img, pixel):
		step += 1
		pixel = pixel + d45
	
	return step

def d45_thickness(img, pixel_at):
	length  = d45_direction(img, pixel_at)

	pixel   = pixel_at.copy()
	dir_vec = d45
	ort_vec = d135

	if length == 0:
		return 0
	else:
		thickness = np.zeros(length + 1)

		for i in range(length + 1):
			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec + step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec + step * ort_vec):

				  step += 1

			thickness[i] = step - 1

			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec - step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec - step * ort_vec):

				  step += 1

			thickness[i] += step - 1

		return thickness.mean()

# counting how many black pixels on the direction d90 = 1 * (cos 90, sin 90)
# from the position (pixel[0], pixel[1])
def d90_direction(img, pixel_at):
	step = 0
	pixel = pixel_at + d90
	while is_pixel_inside_img(img, pixel) and is_pixel_black(img, pixel):
		step  += 1
		pixel = pixel + d90
	
	return step

def d90_thickness(img, pixel_at):
	length  = d90_direction(img, pixel_at)

	pixel   = pixel_at.copy()
	dir_vec = d90
	ort_vec = d180

	if length == 0:
		return 0
	else:
		thickness = np.zeros(length + 1)

		for i in range(length + 1):
			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec + step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec + step * ort_vec):

				  step += 1

			thickness[i] = step - 1

			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec - step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec - step * ort_vec):

				  step += 1

			thickness[i] += step - 1

		return thickness.mean()

# counting how many black pixels on the direction d135 = 1 * (cos 135, sin 135)
# from the position (pixel[0], pixel[1])
def d135_direction(img, pixel_at):
	step = 0
	pixel = pixel_at + d135
	while is_pixel_inside_img(img, pixel) and is_pixel_black(img, pixel):
		step  += 1
		pixel = pixel + d135
	
	return step

def d135_thickness(img, pixel_at):
	length  = d135_direction(img, pixel_at)

	pixel   = pixel_at.copy()
	dir_vec = d135
	ort_vec = d225

	if length == 0:
		return 0
	else:
		thickness = np.zeros(length + 1)

		for i in range(length + 1):
			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec + step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec + step * ort_vec):

				  step += 1

			thickness[i] = step - 1

			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec - step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec - step * ort_vec):

				  step += 1

			thickness[i] += step - 1

		return thickness.mean()

# counting how many black pixels on the direction d180 = 1 * (cos 180, sin 180)
# from the position (pixel[0], pixel[1])
def d180_direction(img, pixel_at):
	step  = 0
	pixel = pixel_at + d180
	while is_pixel_inside_img(img, pixel) and is_pixel_black(img, pixel):
		step  += 1
		pixel = pixel + d180
	
	return step

def d180_thickness(img, pixel_at):
	length  = d180_direction(img, pixel_at)

	pixel   = pixel_at.copy()
	dir_vec = d180
	ort_vec = d270

	if length == 0:
		return 0
	else:
		thickness = np.zeros(length + 1)

		for i in range(length + 1):
			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec + step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec + step * ort_vec):

				  step += 1

			thickness[i] = step - 1

			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec - step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec - step * ort_vec):

				  step += 1

			thickness[i] += step - 1

		return thickness.mean()

# counting how many black pixels on the direction d225 = 1 * (cos 225, sin 225)
# from the position (pixel[0], pixel[1])
def d225_direction(img, pixel_at):
	step = 0
	pixel = pixel_at + d225
	while is_pixel_inside_img(img, pixel) and is_pixel_black(img, pixel):
		step  += 1
		pixel = pixel + d225
	
	return step

def d225_thickness(img, pixel_at):
	length  = d225_direction(img, pixel_at)

	pixel   = pixel_at.copy()
	dir_vec = d225
	ort_vec = d315

	if length == 0:
		return 0
	else:
		thickness = np.zeros(length + 1)

		for i in range(length + 1):
			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec + step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec + step * ort_vec):

				  step += 1

			thickness[i] = step - 1

			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec - step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec - step * ort_vec):

				  step += 1

			thickness[i] += step - 1

		return thickness.mean()

# counting how many black pixels on the direction d270 = 1 * (cos 270, sin 270)
# from the position (pixel[0], pixel[1])
def d270_direction(img, pixel_at):
	step = 0
	pixel = pixel_at + d270
	while is_pixel_inside_img(img, pixel) and is_pixel_black(img, pixel):
		step  += 1
		pixel = pixel + d270
	
	return step

def d270_thickness(img, pixel_at):
	length  = d270_direction(img, pixel_at)

	pixel   = pixel_at.copy()
	dir_vec = d270
	ort_vec = d0

	if length == 0:
		return 0
	else:
		thickness = np.zeros(length + 1)

		for i in range(length + 1):
			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec + step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec + step * ort_vec):

				  step += 1

			thickness[i] = step - 1

			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec - step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec - step * ort_vec):

				  step += 1

			thickness[i] += step - 1

		return thickness.mean()

# counting how many black pixels on the direction d315 = 1 * (cos 315, sin 315)
# from the position (pixel[0], pixel[1])
def d315_direction(img, pixel_at):
	step = 0
	pixel = pixel_at + d315
	while is_pixel_inside_img(img, pixel) and is_pixel_black(img, pixel):
		step  += 1
		pixel = pixel + d315
	
	return step

def d315_thickness(img, pixel_at):
	length  = d315_direction(img, pixel_at)

	pixel   = pixel_at.copy()
	dir_vec = d315
	ort_vec = d45

	if length == 0:
		return 0
	else:
		thickness = np.zeros(length + 1)

		for i in range(length + 1):
			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec + step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec + step * ort_vec):

				  step += 1

			thickness[i] = step - 1

			step = 1
			while is_pixel_inside_img(img, pixel + i * dir_vec - step * ort_vec) and \
				  is_pixel_black     (img, pixel + i * dir_vec - step * ort_vec):

				  step += 1

			thickness[i] += step - 1

		return thickness.mean()


## So far these whites_rect measures have shown to be useless
# {{{2
###def whites_rect0(img, pos_x, pos_y):
###
###	dy_plus  = d0(img, pos_x, pos_y)
###	dy_minus = d180(img, pos_x, pos_y)
###
###	dx_plus  = d270(img, pos_x, pos_y)
###	dx_minus = d90(img, pos_x, pos_y)
###
###	r0 = 1 * (img[(pos_x - dx_minus) : (dx_plus + pos_x + 1), \
###	              (pos_y - dy_minus) : (dy_plus + pos_y + 1)] == 255)
###
###	return r0.sum() / ( (dx_plus + 1 + dx_minus) * (dy_plus + 1 + dy_minus))
###
###
###def whites_rect1(img, pos_x, pos_y):
###
###	dy_plus  = d45(img, pos_x, pos_y)
###	dy_minus = d45(img, pos_x, pos_y)
###
###	dx_plus  = d225(img, pos_x, pos_y)
###	dx_minus = d225(img, pos_x, pos_y)
###
###	r1 = 1 * (img[(pos_x - dx_minus) : (dx_plus + pos_x + 1), \
###	              (pos_y - dy_minus) : (dy_plus + pos_y + 1)] == 255)
###
###	return r1.sum() / ( (dx_plus + 1 + dx_minus) * (dy_plus + 1 + dy_minus))
###
###
###def whites_rect2(img, pos_x, pos_y):
###
###	dy_plus  = d135(img, pos_x, pos_y)
###	dy_minus = d135(img, pos_x, pos_y)
###
###	dx_plus  = d315(img, pos_x, pos_y)
###	dx_minus = d315(img, pos_x, pos_y)
###
###	r2 = 1 * (img[(pos_x - dx_minus) : (dx_plus + pos_x + 1), \
###	              (pos_y - dy_minus) : (dy_plus + pos_y + 1)] == 255)
###
###	return r2.sum() / ((dx_plus + 1 + dx_minus) * (dy_plus + 1 + dy_minus))
# 2}}}

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

### FUNCTIONS TO OPERATE OVER THE SAMPLE
def get_pixel():
	columns = range(2)
	return np.loadtxt('sample.txt', delimiter=';', usecols=columns)

def get_dist_pixel_border():
	pixel_x =  np.loadtxt('sample.txt', delimiter=';', usecols=(0))
	pixel_y =  np.loadtxt('sample.txt', delimiter=';', usecols=(1))

	pixel_x = np.column_stack((pixel_x.flatten(), 70 - pixel_x.flatten()))
	pixel_x = pixel_x.min(axis=1)

	pixel_y = np.column_stack((pixel_y.flatten(), 175 - pixel_y.flatten()))
	pixel_y = pixel_y.min(axis=1)

	return np.column_stack((pixel_x, pixel_y))

def dist_pixel_border(pixel_x, pixel_y):
	dist_x = min([pixel_x, 70  - pixel_x])
	dist_y = min([pixel_y, 175 - pixel_y])

	return (dist_x, dist_y)

def get_dist_pixel_center():
	pixel_x =  np.loadtxt('sample.txt', delimiter=';', usecols=(0))
	pixel_y =  np.loadtxt('sample.txt', delimiter=';', usecols=(1))

	pixel_x = np.abs( 70/2 - pixel_x )
	pixel_y = np.abs( 175/2 - pixel_y )

	dist = np.sqrt(pixel_x**2 + pixel_y**2)

	return dist[:, np.newaxis]

def get_direction_vectors():
	columns = range(2,10)
	return np.loadtxt('sample.txt', delimiter=';', usecols=columns)

def get_thickness_vectors():
	columns = range(10,18)
	return np.loadtxt('sample.txt', delimiter=';', usecols=columns)

def get_max_direction_vector():
	directions = get_direction_vectors() 

	directions = directions.max(axis=1)
	return directions[:, np.newaxis]

def get_max_heavy_direction_vector():
	directions = get_direction_vectors() * get_thickness_vectors()

	directions = directions.max(axis=1)
	return directions[:, np.newaxis]

def get_max_ball():
	max_ball =  np.loadtxt('sample.txt', delimiter=';', usecols=(18))
	return max_ball[:, np.newaxis]

def get_neighborhood():
	columns = range(19,67)
	return np.loadtxt('sample.txt', delimiter=';', usecols=columns)

def get_neighborhood_mean():
	neighborhood = get_neighborhood()
	neighborhood_mean = neighborhood.mean(axis=1)
	return neighborhood_mean[:, np.newaxis]

def get_neighborhood_median():
	neighborhood = get_neighborhood()
	neighborhood_median = np.median(neighborhood, axis=1)
	return neighborhood_median[:, np.newaxis]

def get_neighborhood_std():
	neighborhood = get_neighborhood()
	neighborhood_std = neighborhood.std(axis=1)
	return neighborhood_std[:, np.newaxis]

def get_target():
	return np.loadtxt('sample.txt', delimiter=';', usecols=(67))
	


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
	
							 #get_max_direction_vector(),
	sample = np.concatenate((get_dist_pixel_border(),
							 get_direction_vectors() * get_thickness_vectors(),
							 get_max_ball(),
	                         get_neighborhood_mean()), axis=1)

	target = get_target()

	trainSample = sample[:-200,:]
	trainTarget = target[:-200]

	testSample = sample[-200:,:]
	testTarget = target[-200:]

	# Neural Network
	#neuralModel = MLPClassifier(max_iter=1000, hidden_layer_sizes=(6,4))
	#neuralModel.fit(trainSample, trainTarget)
	#print('neural -> ', neuralModel.score(testSample, testTarget) )


	
	#sample = sample[:, (3,5,7,9, 11)]
	# Adjusting the model

	myModel = LogisticRegression(max_iter=10000)
	myModel.fit(trainSample, trainTarget)

	#myModel2 = SVC(max_iter=-1)
	##myModel2 = make_pipeline(StandardScaler(), SVC(kernel='linear', degree=3))
	#myModel2.fit(trainSample, trainTarget)

	#myModel3 = DecisionTreeClassifier(random_state=0)
	#myModel3.fit(trainSample, trainTarget)

	#myModel4 = RandomForestClassifier(random_state=0)
	#myModel4.fit(trainSample, trainTarget)

	print('logistic -> ', myModel.score(testSample, testTarget) )
	#print('SVM -> ', myModel2.score(testSample, testTarget) )
	#print('Tree -> ', myModel3.score(testSample, testTarget) )
	#print('Random Forest -> ', myModel4.score(testSample, testTarget) )
	#print(myModel2.coef_)

	# Cleaning Images
	for file in trainingSampleFiles:
		time.sleep(1)

		# Opennig images
		# gray image
		img        = Image.open('data/' + file)
		img        = img.convert('L')
		#img        = img.filter(ImageFilter.BoxBlur(3))
		#img        = img.filter(ImageFilter.MedianFilter(3))
		img_array  = np.asarray(img) 
		img_array  = makeImageBinary(img_array)

		img_array2 = img_array.copy()

		for x in range(dx, img_array.shape[0] - dx):
			for y in range(dy, img_array.shape[1] - dy):

				if img_array[x,y] == 0:
					pixel = np.array([x,y])

					neighborhood = img_array[ (pixel[0] - dy_dx):(pixel[0] + dy_dx + 1),
											  (pixel[1] - dy_dx):(pixel[1] + dy_dx + 1)]

					neighborhood = neighborhood.flatten()
					size_tmp = int(neighborhood.size / 2)
					neighborhood = np.concatenate((neighborhood[:size_tmp],
												   neighborhood[(size_tmp + 1):]))
					neighborhood = 1 * (neighborhood[np.newaxis, :] == 0)


					pixel_features = np.array([[
								dist_pixel_border(x,y)[0],                            
								dist_pixel_border(x,y)[1],                            
								d0_direction  (img_array, pixel) *d0_thickness  (img_array, pixel),          
								d45_direction (img_array, pixel) *d45_thickness (img_array, pixel),       
								d90_direction (img_array, pixel) *d90_thickness (img_array, pixel),       
								d135_direction(img_array, pixel) *d135_thickness(img_array, pixel),       
								d180_direction(img_array, pixel) *d180_thickness(img_array, pixel),      
								d225_direction(img_array, pixel) *d225_thickness(img_array, pixel),      
								d270_direction(img_array, pixel) *d270_thickness(img_array, pixel),      
								d315_direction(img_array, pixel) *d315_thickness(img_array, pixel),      
								max_ball(img_array, pixel[0], pixel[1]),
								neighborhood.mean()]])



					prob = myModel.predict_proba(pixel_features)
					if prob[0,0] > 0.8:
						img_array2[x,y] = 255 
						#img_array[x,y]  = 255 
					#print( prob_not_a_word * K)


		print ('Image ' + file + ' --> DONE')

		
		img2 = Image.fromarray(np.uint8(img_array2), mode='L')
		#img2 = img2.filter(ImageFilter.MedianFilter(3))
		#img_gray = img_gray.filter(ImageFilter.BoxBlur(3))
		#img2 = img2.filter(ImageFilter.MinFilter(3))
		img2.save('tmp.jpg')


#pixel_features = np.array([[pixel[0],                            
#            pixel[1],                            
#			d0_direction  (img_array, pixel),          
#			d45_direction (img_array, pixel),         
#			d90_direction (img_array, pixel),         
#			d135_direction(img_array, pixel),         
#			d180_direction(img_array, pixel),        
#			d225_direction(img_array, pixel),        
#			d270_direction(img_array, pixel),        
#			d315_direction(img_array, pixel),        
#			d0_thickness  (img_array, pixel),          
#			d45_thickness (img_array, pixel),         
#			d90_thickness (img_array, pixel),         
#			d135_thickness(img_array, pixel),         
#			d180_thickness(img_array, pixel),        
#			d225_thickness(img_array, pixel),        
#			d270_thickness(img_array, pixel),        
#			d315_thickness(img_array, pixel),        
#			max_ball(img_array, pixel[0], pixel[1]) ]])

#pixel_features = np.array( [[x,                            
#							y,                            
#							d0_direction  (img_array, pixel),          
#							d45_direction (img_array, pixel),         
#							d90_direction (img_array, pixel),         
#							d135_direction(img_array, pixel),         
#							d180_direction(img_array, pixel),        
#							d225_direction(img_array, pixel),        
#							d270_direction(img_array, pixel),        
#							d315_direction(img_array, pixel),        
#							max_ball(img_array, x, y)]])
