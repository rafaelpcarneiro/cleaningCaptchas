#!/usr/bin/env python3
# vim: set ts=4 foldmethod=marker:

############################### README ########################################
# {{{1
#
# Objective: Consider a set consisting of images whose content is a finite
#            sequence of letters (3 or 4 in general) mixed with noise. For 
#            example:
#                         +-------------------+
#                         |*  *            *  |
#                         |   A****** H   ****|
#                         |     *b            |
#                         |*      ****        |
#                         |   *   ****** d    |
#                         +-------------------+
# 
#            where '*' is some noise.
#
#            This script simply aims to sample pixels from these images so it can
#            try to fit a hypothesis class based on a linear function with its
#            kernel separating the topological vector space between to connected
#            components so we can infer whether or not black pixel is part of a 
#            word
#
#
#
# Output: a txt file where each row - separated by ';' - is a sample of our
#         feature set.
#         The file looks like:
#
#         +-- sample.txt ---+
#         | row 1           |
#         | row 2           |
#         | row 3           |
#         | ..............  |
#         | ..............  |
#         | ..............  |
#         | row N           |
#         +-----------------+
# 
#
#         Here N is the sample's size and for every 0 <= i < N we have
#              row_i = (d0, d45, d90, d135, d180, d225, d270, d315, 
#                       pos_x, pos_y, max_ball, 
#                       whites_rect0, whites_rect1, whites_rect2, target)
#         
#         Example: Consider the image where '*' is a black pixel 
#                  (white background image) and 'o' the pixel being checked
#                  at position (pos_x, pos_y)
#
#                  +-----------------------------------+
#                  |                                   |
#                  |           ******                  |
#                  |    *******************            |
#                  |   ********      *******           |
#                  |  ********                         |
#                  |  ********                         |
#                  |  ********                         |
#                  |  ********                         |
#                  |  *******************              |
#                  |  ****o**************              |
#                  |  *******************              | 
#                  |  ********                         |
#                  |  ********                         |
#                  |  ********                         |
#                  |  ********                         |
#                  |  ********                         |
#                  |  ********                         |
#                  |  ********                         |
#                  |  ********                         |
#                  |                                   |
#                  +-----------------------------------+
#
#                Then we have
#                  +-----------------------------------+
#                  |                                   |
#                  |      d90  ******                  |
#                  |    **|****************            |
#                  |   ***|****      *******           |
#                  |  ****|***                         |
#                 d135\***|***d45                      |
#                  |  *\**|**/                         |
#                  |  **\*|*/*                         |
#                  |  ***\|/*************              |
#                 d180----o--------------d0            |
#                  |  ***/|\*************              | 
#                  |  **/*|*\*                         |
#                  |  */**|**\                         |
#                  |  /***|***d315                     |
#                 d225****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |      d270                         |
#                  +-----------------------------------+
#
#                 with d0, d45, ... being the lenght of these diagonal vectors
#                 of continuos black points; for instance d0 = 14, d45 = 3, ...
#
#            
#                with the rectangles 
#
#                        white_rect0 = (amount of white points inside r0)
#                                      -----------------------------------
#                                               area(r0)
#                  +-----------------------------------+
#                  |                                   |
#                  |      d90  ******                  |
#                  |  +-----------------+**            |
#                  |  |***|****      ***|***           |
#                  |  |***|***          |              |
#                 d135|***|***d45       |              |
#                  |  |\**|**/     r0   |              |
#                  |  |*\*|*/*          |              |
#                  |  |**\|/************|              |
#                 d180|---o-------------|d0            |
#                  |  |**/|\************|              | 
#                  |  |*/*|*\*          |              |
#                  |  |/**|**\          |              |
#                  |  |***|***d315      |              |
#                 d225|***|***          |              |
#                  |  |***|***          |              |
#                  |  |***|***          |              |
#                  |  |***|***          |              |
#                  |  +-----------------+              |
#                  |      d270                         |
#                  +-----------------------------------+
#
#                        white_rect1 = (amount of white points inside r1)
#                                      -----------------------------------
#                                               area(r1)
#                  +-----------------------------------+
#                  |                                   |
#                  |      d90  ******                  |
#                  |    **|****************            |
#                  |   ***|****      *******           |
#                  |  ****|***                         |
#                 d135\***|***d45                      |
#                  |  +------+                         |
#                  |  |*\*|*/|                         |
#                  |  |**\|/*|***********              |
#                 d180|---o--|-----------d0            |
#                  |  |**/|\*|***********              | 
#                  |  |*/*|*\|                         |
#                  |  |/**|**|---------> r1            |
#                  |  +------+d315                     |
#                 d225****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |      d270                         |
#                  +-----------------------------------+
#
#                        white_rect2 = (amount of white points inside r2)
#                                      -----------------------------------
#                                               area(r2)
#                  +-----------------------------------+
#                  |                                   |
#                  |      d90  ******                  |
#                  |    **|****************            |
#                  |   ***|****      *******           |
#                  |  ****|***                         |
#                 d135+------+d45                      |
#                  |  |\**|**|                         |
#                  |  |*\*|*/|                         |
#                  |  |**\|/*|***********              |
#                 d180|---o--|-----------d0            |
#                  |  |**/|\*|***********              | 
#                  |  |*/*|*\|--------------> r2       |
#                  |  +------+                         |
#                  |  /***|***d315                     |
#                 d225****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |  ****|***                         |
#                  |      d270                         |
#                  +-----------------------------------+
#
#                 Finally max_ball is the largest r >= 0 so the closed ball
#                 B( (pos_x, pos_y), r) contains only black pixels (Here the
#                 norm used is the maximum) and target is 1 if the pixel
#                 belongs to a word and 0 otherwise. 
#                  
#
# How to use it: the script will generate at each iteration a check box on
#                the image 'tmp.jpg' and it will as if the pixel at the center
#                of the box is part of a word or not. In that manner the sample
#                will be collected.
#
#                I suggest to keep the file 'tmp.jpg' openned side by side
#                with the terminal so you can easily answer the questions.
#  
# 1}}}


import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

# constants
dx, dy = 7, 7

dy_dx = 3

dx_checkBox, dy_checkBox = 3, 3

IMG_X, IMG_Y = 70, 175

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


if __name__ == "__main__":

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

	
	sampleFile = open('sample.txt', 'a')

	# feature names
	# {{{1
	#features = (
	#	'#pos_x;'        
	#	' pos_y;'        
	#	' d0;'           
	#	' d45;'           
	#	' d90;'           
	#	' d135;'          
	#	' d180;'          
	#	' d225;'          
	#	' d270;'          
	#	' d315;'         
	#	' d0_thickness;'           
	#	' d45_thickness;'           
	#	' d90_thickness;'           
	#	' d135_thickness;'          
	#	' d180_thickness;'          
	#	' d225_thickness;'          
	#	' d270_thickness;'          
	#	' d315_thickness;'         
	#	' max_ball;'     
	#	' neighbor00;'         
	#	' neighbor01;'         
	#	' neighbor02;'         
	#	' neighbor03;'         
	#	' neighbor04;'         
	#	' neighbor10;'         
	#	' neighbor11;'         
	#	' neighbor12;'         
	#	' neighbor13;'         
	#	' neighbor14;'         
	#	' neighbor20;'         
	#	' neighbor21;'         
	#	' neighbor23;'         
	#	' neighbor24;'         
	#	' neighbor30;'         
	#	' neighbor31;'         
	#	' neighbor32;'         
	#	' neighbor33;'         
	#	' neighbor34;'         
	#	' neighbor40;'         
	#	' neighbor41;'         
	#	' neighbor42;'         
	#	' neighbor43;'         
	#	' neighbor44;'         
	#	' target\n')
		# 1}}}

	#sampleFile.write(features)

	for iteration in range(10):
		for file in trainingSampleFiles:
			time.sleep(1)

			# Opennig images
			# gray image
			img        = Image.open('data/' + file)
			img        = img.convert('L')
			img_array  = np.asarray(img) 
			img_array  = makeImageBinary(img_array)

			# colored images to check samples
			img2       = Image.open('data/' + file)
			img_array2 = np.asarray(img2) 

			# Select only black pixels in a centered window
			X_MAX = img_array.shape[0] - dx 
			Y_MAX = img_array.shape[1] - dy

			pos_black_pixels = (img_array[dx:X_MAX, dy:Y_MAX] == 0).nonzero()
			pos_black_pixels = np.array(pos_black_pixels)

			random_val = np.random.randint(0, pos_black_pixels.shape[1])


			pixel = np.array([dx + pos_black_pixels[0, random_val],
			                  dy + pos_black_pixels[1, random_val]])

			x = pixel[0]
			y = pixel[1]

			neighborhood = img_array[ (pixel[0] - dy_dx):(pixel[0] + dy_dx + 1),
			                          (pixel[1] - dy_dx):(pixel[1] + dy_dx + 1)]

			neighborhood = neighborhood.flatten()
			size_tmp = int(neighborhood.size / 2)
			neighborhood = np.concatenate((neighborhood[:size_tmp],
			                               neighborhood[(size_tmp + 1):]))
			neighborhood = 1 * (neighborhood[np.newaxis, :] == 0)



			# Taking the sample values
			sample = np.array([[pixel[0],                            
			                    pixel[1],                            
								d0_direction  (img_array, pixel),          
								d45_direction (img_array, pixel),         
								d90_direction (img_array, pixel),         
								d135_direction(img_array, pixel),         
								d180_direction(img_array, pixel),        
								d225_direction(img_array, pixel),        
								d270_direction(img_array, pixel),        
								d315_direction(img_array, pixel),        
								d0_thickness  (img_array, pixel),          
								d45_thickness (img_array, pixel),         
								d90_thickness (img_array, pixel),         
								d135_thickness(img_array, pixel),         
								d180_thickness(img_array, pixel),        
								d225_thickness(img_array, pixel),        
								d270_thickness(img_array, pixel),        
								d315_thickness(img_array, pixel),        
								max_ball(img_array, pixel[0], pixel[1]) ]])

			sample = np.concatenate((sample, neighborhood, np.zeros((1,1))), axis=1)


			# Print the check Box
			img_array2[(x - dx_checkBox):(x + dx_checkBox + 1), 
					   (y - dy_checkBox):(y + dy_checkBox + 1), 
					   0] = 255

			img_array2[(x - dx_checkBox):(x + dx_checkBox + 1), 
					   (y - dy_checkBox):(y + dy_checkBox + 1), 
					   (1,2)] = 0

			img_array2[x,y,:] = 0

			img2 = Image.fromarray(np.uint8(img_array2))
			img2.save('tmp.jpg')

			# Ask whether or not the pixel is part of a word
			ans = 0
			ans = input('Is that pixel a part of a word? [y/n] ')
			sample[0,-1] = 1 if ((ans == 'y') or (ans == '')) else 0


			# Saving the sample

			features_fmt = ['%d'] * 10 + ["%.2f"] * 8 + ["%d"] + ["%d"] * 49
			#print(sample)
			np.savetxt(sampleFile, 
						sample, 
						delimiter=';', 
						fmt=features_fmt)

		print("iteration = ", iteration)
	
	sampleFile.close()
