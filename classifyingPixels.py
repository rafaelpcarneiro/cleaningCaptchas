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
#            With these images, this script aims to fit a Naive Bayes
#            classifier  so it will be capable of removing all
#            noise pixels from the image lasting the pixels that are most likely
#            to be part of the letters.
#
#
#
# Output: The parameters used to estimate the following random variables:
#         (i) X ~ Bernoulli(theta) (X measures the likelihood of a pixel be part 
#                                   of a word)
#
#             On file 'pixel_is_a_word.txt' we have:
#
#             +- pixel_is_a_word.txt -----------------------------+
#             | number of pixels part of a letter; total of tests |
#             +---------------------------------------------------+
#
#         (ii) N | X = 1     (the conditional independent variable that checks
#                             the neighborhood of a pixel)
#
#             On file 'neighborhood_given_word_param_x.txt' we have a
#             matrix counting how many times a black pixel that is part
#             of a letter was found:
#
#             +- 'neighborhood_given_word_param_x.txt' -----------+
#             | N(1,1); N(1,2); ... ; N(1,dy);                    |
#             | N(2,1); N(2,2); ... ; N(2,dy);                    |
#             | ..............................                    |
#             | ..............................                    |
#             | ..............................                    |
#             | N(dx,1); N(dx,2); ... ; N(dx,dy);                 |
#             +---------------------------------------------------+
#
#             Also a matrix counting how many times a pixel that belongs
#             to a letter was found:
#
#             +- 'neighborhood_given_word_param_total.txt' --_----+
#             | total(1,1); total(1,2); ... ; total(1,dy);        |
#             | total(2,1); total(2,2); ... ; total(2,dy);        |
#             | ..............................                    |
#             | ..............................                    |
#             | ..............................                    |
#             | total(dx,1); total(dx,2); ... ; total(dx,dy);     |
#             +---------------------------------------------------+
#
#         (ii) N | X = 0     (the conditional independent variable that checks
#                             the neighborhood of a pixel)
#
#              The same files as above are created but now counting black
#              pixels that are noise:
#
#             +- 'neighborhood_given_not_word_param_x.txt' -------+
#             | N(1,1); N(1,2); ... ; N(1,dy);                    |
#             | N(2,1); N(2,2); ... ; N(2,dy);                    |
#             | ..............................                    |
#             | ..............................                    |
#             | ..............................                    |
#             | N(dx,1); N(dx,2); ... ; N(dx,dy);                 |
#             +---------------------------------------------------+
#
#             Also a matrix counting how many times a pixel that belongs
#             to a letter was found:
#
#             +- 'neighborhood_given_not_word_param_total.txt' ---+
#             | total(1,1); total(1,2); ... ; total(1,dy);        |
#             | total(2,1); total(2,2); ... ; total(2,dy);        |
#             | ..............................                    |
#             | ..............................                    |
#             | ..............................                    |
#             | total(dx,1); total(dx,2); ... ; total(dx,dy);     |
#             +---------------------------------------------------+
#
#
# How to use it: at any iteration the script will ask if the pixel at the
#                center of a box, painted as red or blue, on the file
#                'tmp.jpg' is part of a letter or not; here, red means 
#                that the pixel is more likely to be part of a letter and
#                blue means not.
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

dx_checkBox, dy_checkBox = 3, 3

IMG_X, IMG_Y = 70, 175

def makeImageBinary(img):
	return 255 * (img >= 255/2)



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

	# probabilty model -- matrices
	dtype0                = [('x', int), ('total', int)]
	

	pixel_is_part_of_word       = np.ones(1, dtype=dtype0)
	neighborhood_given_word     = np.ones((2*dx + 1, 2*dy + 1), dtype=dtype0)
	neighborhood_given_not_word = np.ones((2*dx + 1, 2*dy + 1), dtype=dtype0)


	pixel_is_part_of_word['total']       += pixel_is_part_of_word['total']
	neighborhood_given_word['total']     += neighborhood_given_word['total']
	neighborhood_given_not_word['total'] += neighborhood_given_not_word['total']
	

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

			# Select the pixels
			X_MAX = img_array.shape[0] - dx 
			Y_MAX = img_array.shape[1] - dy

			pos_black_pixels = (img_array[dx:X_MAX, dy:Y_MAX] == 0).nonzero()
			pos_black_pixels = np.array(pos_black_pixels)

			random_val = np.random.randint(0, pos_black_pixels.shape[1])

			x = dx + pos_black_pixels[0, random_val]
			y = dy + pos_black_pixels[1, random_val]


			neighbor = img_array[(x - dx):(x + dx + 1), (y - dy):(y + dy + 1)]
			
			## Probabilities
			prob_pixel = pixel_is_part_of_word['x'] / pixel_is_part_of_word['total']
			neighbor_1 = neighbor == 255
			neighbor_0 = neighbor == 0


			# Probability of being a word given a neighborhood
			temp_1_word = (1 - neighborhood_given_word['x'][neighbor_1] / 
							   neighborhood_given_word['total'][neighbor_1]).prod()

			temp_1_word = temp_1_word if temp_1_word > 0 else 1.0


			temp_0_word = (neighborhood_given_word['x'][neighbor_0] / 
						   neighborhood_given_word['total'][neighbor_0]).prod()

			temp_0_word = temp_0_word if temp_0_word > 0 else 1.0


			prob_is_a_word = prob_pixel * temp_1_word * temp_0_word

			# Probability of not being a word given a neighborhood
			temp_1_word = (1 - neighborhood_given_not_word['x'][neighbor_1] / 
							   neighborhood_given_not_word['total'][neighbor_1]).prod()

			temp_1_word = temp_1_word if temp_1_word > 0 else 1.0


			temp_0_word = (neighborhood_given_not_word['x'][neighbor_0] / 
						   neighborhood_given_not_word['total'][neighbor_0]).prod()

			temp_0_word = temp_0_word if temp_0_word > 0 else 1.0


			prob_not_a_word = (1 - prob_pixel) * temp_1_word * temp_0_word

			# Print the check Box
			if prob_is_a_word >= prob_not_a_word:
				scale = (1/(prob_is_a_word + prob_not_a_word)) * prob_is_a_word

				img_array2[(x - dx_checkBox):(x + dx_checkBox + 1), \
						   (y - dy_checkBox):(y + dy_checkBox + 1), \
						   0] = int(255 * scale)

				img_array2[(x - dx_checkBox):(x + dx_checkBox + 1), \
						   (y - dy_checkBox):(y + dy_checkBox + 1), \
						   (1,2)] = 0

			else:
				scale = (1/(prob_is_a_word + prob_not_a_word)) * prob_not_a_word
				img_array2[(x - dx_checkBox):(x + dx_checkBox + 1), \
						   (y - dy_checkBox):(y + dy_checkBox + 1), \
						   2] = int(255 * scale)

				img_array2[(x - dx_checkBox):(x + dx_checkBox + 1), \
						   (y - dy_checkBox):(y + dy_checkBox + 1), \
						   (0,1)] = 0

			img_array2[x,y,:] = 0

			img2 = Image.fromarray(np.uint8(img_array2))
			img2.save('tmp.jpg')

			# Calculate whether or not the pixel is part of a word
			ans = 0
			ans = input('Is that pixel a part of a word? [y/n] ')
			ans = 1 if ((ans == 'y') or (ans == '')) else 0


			if ans == 1:
				pixel_is_part_of_word['x']     += 1
				pixel_is_part_of_word['total'] += 1

				
				neighborhood_given_word['x']     += 1 * (neighbor == 0)
				neighborhood_given_word['total'] += 1

			else:
				pixel_is_part_of_word['total'] += 1

				neighborhood_given_not_word['x']     += 1 * (neighbor == 0)
				neighborhood_given_not_word['total'] += 1
		
			# Saving parameters
			save_tmp = [pixel_is_part_of_word['x'][0], \
						pixel_is_part_of_word['total'][0]]

			np.savetxt('parameters/pixel_is_a_word.txt', \
						np.array(save_tmp).reshape(1,2), \
						fmt="%d", \
						delimiter=';')

			np.savetxt('parameters/neighborhood_given_word_param_x.txt', \
						neighborhood_given_word['x'], \
						fmt="%d", \
						delimiter=';')

			np.savetxt('parameters/neighborhood_given_word_param_total.txt', \
						neighborhood_given_word['total'], \
						fmt="%d", \
						delimiter=';')

			np.savetxt('parameters/neighborhood_given_not_word_param_x.txt', \
						neighborhood_given_not_word['x'], \
						fmt="%d", \
						delimiter=';')

			np.savetxt('parameters/neighborhood_given_not_word_param_total.txt', \
						neighborhood_given_not_word['total'], \
						fmt="%d", \
						delimiter=';')

	print("iteration = ", iteration)
