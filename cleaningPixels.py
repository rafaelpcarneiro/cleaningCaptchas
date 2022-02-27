#!/usr/bin/env python3
# vim: set ts=4 foldmethod=marker:

import numpy as np
from PIL import Image, ImageFilter
import time

dx, dy = 9, 9

def makeImageBinary(img):
	return 255 * (img >= 255/2)

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


	# Loading parameters
	pixel_is_part_of_word = \
		np.loadtxt('parameters_9x9/pixel_is_a_word.txt', \
					delimiter=';')

	neighborhood_given_word_param_x = \
		np.loadtxt('parameters_9x9/neighborhood_given_word_param_x.txt', \
					delimiter=';')

	neighborhood_given_word_param_total = \
		np.loadtxt('parameters_9x9/neighborhood_given_word_param_total.txt', \
					delimiter=';')

	neighborhood_given_not_word_param_x = \
		np.loadtxt('parameters_9x9/neighborhood_given_not_word_param_x.txt', \
					delimiter=';')

	neighborhood_given_not_word_param_total = \
		np.loadtxt('parameters_9x9/neighborhood_given_not_word_param_total.txt', \
					delimiter=';')

	# Cleaning Images
	for file in trainingSampleFiles:
		time.sleep(1)

		# Opennig images
		# gray image
		img        = Image.open('data/' + file)
		img        = img.convert('L')
		#img        = img.filter(ImageFilter.BoxBlur(3))
		img_array  = np.asarray(img) 
		img_array  = makeImageBinary(img_array)

		img_array2 = img_array.copy()

		#img_array2[:dx, :]                        = 255
		#img_array2[(img_array.shape[0] - dx):, :] = 255

		#img_array2[:, :dy]                        = 255
		#img_array2[:, (img_array.shape[1] - dy):] = 255

		for x in range(dx, img_array.shape[0] - dx):
			for y in range(dy, img_array.shape[1] - dy):

				if img_array[x,y] == 0:
					neighbor = img_array[(x - dx):(x + dx + 1), \
										 (y - dy):(y + dy + 1)]
					
					## Probabilities
					prob_pixel = pixel_is_part_of_word[0] / pixel_is_part_of_word[1]
					neighbor_1 = neighbor == 255
					neighbor_0 = neighbor == 0


					# Probability of being a word given a neighborhood
					temp_1_word = (1 - neighborhood_given_word_param_x[neighbor_1] / 
									   neighborhood_given_word_param_total[neighbor_1]).prod()

					temp_1_word = temp_1_word if temp_1_word > 0 else 1.0


					temp_0_word = (neighborhood_given_word_param_x[neighbor_0] / 
								   neighborhood_given_word_param_total[neighbor_0]).prod()

					temp_0_word = temp_0_word if temp_0_word > 0 else 1.0


					prob_is_a_word = prob_pixel * temp_1_word * temp_0_word

					# Probability of not being a word given a neighborhood
					temp_1_word = (1 - neighborhood_given_not_word_param_x[neighbor_1] / 
									   neighborhood_given_not_word_param_total[neighbor_1]).prod()

					temp_1_word = temp_1_word if temp_1_word > 0 else 1.0


					temp_0_word = (neighborhood_given_not_word_param_x[neighbor_0] / 
								   neighborhood_given_not_word_param_total[neighbor_0]).prod()

					temp_0_word = temp_0_word if temp_0_word > 0 else 1.0


					prob_not_a_word = (1 - prob_pixel) * temp_1_word * temp_0_word

					# Calculating the classifier for each pixel
					K = 1 / (prob_is_a_word + prob_not_a_word)

					#img_array2[x,y] = 255 if prob_not_a_word * K >= 0.98 else 0
					img_array2[x,y] = 255 if prob_is_a_word * K <= 0.05 else 0
					#print( prob_not_a_word * K)


		print ('Image ' + file + ' --> DONE')

		
		img2 = Image.fromarray(np.uint8(img_array2), mode='L')
		img2.save('tmp.jpg')


