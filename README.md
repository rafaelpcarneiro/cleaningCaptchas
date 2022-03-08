# Cleaning captcha images (==under development==)


Here my objective is to develop a program
whose input is a captcha image, such as the 
image 
<picture style="display:block;">
<img style="text-align:center; 
            display:block;"
     src="data/20220222182850.jpg" /> 
</picture>
into a cleaner image, without noise and visual
information related not directly with letters.

## Results so far

I have applied algorithms such as Naive Bayes,
Logistic Regression and SVM to each pixel of a
image. Such methods are on their respective 
branch.

As an example, down below I have 
two images, the original, and the version
which the algorithm understand as the real
image without noise, under a certain risk:

1. First example
[](./data/20220222182850.jpg)
[](./example/20220222182850.jpg)

2. Example
(./data/20220222182933.jpg)
(./example/20220222182933.jpg)

## Future work
I still need to
* Implement an algorithm that split the cleaned
captcha into 4 images containing possible letters
* Finally, After spliting the image into 4 smaller
images I need to implement an algorithm to classify
the letters.

So far I think that the segmentation problem is 
going to be the more tedious...

