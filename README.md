# Cleaning captcha images (*under development*)


My objective is to develop a program
whose input is a captcha image, such as the 
image down bellow 
<p align="center">
    <img src="data/20220222182850.jpg" /> 
</p>
and to output a cleaner image, that is, an image without noise 
and whose pixels are visual components of letters.

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
<p align="center">
    <img src="data/20220222182850.jpg" /> 
    <img src="example/20220222182850.jpg" /> 
</p>

2. Example
<p align="center">
    <img src="data/20220222182933.jpg" /> 
    <img src="example/20220222182933.jpg" /> 
</p>

## Future work
I still need to
* Implement an algorithm that split the cleaned
captcha into 4 images containing possible letters
* Finally, After spliting the image into 4 smaller
images I need to implement an algorithm to classify
the letters.

So far I think that the segmentation problem is 
going to be the more tedious...

