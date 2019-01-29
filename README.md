# Oversampling Image classification datasets for Fastai - a simple approach

This package aims to make it easy to use oversampling in image classification datasets.
Datasets with an imbalance between the number of data points per category are pretty common.<br>
This package converts data into a format that works well with Fastai's ImageDataBunch class.
<br>
It is assumed that the train folder has all the image files in folders of their respective categories. (Although you can also use the methods in the Oversampler class on data that has all images in a single folder with a csv file containing the labels).
To show just how easy it is, we start with a few lines of code.