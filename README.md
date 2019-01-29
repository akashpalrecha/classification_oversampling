# Oversampling Image classification datasets for Fastai - a simple approach

>This README is INCOMPLETE <br>
>It will be updated within a week from now

This package aims to make it easy to use oversampling in image classification datasets.
Datasets with an imbalance between the number of data points per category are pretty common.<br>
This package converts data into a format that works well with Fastai's ImageDataBunch class.
<br>
To show just how easy it is, we start with a few lines of code:
```
from Oversampler import *

oversampler = Oversampler(PATH_TO_DATA, OUTPUT_DIR)
oversample_dict = {'category_1': 0.5, 'category_2': 0.3}
oversampled_df = oversampler.df_val_train_by_pct(valid_pct=0.2, cats_to_pct=oversample_dict)
oversampler.copy_to_output_with_csv(oversample=True)
```
This will do the following:
1. Grab all filenames along with their categories (inferred from folder name) from PATH_TO_DATA
2. Create a Pandas DataFrame containing file names and corresponding categories
3. Split the data into validation and training sets with each class being represented in the validation set in proportion to it's presence in the data.
4. In the training set, increase the number of samples of `category_1` and `category_2` by a fraction of `0.5` and `0.3` respectively. (Randomly repeat some filenames)
5. Put all the validation file names in the beginning of the DataFrame and store their count.
6. Copy all the data into one folder (from separate folders previously present in PATH_TO_DATA) and generate a `labels.csv` file containing the oversampled file names and labels.
7. Print out the number of samples in the validation set for reference. <br>

>This also takes care not to oversample the images in the validation set into the training set.  <br>

---
Here is what is assumed about the structure of the `PATH_TO_DATA` folder:<br>
- Let the categories in the dataset be : cat1, cat2, cat3, cat4.
- In this case, the `PATH_TO_DATA` contains 4 folders of names `cat1`, `cat2`, `cat3`, and `cat4`.
- Each folder contains images only of that category.
- `PATH_TO_DATA` contains no other files.
---
It is possible to achieve better results with a classifer using this method only if one uses Data Augmentation. The idea is that images in categories with less data will be used for training the classifer more number of times with each image augmented before feeding to the classifier.<br>
---
---
Hope this helps!