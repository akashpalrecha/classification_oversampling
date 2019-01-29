import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

class FolderToDf(object):
    def __init__(self, PATH, OUT=''):
        """
        PATH: PATH to folder with data. The folders in PATH must have their names as the same as category names.
        Also, each folde should contain samples to be labelled with the folder_name
        OUT: Folder to which to export all the data with a 'labels.csv' file mapping file names to labels
        """
        self.PATH = PATH
        self.OUT = OUT if OUT is not '' else None
        self.categories = self.get_cats(PATH)
        self.cat_to_file_dict = self.cat_to_files(PATH)[0]
        self.df = self.file_dict_to_df(self.cat_to_file_dict)

    def get_cats(self, PATH):
        """
        returns a list of categories from folder PATH
        Excludes any folder named 'models'
        """
        cats = os.listdir(PATH)
        try:
            cats.remove('models')
        except:
            pass
        return cats

    def get_cat_files(self, PATH, cat):
        """
        returns a list of files for given category
        """
        return os.listdir(PATH/cat)

    def cat_to_files(self, PATH):
        """
        Returns a dict of format: {category[i]:[list of file names]} for all i
        """
        cats = self.get_cats(PATH)
        file_dict = dict((i, self.get_cat_files(PATH, i)) for i in cats)
        num_files = dict((i, len(file_dict[i])) for i in file_dict)
        return file_dict, num_files

    def file_dict_to_df(self, file_dict):
        """
        returns a Pandas dataframe mapping all file names to labels using two columns
        file_dict: passed by self.cat_to_files (You do not need to use this function or any of the above by hand)
        """
        num_files = dict((i, len(file_dict[i])) for i in file_dict)
        length = sum(num_files.values())
        df_pre = []
        c = 0
        for i in tqdm(file_dict):
            for j in file_dict[i]:
                df_pre.append([j, i])
        return pd.DataFrame(df_pre, columns=['files', 'categories'])

    def get_df_from_folder(self, PATH):
        """
        Same as self.file_dict_to_df.
        This is the function that you will probably use
        """
        return self.file_dict_to_df(self.cat_to_files(PATH)[0])
    
    def copy_to_output(self, out=None, action='copy'):
        """
        Copies all samples in PATH to OUT
        out: override OUT location as default location to copy all files to along with labels.csv
        action: pass 'move' to copy all files and delete from previous location.
        """
        PATH = self.PATH
        files_dict = self.cat_to_file_dict
        OUT = self.OUT if out is None else out
        func = shutil.move if action == 'move' else shutil.copy
        assert OUT is not None
        for cat in files_dict:
            print(cat)
            for f in tqdm(files_dict[cat]):
                func(str(PATH/cat/f), str(OUT))
        self.df.to_csv(OUT/'labels.csv', index=False)
        print('Done!')

class Oversampler(FolderToDf):
    def __init__(self, PATH, OUT):
        super(Oversampler, self).__init__(PATH, OUT)
    
    def category_counts(self):
        """
        Prints the number of samples in each category
        """
        return self.df.categories.value_counts()
    
    def split_val_by_pct(self, valid_pct=0.2, categories_col='categories', cats_to_pct=None, oversample=False):
        """
        Splits the data into training and validation sets in the following way:
        - First splits each category into vaid and train using valid_pct
          (to ensure representation of each class in the validation set)
        - Collates the results into train and valid dataframes.
        - Oversamples the train dataframe if cats_to_pct is passed
        Parameters:
        valid_pct: Percentage of data to be (randomly) used as validation data.
        categories_col: Name of the column with labels in dataframe
        cate_to_pct: To be passed to self.oversample
        oversample: Set to True if oversampling happens. You do not need to pass this argument.
        """
        if cats_to_pct is not None:
            oversample = True
        valid_df = pd.DataFrame()
        for cat in set(self.df[categories_col]):
            valid_df = valid_df.append(self.df[self.df[categories_col] == cat].sample(frac=valid_pct))
        return (self.oversample(self.df.drop(index=valid_df.index), cats_to_pct, do=oversample).sample(frac=1),
                valid_df)
    
    def df_val_train_by_pct(self, valid_pct=0.2, categories_col='categories', oversample=False, cats_to_pct=None):
        """
        returns a single DataFrame.
        The train dataframe returned from self.split_val_by_pct is returned is appended to the
        valid dataframe.
        """
        trn, val = self.split_val_by_pct(valid_pct, categories_col, cats_to_pct, oversample)
        self.df_oversampled = val.append(trn)
        self.len_val = len(val)
        self.len_trn = len(trn)
        return self.df_oversampled, (self.len_val, self.len_trn)
    
    def oversample(self, df, cats_to_pct={}, categories_col='categories', do=True):
        """
        Applies oversampling on the DataFrame passed.
        Parameters:
        df: The Dataframe to use for oversampling
        cats_to_pct: Dictionary of the form {'category1':0.4, 'category2': 1.2 ...} 
                    Here, let's say cats_to_pct['category1'] = 0.5
                    This would mean that the data in category1 is to be repeated an *additional* 0.5 times.
                    That is, the total data in category1 would now be 1.5 times of what it was before
        
        This method will use a naive solution of simply repeating random samples of the data in `df`.
        This can be useful in cases where each image goes through data augmentation and hence would appear
        multiple times in the training process but in different forms.
        """

        if do == False:
            return df
        sample = pd.DataFrame()
        for i in cats_to_pct.keys():
            amt = cats_to_pct[i]
            frac = min(amt, 1)
            while amt > 0:
                frac = min(frac, amt)
                sample = sample.append(df[df[categories_col] == i].sample(frac=frac))
                amt -= frac
        return df.append(sample)
    def copy_to_output_with_csv(self, out=None, action='copy', oversample=False):
        """
        Same as `copy_to_output` except that it overwrites `df` to be the oversampled DataFrame
        """
        if oversample == True:
            self.df = self.df_oversampled
        self.copy_to_output(out=out, action=action)
        print('Validation indexes are the first',self.len_val,'indexes in the labels.csv file')