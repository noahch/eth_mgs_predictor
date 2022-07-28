from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder


class MGSDataset(Dataset):

    def __init__(self, ids, config, transform=None):

        df = pd.read_csv(config.dataset.dataset_labels_filename)
        self.config = config
        self.data = df[df['id'].isin(ids)]
        self.data.reset_index(drop=True, inplace=True)
        if config.model.mgs_attributes and config.model.mgs_one_hot:
            self.mgs_attr = ['eye', 'nose', 'cheek', 'whiskers', 'ears']
            self.encode_one_hot()
        self.root_dir = config.dataset.dataset_image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['index'][:-4] + '.jpg')
        image = Image.open(img_name).convert('RGB')

        if self.config.model.mgs_attributes == 0:
            target = self.data.iloc[idx]['pain']
        else:
            if self.config.model.mgs_one_hot == 1:
                target = self.data.iloc[idx][[a for sub in [[f'{y}_0', f'{y}_1', f'{y}_2'] for y in self.mgs_attr] for a in sub]].to_numpy()
                target = torch.tensor(target.astype(np.int8))
            else:
                raise Exception("not implemented")

        if self.transform:
            image = self.transform(image)
        # TODO: Check if it does not mess up non mgs
        return image, target

    def encode_one_hot(self):

        # creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        temp_df = self.data
        final_df = self.data

        for mgs in self.mgs_attr:
            # perform one-hot encoding on 'team' column
            encoder_df = pd.DataFrame(encoder.fit_transform(temp_df[[mgs]]).toarray())
            # encoder_df.rename(columns = {'0':'eye_0', '1':'eye_1', '2':'eye_2'}, inplace = True)
            encoder_df.columns = [f'{mgs}_0', f'{mgs}_1', f'{mgs}_2']
            # merge one-hot encoded columns back with original DataFrame
            final_df = final_df.join(encoder_df)

        self.data = final_df
