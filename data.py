import torch
from torch.utils.data.dataset import Dataset
from numpy import genfromtxt
import pandas as pd
import numpy as np

class pose_set(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            data_csv_path (string): path to the folder where data_csvs are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the data_csv paths
        self.data_csv_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get data_csv name from the pandas df
        single_data_csv_name = self.data_csv_arr[index]
        # Open data_csv
        data_as_np = genfromtxt(single_data_csv_name, delimiter=',')
        data_as_tensor = torch.from_numpy(data_as_np.astype(float))
        # Get label(class) of the data_csv based on the cropped pandas column
        single_data_csv_label=self.label_arr[index]

        return (data_as_tensor, single_data_csv_label)

    def __len__(self):
        return self.data_len

