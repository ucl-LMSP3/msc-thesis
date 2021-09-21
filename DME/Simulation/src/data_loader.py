import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DeepMEDataset(Dataset):
    def __init__(self, dataframe, random_effects_column_names,
                 group_column_name, y_column_name, n_samples_chosen_per_group):
        """
        Args:
        dataframe: Pandas dataframe, already processed.
        random_effects_column_names: List of names of random effects columns.
        group_column_name: Name of column of group variable.
        y_column_name: Name of column of y variable.
        n_samples_chosen_per_group: Number of samples chosen in each group for interpolation test set.
        """
        self.dataframe = dataframe
        self.random_effects_column_names = random_effects_column_names
        self.random_effects_dim = len(random_effects_column_names)
        self.group_column_name = group_column_name
        self.y_column_name = y_column_name
        self.input_dim = dataframe.shape[1] - self.random_effects_dim - 3 # Minus y, F, group, and random effects columns
        self.n_samples_chosen_per_group = n_samples_chosen_per_group
        self.X, self.Y, self.F, self.Z = self.load_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.F[idx], self.Z[idx], self.n_samples_chosen_per_group[idx]

    def load_data(self):
        X_list, y_list, F_list, Z_list = [], [], [], []
        group_object = self.dataframe.groupby(self.group_column_name)
        grouped_data = [group_object.get_group(x) for x in group_object.groups]
        for group in grouped_data:
            columns_to_be_dropped = self.random_effects_column_names + [self.group_column_name, self.y_column_name, 'F']
            X = group.drop(columns=columns_to_be_dropped).to_numpy(dtype=np.float32).reshape(-1, self.input_dim)
            y = group[self.y_column_name].to_numpy(dtype=np.float32).reshape(-1, 1)
            F = group['F'].to_numpy(dtype=np.float32).reshape(-1)
            Z = group[self.random_effects_column_names].to_numpy(dtype=np.float32).reshape(-1, self.random_effects_dim)
            X_list.append(X)
            y_list.append(y)
            F_list.append(F)
            Z_list.append(Z)
        return X_list, y_list, F_list, Z_list


def fetch_dataloaders(dataframes, random_effects_column_names,
                      group_column_name, y_column_name, n_samples_chosen_per_group):
    """
    Args:
    dataframes (list): Pandas dataframes, already processed.
    random_effects_column_names: List of names of random effects columns.
    group_column_name: Name of column of group variable.
    y_column_name: Name of column of y variable.
    n_samples_chosen_per_group: Number of samples chosen in each group for interpolation test set.
    """
    dataloaders = []
    for i, dataframe in enumerate(dataframes):
        dataset = DeepMEDataset(dataframe, random_effects_column_names, group_column_name, y_column_name, n_samples_chosen_per_group)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloaders.append(dataloader)

    return dataloaders
