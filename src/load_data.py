from os import walk
from torch_geometric.data import Data
import torch
from sklearn.model_selection import train_test_split


def load_data_list(train_size=0.75, val_size=0.15, data_list=None):
    val_size = val_size/(1-train_size)

    train, val_test = train_test_split(data_list, train_size=train_size, random_state=42)
    val, test = train_test_split(val_test, train_size=val_size, random_state=42)

    return [train, val, test]


def load_data_files(directory: str, target: int, exclude_atom=1):
    data_list = []

    file_names = next(walk(directory), (None, None, []))[2]

    for file_name in file_names:
        file_path = directory + fr'\{file_name}'
        file = torch.load(file_path, weights_only=False)
        data_list.append(file)

    data_list_mask = []

    for data_tensor in data_list:
        mask = data_tensor.x != exclude_atom

        filtered_atom_numbers = data_tensor.x[mask]
        filtered_coordinates = data_tensor.pos[mask]

        y = data_tensor.y[0, target].reshape(1, -1)

        data = Data(
            x=filtered_atom_numbers,
            pos=filtered_coordinates,
            y=y
        )

        data_list_mask.append(data)

    return data_list_mask
