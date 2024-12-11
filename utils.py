from torch.utils.data import Dataset
import torch
import os
import scipy.io as scio
import torch.nn
import numpy as np
import random
random.seed(0)

dataset_path = {'seed4': 'H:/SEED_IV/eeg_feature_smooth', 'seed3': 'H:/ExtractedFeatures'}

def norminy(data):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i])
    return dataT.T


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# package the data and label into one class


class CustomDataset(Dataset):
    # initialization: data and label
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    # get the size of data

    def __len__(self):
        return len(self.Data)
    # get the data and label

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return data, label

# mmd loss
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def get_number_of_label_n_trial(dataset_name):
    # global variables
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')


def reshape_data(data, label):
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):
        one_data = np.reshape(np.transpose(
            data[i], (1, 2, 0)), (-1, 310), order='F')
        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label

def get_data_label_frommat(mat_path, dataset_name, session_id):
    _, _, labels = get_number_of_label_n_trial(dataset_name)
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,
                   value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())
    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    return one_sub_data, one_sub_label

def get_allmats_name(dataset_name):

    path = dataset_path[dataset_name]
    sessions = os.listdir(path)
    sessions.sort()
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)
    return path, allmats


def load_data(dataset_name):
    path, allmats = get_allmats_name(dataset_name)
    data = [([0] * 15) for i in range(3)]
    label = [([0] * 15) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = path + '/' + str(i+1) + '/' + allmats[i][j]
            one_data, one_label = get_data_label_frommat(
                mat_path, dataset_name, i)
            data[i][j] = one_data.copy()
            label[i][j] = one_label.copy()
    return np.array(data), np.array(label)

