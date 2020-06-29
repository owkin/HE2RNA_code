"""
HE2RNA: Arrange data and labels into pytorch datasets
Copyright (C) 2020  Owkin Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, TensorDataset, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Compose
from tqdm import tqdm
from joblib import Parallel, delayed
from constant import PATH_TO_TILES, PATH_TO_TRANSCRIPTOME


def load_labels(transcriptome_dataset):
    """Clean up RNAseq data and return labels, genes and patients.
    """
    assert hasattr(transcriptome_dataset, 'transcriptomes'), \
        "Transcriptomes have not been loaded for this dataset"

    to_drop = ['Case.ID', 'Sample.ID', 'File.ID', 'Project.ID']
    df = transcriptome_dataset.transcriptomes.copy()
    patients = df['Case.ID'].values
    projects = df['Project.ID']
    df.drop(to_drop, axis=1, inplace=True)
    genes = df.columns
    df = np.log10(1 + df)
    y = df.values

    return y, genes, patients, projects


def load_and_aggregate_file(file, reduce=True):
    x = np.load(file)
    x = x[:, 3:]
    if reduce:
        x = np.mean(x, axis=0)
    else:
        x = np.concatenate((x, np.zeros((8000 - x.shape[0], 2048)))).transpose(1, 0)
    return x

def load_npy_data(file_list, reduce=True):
    """Load and aggregate data saved as npy files.

    Args
        reduce (bool): If True, perform mean pooling on slide.
            Else, pad every slide with zeros.
    """
    X = np.array(Parallel(n_jobs=32)(delayed(load_and_aggregate_file)(file) for file in tqdm(file_list)))
    return X


def make_dataset(dir, file_list, labels):
    """Associate file names and labels"""
    images = []
    dir = os.path.expanduser(dir)

    for fname, label in zip(file_list, labels):
        path = os.path.join(dir, fname)
        if os.path.exists(path):
            item = (path, label)
            images.append(item)

    return images


class AggregatedDataset(TensorDataset):
    """A subclass of TensorDataset to use for whole-slide analysis
    (with aggregated data).

    Args
        genes (list): List of Ensembl IDs of genes to be used as targets.
        patients (list): list of patient IDs to perform patient split.
    """
    def __init__(self, genes, patients, projects, *tensors):
        super(AggregatedDataset, self).__init__(*tensors)
        self.genes = genes
        self.patients = patients
        self.projects = projects
        self.dim = 2048

    @classmethod
    def match_transcriptome_data(cls, transcriptome_dataset):
        """Use a TranscriptomeDataset object to read corresponding .npy files
        and aggregate tiles.

        Args
            transcriptome_dataset (TranscriptomeDataset)
            binarize (bool): If True, target gene expressions are binarized with
                respect to their median value.
        """
        y, cols, patients, projects = load_labels(transcriptome_dataset)

        file_list = [
            os.path.join(
                PATH_TO_TILES, project.replace('-', '_'),
                '0.50_mpp', filename
            )
            for project, filename in transcriptome_dataset.metadata[['Project.ID', 'Slide.ID']].values
        ]
        X = load_npy_data(file_list)
        return cls(cols, patients, projects, torch.Tensor(X), torch.Tensor(y))


class ToTensor(object):
    """A simple transformation on numpy array to obtain torch-friendly tensors.
    """
    def __init__(self, n_tiles=8000):
        self.n_tiles = n_tiles

    def __call__(self, sample):
        x = torch.from_numpy(sample).float()
        if x.shape[0] > self.n_tiles:
            x = x[:self.n_tiles]
        elif x.shape[0] < self.n_tiles:
            x = torch.cat((x, torch.zeros((self.n_tiles - x.shape[0], 2051))))
        return x.t()


class RemoveCoordinates(object):
    """Remove tile levels and coordinates."""
    def __call__(self, sample):
        return sample[3:]


class TCGAFolder(Dataset):
    """A class similar to torchvision.FolderDataset for dealing with npy files
    of one or several TCGA project(s).

    Args
        genes (list): List of Ensembl IDs of genes to be used as targets.
        patients (list): list of patient IDs to perform patient split.
        projectname (str or None): Project.ID
        file_list (list): list of paths to .npy files containing tiled slides.
        labels (list or np.array): the associated gene expression values.
        transform (callable): Preprocessing of the data.
        target_transform (callable): Preprocessing of the targets.
    """
    def __init__(self, genes, patients, projects, projectname, file_list, labels,
                 transform=Compose([ToTensor(), RemoveCoordinates()]),
                 target_transform=None, masks=None):
        root = PATH_TO_TILES
        samples = make_dataset(root, file_list, labels)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root

        self.patients = patients
        self.projects = projects
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        self.genes = genes
        self.dim = 2048
        self.masks = masks

    @classmethod
    def match_transcriptome_data(cls, transcriptome_dataset, binarize=False):
        projectname = transcriptome_dataset.projectname
        labels, cols, patients, projects = load_labels(transcriptome_dataset)
        file_list = [
            os.path.join(
                PATH_TO_TILES, project.replace('-', '_'),
                '0.50_mpp', filename)
            for project, filename in transcriptome_dataset.metadata[['Project.ID', 'Slide.ID']].values]
        return cls(cols, patients, projects, projectname, file_list, labels)

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.masks is not None:
            mask = self.masks[path.split('/')[-1]]
            idx = np.argsort(mask[:, 0])[::-1]
            sample = np.load(path)[idx] * mask[idx]
        else:
            sample = np.load(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


class H5Dataset(Dataset):
    """A class for using data saved in an hdf5 file.

    Args
        genes (list): List of Ensembl IDs of genes to be used as targets.
        patients (list): list of patient IDs to perform patient split.
        filename (str): path to the hdf5 file containing the data.
        labels (list or np.array): the associated gene expression values.
        max_items (int): Maximum number of tiles to use for training.
    """
    def __init__(self, genes, patients, projects, filename, labels, max_items=8000):
        self.data = h5py.File(filename, 'r')
        self.targets = labels
        self.max_items = max_items
        self.genes = genes
        self.patients = patients
        self.projects = projects
        self.dim = self.data['X'].shape[2]

    def __getitem__(self, index):

        sample = torch.Tensor(self.data['X'][index, :self.max_items, 3:]).float().t()
        target = self.targets[index]

        return sample, target

    def __len__(self):
        return self.data['X'].shape[0]


def patient_split(dataset, random_state=0):
    """Perform patient split of any of the previously defined datasets.
    """
    patients_unique = np.unique(dataset.patients)
    patients_train, patients_valid = train_test_split(
        patients_unique, test_size=0.2, random_state=random_state)
    patients_valid, patients_test = train_test_split(
        patients_valid, test_size=0.5, random_state=random_state)

    indices = np.arange(len(dataset))
    train_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               patients_train[np.newaxis], axis=1)]
    valid_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               patients_valid[np.newaxis], axis=1)]
    test_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                              patients_test[np.newaxis], axis=1)]

    return train_idx, valid_idx, test_idx


def match_patient_split(dataset, split):
    """Recover previously saved patient split
    """
    train_patients, valid_patients, test_patients = split
    indices = np.arange(len(dataset))
    train_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               train_patients[np.newaxis], axis=1)]
    valid_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                               valid_patients[np.newaxis], axis=1)]
    test_idx = indices[np.any(dataset.patients[:, np.newaxis] ==
                              test_patients[np.newaxis], axis=1)]

    return train_idx, valid_idx, test_idx


def patient_kfold(dataset, n_splits=5, random_state=0, valid_size=0.1):
    """Perform cross-validation with patient split.
    """
    indices = np.arange(len(dataset))

    patients_unique = np.unique(dataset.patients)

    skf = KFold(n_splits, shuffle=True, random_state=random_state)
    ind = skf.split(patients_unique)

    train_idx = []
    valid_idx = []
    test_idx = []

    for k, (ind_train, ind_test) in enumerate(ind):

        patients_train = patients_unique[ind_train]
        patients_test = patients_unique[ind_test]

        test_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                       patients_test[np.newaxis], axis=1)])

        if valid_size > 0:
            patients_train, patients_valid = train_test_split(
                patients_train, test_size=valid_size, random_state=0)
            valid_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                            patients_valid[np.newaxis], axis=1)])

        train_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        patients_train[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx


def match_patient_kfold(dataset, splits):
    """Recover previously saved patient splits for cross-validation.
    """

    indices = np.arange(len(dataset))
    train_idx = []
    valid_idx = []
    test_idx = []

    for train_patients, valid_patients, test_patients in splits:

        train_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        train_patients[np.newaxis], axis=1)])
        valid_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                        valid_patients[np.newaxis], axis=1)])
        test_idx.append(indices[np.any(dataset.patients[:, np.newaxis] ==
                                       test_patients[np.newaxis], axis=1)])

    return train_idx, valid_idx, test_idx
