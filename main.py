"""
HE2RNA: Train a model to predict gene expression on TCGA slides, either on a single train/valid/test split or in cross-validation 
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
import configparser
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
import copy as cp
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch import optim
from sklearn.metrics import roc_auc_score
from transcriptome_data import TranscriptomeDataset
from wsi_data import load_labels, AggregatedDataset, TCGAFolder, \
    H5Dataset, patient_split, match_patient_split, \
    patient_kfold, match_patient_kfold
from model import HE2RNA, fit, predict
from utils import compute_metrics


class Experiment(object):
    """An class that uses a config file to setup and run a gene expression
    prediction experiment.

    Args:
        configfile (str): Path to the configuration file.
    """

    def __init__(self,
                 configfile='config.ini'):

        # Read configuration file
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

        assert 'main' in self.config.sections(), \
            "No 'main' section in config file"

        if 'path' in self.config['main'].keys():
            self.savedir = self.config['main']['path']
            if not os.path.exists(self.savedir):
                os.mkdir(self.savedir)
        else:
            self.savedir = '.'

        if 'use_saved_model' in self.config['main'].keys():
            self.use_saved_model = self.config['main']['use_saved_model']
        else:
            self.use_saved_model = False

        if 'single_split' in self.config['main'].keys():
            self.split = pkl.load(open(self.config['main']['single_split'], 'rb'))
        else:
            self.split = None
        if 'splits' in self.config['main'].keys():
            self.splits = pkl.load(open(self.config['main']['splits'], 'rb'))
        else:
            self.splits = None

        if 'subsample' in self.config['main'].keys():
            self.subsample = float(self.config['main']['subsample'])
        else:
            self.subsample = None

        if 'p_value' in self.config['main'].keys():
            self.p_value = self.config['main']['p_value']
        else:
            self.p_value = 't_test'
        assert self.p_value in ['empirical', 't_test'], \
            "Unrecognized test, should be 'empirical' or 't_test'"

    def _read_architecture(self):
        model_params = {}
        if 'architecture' in self.config.sections():
            dic = self.config['architecture']
            if 'layers' in dic.keys():
                layers = dic['layers'].split(',')
                model_params['layers'] = [int(dim) for dim in layers]
            if 'dropout' in dic.keys():
                model_params['dropout'] = float(dic['dropout'])
            if 'ks' in dic.keys():
                ks = dic['ks'].split(',')
                model_params['ks'] = [int(k) for k in ks]
            if 'nonlin' in dic.keys():
                if dic['nonlin'] == 'relu':
                    model_params['nonlin'] = nn.ReLU()
                elif dic['nonlin'] == 'tanh':
                    model_params['nonlin'] = nn.Tanh()
                elif dic['nonlin'] == 'sigmoid':
                    model_params['nonlin'] = nn.Sigmoid()
            if 'device' in dic.keys():
                model_params['device'] = dic['device']

        return model_params

    def _read_training_params(self):

        training_params = {}
        if 'training' in self.config.sections():
            dic = self.config['training']
            if 'max_epochs' in dic.keys():
                training_params['max_epochs'] = int(dic['max_epochs'])
            if 'patience' in dic.keys():
                training_params['patience'] = int(dic['patience'])
            if 'batch_size' in dic.keys():
                training_params['batch_size'] = int(dic['batch_size'])
            if 'num_workers' in dic.keys():
                training_params['num_workers'] = int(dic['num_workers'])

        return training_params

    def _setup_optimization(self, model):

        if 'optimization' in self.config.sections():
            dic = self.config['optimization']
            optim_params = {'params': model.parameters(),
                            'lr': float(dic['lr'])}
            if 'momentum' in self.config['optimization'].keys():
                optim_params['momentum'] = float(dic['momentum'])
                optim_params['nesterov'] = True
            if 'weight_decay' in self.config['optimization'].keys():
                optim_params['weight_decay'] = float(dic['weight_decay'])

            if dic['optimizer'] == 'sgd':
                return optim.SGD(**optim_params)
            elif dic['optimizer'] == 'adam':
                return optim.Adam(**optim_params)

        else:
            return optim.Adam(lr=1e-3)

    def _build_dataset(self):

        assert 'data' in self.config.sections(), \
            "'data' not found in config file"
        dic = self.config['data']

        if 'genes' in dic.keys():
            genes = dic['genes']
            if os.path.exists(genes):
                genes = pkl.load(
                    open(genes, 'rb'))
            else:
                genes = genes.split(',')
                for gene in genes:
                    assert gene.startswith('ENSG'), "Unknown gene format"
        else:
            genes = None

        if 'path_to_transcriptome' in dic.keys() and 'projectname' in dic.keys():
            projectname = dic['projectname'].split(',')
            transcriptome_data = TranscriptomeDataset.from_saved_file(
                dic['path_to_transcriptome'], projectname=projectname, genes=genes)
        elif 'path_to_transcriptome' in dic.keys():
            transcriptome_data = TranscriptomeDataset.from_saved_file(
                dic['path_to_transcriptome'], genes=genes)
        elif 'projectname' in dic.keys():
            projectname = dic['projectname'].split(',')
            transcriptome_data = TranscriptomeDataset(projectname, genes)
            transcriptome_data.load_transcriptomes()
        else:
            transcriptome_data = TranscriptomeDataset(None, genes)
            transcriptome_data.load_transcriptomes()

        if 'path_to_data' in dic.keys():

            if dic['path_to_data'].endswith('.pkl'):
                X = pkl.load(open(dic['path_to_data'], 'rb'))
                y, genes, patients, projects = load_labels(transcriptome_data)
                dataset = AggregatedDataset(
                    genes, patients, projects,
                    torch.Tensor(X), torch.Tensor(y))
            elif dic['path_to_data'].endswith('.h5'):
                y, genes, patients, projects = load_labels(transcriptome_data)
                dataset = H5Dataset(
                    genes, patients, projects, dic['path_to_data'], y)

        else:
            dataset = TCGAFolder.match_transcriptome_data(
                transcriptome_data)

        return dataset

    def single_run(self, random_state=0, logdir='./exp'):
        """Experiment with a single train/test split.

        Args:
            random_state (int): Random seed used for splitting the data.
            logdir (str): Path for TensoboardX.

        Returns:
            pandas DataFrame: The metrics per gene.
        """

        model_params = self._read_architecture()
        training_params = self._read_training_params()
        dataset = self._build_dataset()
        evalset = self._build_dataset()
        if dataset.dim == 2051:  # Remove tile levels and coordinates
            model_params['input_dim'] = dataset.dim - 3
        else:
            model_params['input_dim'] = dataset.dim
        model_params['output_dim'] = len(dataset.genes)

        if self.split is None:
            train_idx, valid_idx, test_idx = patient_split(dataset, random_state)
        else:
            train_idx, valid_idx, test_idx = match_patient_split(dataset, self.split)

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(evalset, valid_idx)
        test_set = Subset(evalset, test_idx)

        dic = {}
        for project in dataset.projects.unique():
            if project in ['TCGA-LUAD', 'TCGA-LUSC', 'TCGA_LUAD', 'TCGA_LUSC']:
                dic[project] = 'TCGA-LUNG'
            elif project in ['TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA_KICH', 'TCGA_KIRC', 'TCGA_KIRP']:
                dic[project] = 'TCGA-KIDN'
            elif project in ['TCGA-UCS', 'TCGA-UCEC']:
                dic[project] = 'TCGA-UTER'
            else:
                dic[project] = project
        dataset.projects = dataset.projects.map(dic)

        valid_projects = dataset.projects[valid_idx]
        valid_projects = valid_projects.astype(
            'category').cat.codes.values.astype('int64')
        test_projects = dataset.projects[test_idx].apply(
            lambda x: x.replace('_', '-')).values

        if self.use_saved_model:
            model = torch.load(os.path.join(
                               self.use_saved_model,
                               'model.pt'))
            if 'ks' in model_params.keys():
                model.ks = model_params['ks']
            if 'top_k' in model_params.keys():
                model.top_k = model_params['top_k']
            if 'bottom_ks' in model_params.keys():
                model.bottom_ks = model_params['bottom_ks']
            if 'dropout' in model_params.keys():
                model.do.p = model_params['dropout']

        else:
            # Initialize bias of the last layer with the average target value on the train set
            try:
                model_params['bias_init'] = torch.nn.Parameter(
                    torch.Tensor(
                        np.mean(
                            [sample[1] for sample in train_set], axis=0)
                        ).cuda())
            except ValueError:
                model_params['bias_init'] = torch.nn.Parameter(
                    torch.Tensor(
                        np.mean(
                            [sample[1].numpy() for sample in train_set], axis=0)
                        ).cuda())
            model = HE2RNA(**model_params)
        optimizer = self._setup_optimization(model)

        preds, labels = fit(model,
                            train_set,
                            valid_set,
                            valid_projects,
                            test_set=test_set,
                            params=training_params,
                            optimizer=optimizer,
                            logdir=logdir,
                            path=self.savedir)

        report = {'gene': list(dataset.genes)}

        for project in np.unique(test_projects):
            pred = preds[test_projects == project]
            label = labels[test_projects == project]
            report['correlation_' + project] = compute_metrics(
                label, pred)

        report = pd.DataFrame(report)
        report.to_csv(os.path.join(self.savedir, 'results_single_split.csv'), index=False)
        return report

    def cross_validation(self, n_folds=5, random_state=0, logdir='exp'):
        """N-fold cross-validation.

        Args:
            n (int): Number of folds
            random_state (int): Random seed used for splitting the data.
            logdir (str): Path for TensoboardX.

        Returns:
            pandas DataFrame: The metrics per gene and per fold.
        """

        model_params = self._read_architecture()
        training_params = self._read_training_params()
        dataset = self._build_dataset()
        evalset = self._build_dataset()
        if dataset.dim == 2051:  # Remove tile levels and coordinates
            model_params['input_dim'] = dataset.dim - 3
        else:
            model_params['input_dim'] = dataset.dim
        model_params['output_dim'] = len(dataset.genes)

        if self.subsample is not None:
            np.random.seed(random_state)
            ind = np.random.permutation(len(dataset))[:int(self.subsample * len(dataset))]
            genes = dataset.genes
            patients = dataset.patients[ind]
            projects = dataset.projects[ind].reset_index(drop=True)
            dataset = Subset(dataset, ind)
            dataset.genes = genes
            dataset.patients = patients
            dataset.projects = projects

        if self.splits is None:
            if 'patience' in training_params.keys():
                train_idx, valid_idx, test_idx = patient_kfold(
                    dataset, n_splits=n_folds, valid_size=0.1,
                    random_state=random_state)
            else:
                train_idx, valid_idx, test_idx = patient_kfold(
                    dataset, n_splits=n_folds, valid_size=0,
                    random_state=random_state)
        else:
            train_patients, valid_patients, test_patients = self.splits
            splits = zip(train_patients, valid_patients, test_patients)
            train_idx, valid_idx, test_idx = match_patient_kfold(dataset, splits)

        dic = {}
        for project in dataset.projects.unique():
            if project in ['TCGA-LUAD', 'TCGA-LUSC', 'TCGA_LUAD', 'TCGA_LUSC']:
                dic[project] = 'TCGA-LUNG'
            elif project in ['TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA_KICH', 'TCGA_KIRC', 'TCGA_KIRP']:
                dic[project] = 'TCGA-KIDN'
            elif project in ['TCGA-UCS', 'TCGA-UCEC']:
                dic[project] = 'TCGA-UTER'
            else:
                dic[project] = project
        dataset.projects = dataset.projects.map(dic)

        report = {'gene': list(dataset.genes)}
        if self.p_value == 'empirical':
            random = {'gene': list(dataset.genes)}
        else:
            n_samples = {project: [] for project in dataset.projects}

        for k in range(n_folds):

            train_set = Subset(dataset, train_idx[k])
            test_set = Subset(evalset, test_idx[k])
            if len(valid_idx) > 0:
                valid_set = Subset(evalset, valid_idx[k])
                valid_projects = dataset.projects[valid_idx[k]]
                valid_projects = valid_projects.astype(
                    'category').cat.codes.values.astype('int64')
            else:
                valid_set = None
                valid_projects = None

            test_projects = dataset.projects[test_idx[k]].apply(
                lambda x: x.replace('_', '-')).values

            # Initialize the model and define optimizer
            if self.use_saved_model:
                model = torch.load(os.path.join(
                                   self.use_saved_model,
                                   'model_' + str(k),
                                   'model.pt'))
                if 'ks' in model_params.keys():
                    model.ks = model_params['ks']
                if 'top_k' in model_params.keys():
                    model.top_k = model_params['top_k']
                if 'bottom_ks' in model_params.keys():
                    model.bottom_ks = model_params['bottom_ks']
                if 'dropout' in model_params.keys():
                    model.do.p = model_params['dropout']

            else:
                # Initialize bias of the last layer with the average target value on the train set
                try:
                    model_params['bias_init'] = torch.nn.Parameter(
                        torch.Tensor(
                            np.mean(
                                [sample[1] for sample in train_set], axis=0)
                        ).cuda())
                except ValueError:
                    model_params['bias_init'] = torch.nn.Parameter(
                        torch.Tensor(
                            np.mean(
                                [sample[1].numpy() for sample in train_set], axis=0)
                        ).cuda())
                model = HE2RNA(**model_params)
            optimizer = self._setup_optimization(model)

            # Train model
            preds, labels = fit(model,
                                train_set,
                                valid_set,
                                valid_projects,
                                test_set=test_set,
                                params=training_params,
                                optimizer=optimizer,
                                logdir=logdir,
                                path=os.path.join(
                                    self.savedir,
                                    'model_' + str(k)))

            # Compute metrics for each fold
            for project in np.unique(test_projects):
                pred = preds[test_projects == project]
                label = labels[test_projects == project]
                report['correlation_' + project + '_fold_' + str(k)] = compute_metrics(
                    label, pred)

        report = pd.DataFrame(report)
        report.to_csv(os.path.join(self.savedir, 'results_per_fold.csv'), index=False)

        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the configuration file")
    parser.add_argument(
        "--run", help="type of experiment, 'single_run' or 'cross_validation'",
        default='single_run')
    parser.add_argument(
        "--n_folds", help="number of folds for 'cross_validation'",
        default=5)
    parser.add_argument(
        "--logdir", help="path to the directory used by TensoboardX",
        default='./exp')
    parser.add_argument(
        "--rs", help="random state",
        default=0)
    args = parser.parse_args()
    print("Using configuration defined in {}".format(args.config))
    for config in args.config.split(','):
        exp = Experiment(config)

        assert args.run in ['single_run', 'cross_validation'], \
            "Unrecognized experiment, must be either 'single_run' or 'cross_validation"
        if args.run == 'single_run':
            report = exp.single_run(logdir=args.logdir)
        elif args.run == 'cross_validation':
            report = exp.cross_validation(
                n_folds=int(args.n_folds),
                random_state=int(args.rs), logdir=args.logdir)
        print(report)


if __name__ == '__main__':

    main()
