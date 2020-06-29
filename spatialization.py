"""
HE2RNA: Extract prediction of gene expression per tile and compare to ground truth
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
import argparse
import openslide
import openslide.deepzoom
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr, spearmanr


def compute_heatmap(path_to_model, path_to_tile_features):

    X_he = np.load(path_to_tile_features)
    coords = X_he[:, :3]
        
    all_scores = []
    x = torch.Tensor(X_he[np.newaxis].transpose(1, 2, 0))
    clusters = np.arange(X_he.shape[0])

    # Load all models from cross_validation on TCGA
    models = [torch.load(f'{path_to_model}/model_' +
                         str(k) + '/model.pt', map_location='cpu') for k in range(5)]

    for model in tqdm(models):
        all_scores.append(model.conv(x).detach().numpy())

    # Average over genes and cross-val folds
    tile_scores = np.mean(all_scores, axis=(0, 2))[:, 0]
    
    return coords, tile_scores


def display_heatmap(path_to_slide, coords, tile_scores, path=None):

    slide_he = openslide.OpenSlide(path_to_slide)
    print(f'Dimensions of the slide: {slide_he.dimensions}')

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.set_size_inches((15, 10))
    zoom_he = openslide.deepzoom.DeepZoomGenerator(slide_he, tile_size=224, overlap=0)
    im = np.array(slide_he.get_thumbnail((1000, 1000)))
    ax1.imshow(im)
    ax1.set_xticks([])
    ax1.set_yticks([])

    n_tiles = zoom_he.level_tiles[int(coords[0, 0])]
    grid = (np.array(im.shape[:2]) / n_tiles[::-1]) 

    score = tile_scores
    # Clip scores to increase contrast
    score = np.clip(score, np.percentile(score, 10), np.percentile(score, 99))

    mask = np.zeros_like(im[:, :, 0]).astype(float)
    for s, coord in zip(score, coords):
        x = int((coord[2] + 6))
        y = int((coord[1] + 3))
        mask[int(x * grid[0]): int((x + 1) * grid[0]),
             int(y * grid[0]): int((y + 1) * grid[0])] = s
    ims = ax2.imshow(mask, cmap='inferno')
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar = plt.colorbar(ims, ax=ax2)
    ims.set_clim(np.min(mask[mask > 0]), np.max(mask[mask > 0]))
    cbar.ax.tick_params(labelsize=16) 

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def compute_aucs_CRC(path_to_model, path_to_tiles):
    scores = []
    cats = ['LYM', 'ADI', 'STR', 'NORM', 'TUM', 'DEB', 'MUS', 'MUC', 'BACK']
    for cat in tqdm(cats):
        all_scores = []
        X_he = np.load(os.path.join(path_to_tiles, f'{cat}.npy'))
        x = torch.Tensor(X_he.transpose(1, 2, 0))
        clusters = np.arange(X_he.shape[0])

        models = [torch.load(f'{path_to_model}/model_' + str(k) +
                         '/model.pt', map_location='cpu') for k in range(5)]

        for model in models:
            all_scores.append(model.conv(x).detach().numpy())

        all_scores = np.mean(all_scores, axis=(0, 1, 2))
        scores.append(all_scores)
    labels = np.concatenate([np.ones_like(scores[0]), np.zeros_like(np.concatenate(scores[1:]))])
    auc_lym_vs_all = roc_auc_score(labels, np.concatenate(scores))
    print(f'AUC for lymphocytes vs all other classes: {auc_lym_vs_all:.4f}')
    dic = {}
    for i in range(1, 8):
        labels = np.concatenate([np.ones_like(scores[0]), np.zeros_like(scores[i])])
        auc = roc_auc_score(labels, np.concatenate([scores[0], scores[i]]))
        print(f'AUC for lymphocytes vs class {cats[i]}: {auc:.4f}')
        dic[f'AUC LYM vs {cats[i]}'] = auc
    return auc_lym_vs_all, dic


def post_processing(seg):
    seg = seg[:, :, 0]
    seg = (seg > 1).astype(float)
    return np.mean(np.clip(seg, 0, 1))


def compute_correlation_PESO(path_to_model, path_to_tiles, path_to_masks, corr='pearson'):
    scores = []
    gts = []
    files = os.listdir(path_to_tiles)
    ns = np.unique([file.split('_')[1] for file in files])
    models = [torch.load(f'{path_to_model}/model_' + str(k) +
                         '/model.pt', map_location='cpu') for k in range(5)]
    for n in tqdm(ns):

        X_he = np.load(os.path.join(path_to_tiles, '0.50_mpp', f'pds_{n}_HE.npy'))
        coords = X_he[:, :3]
        mask_ = openslide.OpenSlide(os.path.join(path_to_masks, f'pds_{n}_HE_training_mask.tif'))

        zoom_mask = openslide.deepzoom.DeepZoomGenerator(mask_, tile_size=224, overlap=0)

        tile_scores = []
        x = torch.Tensor(X_he[np.newaxis].transpose(1, 2, 0))
        clusters = np.arange(X_he.shape[0])

        for model in models:
            tile_scores.append(model.conv(x).detach().numpy())
        tile_scores = np.mean(tile_scores, axis=(0, 2, 3))
        scores.append(tile_scores)

        gt = []
        for coord in tqdm(coords):
            img_mask = np.array(
                zoom_mask.get_tile(int(coord[0]), (int(coord[1]), int(coord[2]))))
            ep = post_processing(img_mask)
            gt.append(np.mean(ep))
        gt = np.array(gt)
        gts.append(gt)
    gts = np.concatenate(gts)
    scores = np.concatenate(scores)
    if corr == 'pearson':
        return pearsonr(gts, scores)
    elif corr == 'spearman':
        return spearmanr(gts, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", help="dataset on which to carry spatialization experiment, CRC or PESO")
    parser.add_argument("--path_to_model", help="path to the folder containing the models trained by cross-validation",
                        default='epithelium_selection')
    parser.add_argument("--path_to_tiles", help="path to folder containing .npy files of tile features")
    parser.add_argument("--path_to_masks", help="path to folder containing training masks from PESO")
    parser.add_argument("--corr", help="type of correlation to compute, pearson or spearman", default='pearson')
    args = parser.parse_args()
    if args.experiment == 'CRC':
        compute_aucs_CRC(args.path_to_model, args.path_to_tiles)
    elif args.experiment == 'PESO':
        compute_correlation_PESO(args.path_to_model, args.path_to_tiles, args.path_to_masks, args.corr)
    else:
        print("unrecognized experiment")

if __name__ == '__main__':

    main()