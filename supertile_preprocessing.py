"""
HE2RNA: Apply super-tile preprocessing to ResNet features of tiles extracted from whole-slide images
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
import numpy as np
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm
from libKMCUDA import kmeans_cuda
from wsi_data import TCGAFolder, ToTensor
from transcriptome_data import TranscriptomeDataset


def cluster_dataset(dataset, n_tiles=100,
                    path_to_data='data/TCGA_slic_100.h5'):
    """Perform KMeans on each tiles to create 'supertiles'. Supertile
    features are obtained by averaging resnet features.

    Args
        dataset (torch.utils.data.Dataset)
        n_tiles (int): number of supertiles to generate
        path_to_data (str): path to hdf5 file to save the clustered dataset
    """

    file = h5py.File(path_to_data, 'w')
    file.create_dataset('X', (len(dataset), n_tiles, 2051))
    file.create_dataset('cluster_attribution', (len(dataset), 8000))

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=16,)

    n = 0

    for x, y in tqdm(dataloader):
        x = x[0].numpy().T
        # Remove padding
        mask = (x[:, 0] > 0)
        x = x[mask]
        c = x[:, :3]
        centroids, clusters = kmeans_cuda(x[:, 1:3].astype('float32'),
                                          min(n_tiles, x.shape[0]),
                                          yinyang_t=0,
                                          verbosity=0)
        new_x = []
        new_c = []
        for cl in np.unique(clusters):
            if np.sum(clusters == cl) > 0:
                new_x.append(np.mean(x[clusters == cl, 3:], axis=0))
                new_c.append(np.mean(c[clusters == cl], axis=0))
        x = np.array(new_x)
        c = np.array(new_c)
        x = np.concatenate([c, x], axis=1)
        if len(x) < n_tiles:
            x = np.concatenate([x, np.zeros((n_tiles - len(x), 2051))])

        file['X'][n] = x
        file['cluster_attribution'][n, :len(clusters)] = clusters

        n += 1
    file.close()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_transcriptome", help="path to transcriptome data saved as a csv file",
                        default='data/TCGA_transcriptome/all_transcriptomes.csv')
    parser.add_argument("--path_to_save_processed_data", help="path where supertile-preprocessed data should be saved",
                        default='data/TCGA_slic_100.h5')
    parser.add_argument("--n_tiles", help="number of supertiles",
                        default=100, type=int)
    args = parser.parse_args()
    rna_data = TranscriptomeDataset.from_saved_file(args.path_to_transcriptome, genes=[])
    histo_data = TCGAFolder.match_transcriptome_data(rna_data)
    histo_data.transform = ToTensor()
    cluster_dataset(histo_data, n_tiles=args.n_tiles, path_to_data=args.path_to_save_processed_data)


if __name__ == '__main__':

    main()
