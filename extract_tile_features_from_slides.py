"""
HE2RNA: Divide whole-slide images in tiles and extract ResNet features
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

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.50 # To use 50% of memory
set_session(tf.Session(config=config))

import os
import numpy as np
import pickle as pkl
import argparse
import openslide
import openslide.deepzoom
import colorcorrect
from joblib import Parallel, delayed
from PIL import Image
from colorcorrect.util import from_pil, to_pil
from colorcorrect import algorithm as cca
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.applications.resnet50 import preprocess_input

def extract_tile_features(level, coord, zoom):
    tile = np.array(zoom.get_tile(level, (coord[1], coord[2])))
    tile = Image.fromarray(tile)
    tile = to_pil(cca.stretch(from_pil(tile)))
    tile = np.array(tile)
    return tile

def save_numpy_features(path2slides, folder, slidename, coords, path):
    model = ResNet50(weights='imagenet', include_top=True)
    model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)

    slide = openslide.OpenSlide(os.path.join(path2slides, folder, slidename))
    zoom = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=0)
    level = int(coords[0, 0])
    tiles = np.array([extract_tile_features(level, coord, zoom) for coord in tqdm(coords)])
    tiles = preprocess_input(tiles)
    X = model.predict(tiles, batch_size=32)
    X = np.concatenate([coords, X], axis=1)
    np.save(os.path.join(path, '0.50_mpp', slidename.split('.')[0] + '.npy'), X)

def process_all_slides(path2slides, tile_coords, path):

    subfolder = {}
    slide_dirs = [d for d in os.listdir(path2slides) if os.path.isdir(os.path.join(path2slides, d))]

    slidenames = []
    subfolders = []

    for d in slide_dirs:
        for f in os.listdir(os.path.join(path2slides, d)):
            if f.endswith('.svs') or f.endswith('.tif') and 'mask' not in f:
                slidenames.append(f)
                subfolders.append(d)

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, '0.50_mpp')):
        os.mkdir(os.path.join(path, '0.50_mpp'))

    for folder, slidename in zip(subfolders, slidenames):
        if slidename in tile_coords.keys():
            save_numpy_features(path2slides, folder, slidename, tile_coords[slidename], path)
        else:
            print(f'Warning: tile coordinates not found for file {slidename}, skipping it')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_slides", help="path to folder containing subfolders with whole-slide images",
                        default='data/PESO/')
    parser.add_argument("--path_to_save_features", help="path to save features as npy files",
                        default='data/PESO_tiles')
    parser.add_argument("--tile_coordinates", help="path to pkl file containing tile coordinates",
                        default='tile_coordinates/tile_coordinates_PESO.pkl')
    args = parser.parse_args()
    path2slides = args.path_to_slides
    path = args.path_to_save_features
    tile_coords = args.tile_coordinates
    process_all_slides(path2slides, pkl.load(open(tile_coords, 'rb')), path)
    
if __name__ == '__main__':

     main()