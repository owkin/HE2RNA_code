# Gene expression prediction

Predict gene expression from WSIs taken from TCGA with HE2RNA [1]. The model takes as inputs arrays of size n_tiles * 2048, where n_tiles = 100 when super-tile preprocessing is used, and n_tiles = 8,000 when all tiles are treated separately. The model is implemented as a succession of 1D convolution (equivalent to an MLP shared among all tiles).
Additionally, Model interpretability can be explored at: https://owkin.com/he2rna-result-visualization/.

## Installation

Create a virtual environment and install the required packages (the variable CUDA_TOOLKIT_ROOT_DIR is needed to install libKMcuda):
```bash
python3 -m venv .env
source .env/bin/activate

export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
pip install -r requirements.txt
```
NOTE: code was run with python 3.7.4

## Data collection and preprocessing

To ensure reproducibility of the results, coordinates of the tiles used in the paper (necessary to extract tile images and features from whole-slide images) are provided in the archive tile_coordinates.gz.

EDIT: due to an issue related to data quota, file tile_coordinates.gz should be downloaded instead from https://drive.google.com/file/d/1PJsUv1SQieJs7hqtWOqW68v1K9c-mIF6/view?usp=sharing.

To uncompress it, run
```bash
tar -xzvf tile_coordinates.gz
```
Splits used in the paper are also provided in patient_splits.pkl.

### TCGA

We originally downloaded the whole-slide images from the TCGA data portal https://portal.gdc.cancer.gov/ via the gdc-client tool. To access all of TCGA data used in this work, follow the steps described below.
First, create a folder to store data.
```bash
mkdir data
```
Paths to folders containing slides, tile features and RNAseq data should be consistent with the contant of file constant.py. If data is saved in a different location, constant.py has to be modified accordingly, as well as the example config files.

#### Download TCGA slides
Go to the dedicated folder to store files (all FFPE slides from TCGA is approx. 10To)
```bash
cd data
mkdir TCGA_slides/
cd TCGA_slides/
```
For each project, create a subfolder, e.g.
```bash
mkdir TCGA_LIHC/
cd TCGA_LIHC/
```
Download images using the corresponding manifest:
```bash
gdc-client download -m gdc_manifests/gdc_manifest.2018-06-26_TCGA-LIHC.txt
```
Frozen slides from COAD and READ have been grouped together in CRC.
```bash
mkdir TCGA_CRC_frozen/
cd TCGA_CRC_frozen/
gdc-client download -m gdc_manifests/gdc_manifest.CRC_frozen_slides.txt
```
   
#### Tile feature extraction
The code in extract_tile_features_from_slides.py is designed to extract resnet features of tile images directly from whole-slide images, using the coordinates of the tiles in Openslide format. To extract tile features from WSIs from a given TCGA project, e.g. LIHC, run:
```bash
mkdir TCGA_tiles/

python extract_tile_features_from_slides.py --path_to_tiles /TCGA_slides/TCGA_LIHC --tile_coordinates tile_coordinates/tile_coordinates_TCGA_LIHC.pkl --path_to_save_features TCGA_tiles/TCGA_LIHC
```

#### Download and preprocess RNAseq data
Create a folder to store rnaseq data and download transcriptomes:
```bash
cd data
mkdir TCGA_transcriptome
cd TCGA_transcriptome
gdc-client download -m gdc_manifests/gdc_manifest.2018-03-13_alltranscriptome.txt
```
At this stage, there should be one folder per sample, containing a .gz archive. Extract the archives, using for instance gunzip
```bash
gunzip */*.txt.gz
```
To make things more convenient, we already save a file containing transcriptomes matched to whole-slide images, using
```bash
python transcriptome_data.py
```

#### Supertile preprocessing
Finally, once all previous steps have been performed, supertile preprocessing can be performed using the following command (the csv file containing transcriptome is used here to ensure consistency between preprocessed image samples and RNAseq data),
```bash
python supertile_preprocessing.py --path_to_slides data/TCGA_slides --path_to_transcriptome data/TCGA_transcriptome/all_transcriptomes.csv --path_to_save_processed_data data/TCGA_100_supertiles.h5 --n_tiles 100
```

### 100,000 histological images of human colorectal cancer and healthy tissue
The dataset '100,000 histological images of human colorectal cancer and healthy tissue' [2] is available from https://zenodo.org/record/1214456#.XpgF4m46--w. The file we use here is NCT-CRC-HE-100K-NONORM.zip. Download this file and unzip it. You should have a folder (e.g. data/NCT-CRC-HE-100K-NONORM) containing one subfolder per class (ADI, LYM, etc...).

The code in extract_tile_features.py is designed to extract resnet features from those tile images

```bash
python extract_tile_features.py --path_to_tiles data/NCT-CRC-HE-100K-NONORM --path_to_save_features data/NCT-CRC-HE-100K-NONORM_tiles
```

### PESO
The Prostate Epithelium Segmentation dataset (PESO) [3] (whole-slide images and segmentation masks) are available from https://zenodo.org/record/1485967#.Xusr2PI6--x (peso_training_wsi_x.zip and peso_training_masks.zip). Download and unzip those files in a folder (e.g. data/PESO) so that this folder contains subfolders named peso_training_wsi_x/

he code in extract_tile_features_from_slides.py can be used to extract features from the PESO dataset, using tile_coordinates_PESO.pkl

```bash
python extract_tile_features_from_slides.py --path_to_slides data/PESO --tile_coordinates tile_coordinates/tile_coordinates_PESO.pkl --path_to_save_features data/PESO_tiles
```

## Gene expression prediction experiment

To run an experiment, write first a config file or use one of the examples available in folder condigs. 

* config_all_genes.ini: simultaneous prediction of all genes on all TCGA data, using super-tile-preprocessed data.
* config_CD3_all_TCGA.ini: prediction of CD3 genes on all TCGA data, using super-tile-preprocessed data.
* config_CD3_selection.ini: prediction of CD3 genes on a subset of cancers (COAD/LIHC/PRAD/LUAD/LUSC/BRCA), using all available tiles (8,000) per slide, and starting training from checkpoint previously saved.
Similarly for CD19/CD20 genes, epithelium genes (TP63, KRT8 and KRT18) and MKI67.

Launch experiment with a single train-test split:
```bash
python main.py --config <config_file> --run single_run --logdir ./exp
```
Launch cross-validation:
```bash
python main.py --config <config_file> --run cross_validation --n_folds 5 --logdir ./exp
```
Launch TensorboardX for visualizing training curves
```bash
tensorboard --logdir=./exp --port=6006
```

Results will be saved in the specified path as follows:
* for a single train/valid/test split, the model will be saved as model.pt and the correlation per gene and cancer type will be saved as results_single_split.csv
* for a cross-validation, each model will be saved in a dedicated folder model_i/model.pt, the correlation per gene, cancer type and fold will be saved as results_per_fold.csv.

### Config file options

* [main]
	* path: Path to the directory where model's weights will be saved.
	* use_saved_model (optional): Path to previous experiment to reload saved models
	* splits (optional): Path to Pickle file containing saved patient splits for cross-validation, useful in particular when finetuning a model on a subset of the data, to ensure consistency of the train and test set with those used for pretraining.
    * single_split (optional): Path to Pickle file containing saved patient split for single run

* [data]
	* genes (optional): List of coma-separated Ensembl IDs, or path to a pickle file containing such a list. If None, all available genes with nonzero median expression are used.
	* path_to_transcriptome (optional): If None, build targets from projectname and list of genes. Otherwise, load transcriptome data from a saved csv file.
	* path_to_data (optional): Path to the data, saved either in a pickle file (for aggregated data) or in an hdf5 file. If None, build the dataset from .npy files.

* [architecture]
	* layers: Integers defining the number of feature maps of the model's 1D convolutional layers
	* dropout: Float between 0 and 1.
	* ks: List of ks to sample from
	* nonlin: 'relu', 'sigmoid' or 'tanh'.
	* device: 'cpu' or 'cuda'.

* [training]
	* max_epochs: Integer, defaults to 200.
    * patience: Integer, defaults to 20.
    * batch_size: Integer, defaults to 16.
    * num_workers: number of workers used for loading batches, defaults to 0 (value should be 0 when working with hdf5-stored data)

* [optimization]
	* algo: 'sgd' or 'adam'.
	* lr: Float.
	* momentum: Float, optional
    
## Spatialization of gene expression

### Spatialization of lymphocyte genes in colorectal cancer

Once a model has been trained to predict the expression of genes specifically expressed by lymphocytes (for instance CD3), the following script can be used to compute the AUCs for distinguishing tiles labelled with lymphocytes (LYM) from other categories
```bash
python spatialization.py --experiment CRC --path_to_model CD3_selection --path_to_tiles data/NCT-CRC-HE-100K-NONORM_tiles
```

### Spatialization of epithelium genes in prostate adenocarcinoma

Once a model has been trained to predict the expression of genes specifically expressed by the epithelium in prostate,
the following script can be used to compare the average expression predicted by the model for those genes and the ground truth segmentation of epithelium
```bash
python spatialization.py --experiment PESO --path_to_model epithelium_selection --path_to_tiles data/PESO_tiles --path_to_masks data/PESO/peso_training_masks --corr pearson
```

## MSI prediction

This part is relatively independant. All that is needed here is:
* preprocessed tiles from a dataset with MSI status: COAD(FFPE or frozen), READ (FFPE or frozen) or STAD (FFPE)
* rnaseq data from this dataset
```bash
python msi_prediction.py --cancer_types COAD READ --type_of_slides FFPE --msi_l 0 --Nsplit 50 --Ncval_all 10 --Ncval 10 --n_internsplit_A 3 --n_internsplit_B 3 --n_epoch 50
```
Note: for this part, tile features from CRC frozen slides are expected to be located in PATH_TO_TILES/TCGA_CRC_frozen.


## References

[1] Schmauch, B., Romagnoni, A., Pronier, E., Saillard, C., Maillé, P., Calderaro, J., ... & Courtiol, P. (2019). Transcriptomic learning for digital pathology. bioRxiv, 760173.

[2] Kather, J. N et al. 100,000 histological images of human colorectal cancer and healthy tissue (Version v0.1). Zenodo. http://doi.org/10.5281/zenodo.1214456 (2018).

[3] Bulten, W., et al. PESO: Prostate Epithelium Segmentation on H&E-stained prostatectomy whole slide images (Version 1). Zenodo. http://doi.org/10.5281/zenodo.1485967 (2018).

# License

GPL v3.0
