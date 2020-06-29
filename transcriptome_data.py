"""
HE2RNA: Match RNAseq data from TCGA with whole-slide images
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
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from constant import PATH_TO_TILES, PATH_TO_TRANSCRIPTOME


class TranscriptomeDataset:
    """A class for dealing with RNAseq data and matching them with available
    slides.

    Args:
        projectname (list): If None, all TCGA projects are included.
        genes (list or None): list of genes Ensembl IDs. If None, all
            available genes are used.
    """

    def __init__(self, projectname=None, genes=None):

        self.projectname = projectname
        self.genes = genes

        transcriptome_metadata = pd.read_csv(
            os.path.join(
                'metadata',
                'samples_description.csv'),
            sep='\t')

        # Select primary tumor samples from the chosen project
        if self.projectname is not None:
            directories = [
                project.replace('_', '-') for project in self.projectname]
            self.transcriptome_metadata = transcriptome_metadata.loc[
                (transcriptome_metadata['Project.ID'].isin(directories)) &
                (transcriptome_metadata['Sample.Type'] == 'Primary Tumor')]
        else:
            self.transcriptome_metadata = transcriptome_metadata.loc[
                transcriptome_metadata['Sample.Type'] == 'Primary Tumor']

        self.image_metadata = self._get_infos_on_tiles(self.projectname)
        self._match_data()

    @classmethod
    def from_saved_file(cls, path, projectname=None, genes=None):
        """Build TranscriptomeDataset instance from a saved csv file.
        """
        if genes is None:
            usecols = None
        else:
            usecols = list(genes) + ['File.ID', 'Sample.ID', 'Case.ID', 'Project.ID']
        transcriptomes = pd.read_csv(path, usecols=usecols)
        if projectname is None:
            projectname = transcriptomes['Project.ID'].apply(
                lambda x: x.replace('-', '_')).unique()
        else:
            transcriptomes = transcriptomes.loc[transcriptomes['Project.ID'].apply(
                lambda x: x.replace('-', '_')).isin(projectname)]
        genes = [col for col in transcriptomes.columns if col.startswith('ENSG')]
        dataset = cls(projectname, genes)
        transcriptomes.sort_values('Sample.ID', inplace=True)
        transcriptomes.reset_index(inplace=True, drop=True)
        dataset.transcriptomes = transcriptomes
        return dataset

    def _get_infos_on_tiles(self, subdirs, zoom='0.50_mpp'):
        """Find all slides tiled at a given level of a TCGA project and return a
        dataframe with their metadata.
        """

        if subdirs is not None:
            df = []
            for subdir in subdirs:
                dir_tiles = os.path.join(PATH_TO_TILES, subdir, zoom)
                filenames = [f for f in os.listdir(dir_tiles) if f.endswith('.npy') and 'mask' not in f]
                case_ids = [f[:12] for f in filenames]
                sample_ids = [f[:16] for f in filenames]
                full_ids = [f.split('.')[0] for f in filenames]

                df.append(pd.DataFrame(
                    {'Project.ID': subdir, 'Case.ID': case_ids, 'Sample.ID_image': sample_ids,
                     'ID': full_ids, 'Slide.ID': filenames}))
            return pd.concat(df)
        else:
            subdirs = []
            for subdir in os.listdir(PATH_TO_TILES):
                if os.path.isdir(os.path.join(PATH_TO_TILES, subdir)) and subdir.startswith('TCGA'):
                    subdirs.append(subdir)
            return self._get_infos_on_tiles(subdirs)

    def _match_data(self):
        """Associate transcriptomes with availables slides.
        """
        self.transcriptome_metadata['Sample'] = self.transcriptome_metadata['Sample.ID'].apply(
            lambda x: x[:-1])
        self.image_metadata['Sample'] = self.image_metadata['Sample.ID_image'].apply(
            lambda x: x[:-1])
        self.transcriptome_metadata.drop('Project.ID', axis=1, inplace=True)
        self.metadata = self.transcriptome_metadata.merge(
            self.image_metadata[['Project.ID', 'Sample', 'Sample.ID_image', 'ID', 'Slide.ID']],
            on='Sample')
        # If several transcriptomes can be associated with a slide, pick only one.
        self.metadata = self.metadata.groupby('Slide.ID').first().reset_index()
        self.metadata.sort_values('Sample.ID', inplace=True)
        self.metadata.reset_index(inplace=True, drop=True)

    def load_transcriptomes(self):
        """Select transcriptomic data of the selected project and genes.
        """
        df = pd.read_csv(os.path.join(
            PATH_TO_TRANSCRIPTOME,
            'transcriptome_fpkmuq_allsamps.csv'), sep='\t', usecols=self.genes, index_col=0)

        df['File.ID'] = df.index
        df = df.merge(self.metadata[['File.ID', 'Sample.ID',
                                     'Case.ID', 'Project.ID']],
                      on='File.ID', how='inner')
        df.sort_values('Sample.ID', inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.transcriptomes = df

        
def main():
    df = []
    path = Path(PATH_TO_TRANSCRIPTOME)
    for f in tqdm(path.glob('*/*.txt')):
        df_ = pd.read_csv(f, sep='\t', header=None, index_col=0)
        df_.columns = [str(f).split('/')[-2]]
        df.append(df_.T)
    df = pd.concat(df)
    df.to_csv(path / 'transcriptome_fpkmuq_allsamps.csv', index=True, sep='\t')
    dataset = TranscriptomeDataset()
    dataset.load_transcriptomes()
    dataset.transcriptomes.to_csv(path / 'all_transcriptomes.csv', index=False)

if __name__ == '__main__':

    main()