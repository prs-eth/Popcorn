"""
Code is adapted from https://github.com/SebastianHafner/DDA_UrbanExtraction
Modified: Arno Rüegg, Nando Metzger
"""
import torch
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod
import affine
import numpy as np
from ..utils import augmentations, geofiles
import random
from torch.utils.data import Sampler

import json
import os


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


class LabeledUnlabeledSampler(Sampler):
    """
    Samples a batch of labeled and unlabeled data
    """
    def __init__(self, labeled_indices, unlabeled_indices, batch_size):
        """
        input:
            labeled_indices: list of indices of labeled data points
            unlabeled_indices: list of indices of unlabeled data points
            batch_size: batch size
        """

        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.batch_size = batch_size

    def __iter__(self):
        """
        Returns an iterator that yields a batch of labeled and unlabeled data
        input:
            None
        output:
            iterator that yields a batch of labeled and unlabeled data
        """
        # Number of labeled and unlabeled data points
        labeled_batch_size = self.batch_size // 2
        unlabeled_batch_size = self.batch_size - labeled_batch_size

        # legth definition
        length = len(self.labeled_indices) // labeled_batch_size

        # Sample labeled data points
        labeled_batches = [random.sample(self.labeled_indices, labeled_batch_size) for _ in range(length)]
        unlabeled_batches = [random.sample(self.unlabeled_indices, unlabeled_batch_size) for _ in range(length)]
 
        mixed_batches = torch.concat([torch.tensor(labeled_batches), torch.tensor(unlabeled_batches)],1).tolist()

        # random.shuffle(mixed_batches)
        return iter(batch for batches in mixed_batches for batch in batches)
    
    def __len__(self):
        return len(self.labeled_indices)*2
    


class AbstractUrbanExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.root_path = Path(cfg.PATHS.DATASET)

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = ['VV', 'VH']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)

        if cfg.PATHS.DATASET.endswith('new') or cfg.PATHS.DATASET.endswith('new2') or cfg.PATHS.DATASET.endswith('new3'):
            self.s2_indices = [0,1,2,3]
            self.new = True
        else:
            s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)
            self.new = False

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _get_sentinel1_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel1' / f'sentinel1_{site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, site, patch_id):
        file = self.root_path / site / 'sentinel2' / f'sentinel2_{site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, site, patch_id):
        label = self.cfg.DATALOADER.LABEL
        label_file = self.root_path / site / label / f'{label}_{site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(label_file)
        img = img > 0
        return np.nan_to_num(img).astype(np.float32), transform, crs

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]


# dataset for urban extraction with building footprints
class UrbanExtractionDataset(AbstractUrbanExtractionDataset):

    def __init__(self, cfg, dataset: str, include_projection: bool = False, no_augmentations: bool = False,
                 include_unlabeled: bool = True):
        super().__init__(cfg)

        self.dataset = dataset
        if dataset == 'training':
            self.sites = list(cfg.DATASET.TRAINING)
            # using parameter include_unlabeled to overwrite config
            if include_unlabeled and cfg.DATALOADER.INCLUDE_UNLABELED:
                self.sites += cfg.DATASET.UNLABELED
        elif dataset == 'validation':
            self.sites = list(cfg.DATASET.VALIDATION)
        else:  # used to load only 1 city passed as dataset
            self.sites = [dataset]

        self.no_augmentations = no_augmentations
        if no_augmentations:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform = augmentations.compose_transformations(cfg)

        # load normalization stats
        self.dataset_stats = load_json(os.path.join('configs', 'dataset_stats.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = np.array(val)
            else:
                self.dataset_stats[mkey] = np.array(val)

        self.samples = []
        for site in self.sites:
            samples_file = self.root_path / site / 'samples.json'
            metadata = geofiles.load_json(samples_file)
            samples = metadata['samples']
            # making sure unlabeled data is not used as labeled when labels exist
            if include_unlabeled and site in cfg.DATASET.UNLABELED:
                for sample in samples:
                    sample['is_labeled'] = False
            self.samples += samples

        self.length = len(self.samples)
        self.n_labeled = len([s for s in self.samples if s['is_labeled']])

        self.crop_size = cfg.AUGMENTATION.CROP_SIZE

        self.include_projection = include_projection

        self.ind_labeled = [ind for ind, i in enumerate(self.samples) if i.get('is_labeled')]
        self.ind_unlabeled = [ind for ind, i in enumerate(self.samples) if not i.get('is_labeled')]

    def __getitem__(self, index, aug=True):

        # loading metadata of sample
        sample = self.samples[index]
        is_labeled = sample['is_labeled']
        patch_id = sample['patch_id']
        site = sample['site']
        img_weight = float(sample['img_weight'])
        mode = self.cfg.DATALOADER.MODE

        if mode == 'optical':
            img, geotransform, crs = self._get_sentinel2_data(site, patch_id)
        elif mode == 'sar':
            img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
        else:
            s1_img, geotransform, crs = self._get_sentinel1_data(site, patch_id)
            s2_img, _, _ = self._get_sentinel2_data(site, patch_id)

            if self.new:
                s1_img = ((s1_img - self.dataset_stats["sen1"]['mean'] ) / self.dataset_stats["sen1"]['std']).astype(np.float32)
                s2_img = ((s2_img - self.dataset_stats["sen2springNIR"]['mean'] ) / self.dataset_stats["sen2springNIR"]['std']).astype(np.float32)

            img = np.concatenate([s1_img, s2_img], axis=-1)

        if is_labeled:
            label, _, _ = self._get_label_data(site, patch_id)
        else:
            label = np.zeros((self.crop_size, self.crop_size, 1), dtype=np.float32)
        if aug==False:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        img, label = self.transform((img, label))
        item = {
            'x': img,
            'y': label,
            'site': site,
            'patch_id': patch_id,
            'is_labeled': sample['is_labeled'],
            'image_weight': img_weight,
        }

        if self.include_projection:
            item['transform'] = geotransform
            item['crs'] = str(crs)

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        labeled_perc = self.n_labeled / self.length * 100
        return f'Dataset with {self.length} samples ({labeled_perc:.1f} % labeled) across {len(self.sites)} sites.'


class AbstractSpaceNet7Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.root_path = Path(cfg.PATHS.DATASET) / 'spacenet7'

        # getting patches
        samples_file = self.root_path / 'samples.json'
        metadata = geofiles.load_json(samples_file)
        self.samples = metadata['samples']
        self.length = len(self.samples)

        # getting regional information
        regions_file = self.root_path / 'spacenet7_regions.json'
        self.regions = geofiles.load_json(regions_file)

        self.transform = transforms.Compose([augmentations.ImageCroptest(), augmentations.Numpy2Torch()])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        s1_bands = metadata['sentinel1_features']
        s2_bands = metadata['sentinel2_features']
        self.s1_indices = self._get_indices(s1_bands, cfg.DATALOADER.SENTINEL1_BANDS)

        if cfg.PATHS.DATASET.endswith('new') or cfg.PATHS.DATASET.endswith('new2') or cfg.PATHS.DATASET.endswith('new3'):
            self.s2_indices = [0,1,2,3]
            self.new = True
        else:
            s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)
            self.new = False
        # self.s2_indices = self._get_indices(s2_bands, cfg.DATALOADER.SENTINEL2_BANDS)
        

        # load normalization stats
        self.dataset_stats = load_json(os.path.join('configs', 'dataset_stats.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = np.array(val)
            else:
                self.dataset_stats[mkey] = np.array(val)

                
    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    def _get_sentinel1_data(self, aoi_id):
        file = self.root_path / 'sentinel1' / f'sentinel1_{aoi_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, aoi_id):
        file = self.root_path / 'sentinel2' / f'sentinel2_{aoi_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, aoi_id):
        label = self.cfg.DATALOADER.LABEL
        label_file = self.root_path / label / f'{label}_{aoi_id}.tif'
        img, transform, crs = geofiles.read_tif(label_file)
        img = img > 0
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def get_index(self, aoi_id: str):
        for i, sample in enumerate(self.samples):
            if sample['aoi_id'] == aoi_id:
                return i

    def _get_region_index(self, aoi_id: str) -> int:
        return self.regions['data'][aoi_id]

    def get_region_name(self, aoi_id: str) -> str:
        index = self._get_region_index(aoi_id)
        return self.regions['regions'][str(index)]

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class SpaceNet7Dataset(AbstractSpaceNet7Dataset):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        aoi_id = sample['aoi_id']

        # loading images
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = self._get_sentinel2_data(aoi_id)
        elif mode == 'sar':
            img, _, _ = self._get_sentinel1_data(aoi_id)
        else:  # fusion baby!!! # haha, Nice comment @SebastianHafner
            s1_img, _, _ = self._get_sentinel1_data(aoi_id)
            s2_img, _, _ = self._get_sentinel2_data(aoi_id)
 
            if self.new:
                s1_img = ((s1_img - self.dataset_stats["sen1"]['mean'] ) / self.dataset_stats["sen1"]['std']).astype(np.float32)
                s2_img = ((s2_img - self.dataset_stats["sen2springNIR"]['mean'] ) / self.dataset_stats["sen2springNIR"]['std']).astype(np.float32)


            img = np.concatenate([s1_img, s2_img], axis=-1)

        label, geotransform, crs = self._get_label_data(aoi_id)
        img, label = self.transform((img, label))

        item = {
            'x': img,
            'y': label,
            'aoi_id': aoi_id,
            'country': sample['country'],
            'region': self.get_region_name(aoi_id),
            'transform': geotransform,
            'crs': str(crs)
        }

        return item


# dataset for classifying a scene
class TilesInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, site: str):
        super().__init__()

        self.cfg = cfg
        self.site = site

        self.root_dir = Path(cfg.PATHS.DATASET)
        self.transform = transforms.Compose([augmentations.Numpy2Torch()])

        # getting all files
        samples_file = self.root_dir / site / 'samples.json'
        metadata = geofiles.load_json(samples_file)
        self.samples = metadata['samples']
        self.length = len(self.samples)

        self.patch_size = metadata['patch_size']

        # computing extent
        patch_ids = [s['patch_id'] for s in self.samples]
        self.coords = [[int(c) for c in patch_id.split('-')] for patch_id in patch_ids]
        self.max_y = max([c[0] for c in self.coords])
        self.max_x = max([c[1] for c in self.coords])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(metadata['sentinel1_features'], cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(metadata['sentinel2_features'], cfg.DATALOADER.SENTINEL2_BANDS)
        if cfg.DATALOADER.MODE == 'sar':
            self.n_features = len(self.s1_indices)
        elif cfg.DATALOADER.MODE == 'optical':
            self.n_features = len(self.s2_indices)
        else:
            self.n_features = len(self.s1_indices) + len(self.s2_indices)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        patch_id_center = sample['patch_id']

        y_center, x_center = patch_id_center.split('-')
        y_center, x_center = int(y_center), int(x_center)

        extended_patch = np.zeros((3 * self.patch_size, 3 * self.patch_size, self.n_features), dtype=np.float32)

        for i in range(3):
            for j in range(3):
                y = y_center + (i - 1) * self.patch_size
                x = x_center + (j - 1) * self.patch_size
                patch_id = f'{y:010d}-{x:010d}'
                if self._is_valid_patch_id(patch_id):
                    patch = self._load_patch(patch_id)
                else:
                    patch = np.zeros((self.patch_size, self.patch_size, self.n_features), dtype=np.float32)
                i_start = i * self.patch_size
                i_end = (i + 1) * self.patch_size
                j_start = j * self.patch_size
                j_end = (j + 1) * self.patch_size
                extended_patch[i_start:i_end, j_start:j_end, :] = patch

        if sample['is_labeled']:
            label, _, _ = self._get_label_data(patch_id_center)
        else:
            dummy_label = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32)
            label = dummy_label
        extended_patch, label = self.transform((extended_patch, label))

        item = {
            'x': extended_patch,
            'y': label,
            'i': y_center,
            'j': x_center,
            'site': self.site,
            'patch_id': patch_id_center,
            'is_labeled': sample['is_labeled']
        }

        return item

    def _is_valid_patch_id(self, patch_id):
        patch_ids = [s['patch_id'] for s in self.samples]
        return True if patch_id in patch_ids else False

    def _load_patch(self, patch_id):
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = self._get_sentinel2_data(patch_id)
        elif mode == 'sar':
            img, _, _ = self._get_sentinel1_data(patch_id)
        else:  # fusion baby!!!
            s1_img, _, _ = self._get_sentinel1_data(patch_id)
            s2_img, _, _ = self._get_sentinel2_data(patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)
        return img

    def _get_sentinel1_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel1' / f'sentinel1_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel2' / f'sentinel2_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, patch_id):
        label = self.cfg.DATALOADER.LABEL
        label_file = self.root_dir / self.site / label / f'{label}_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(label_file)
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def get_arr(self, dtype=np.uint8):
        height = self.max_y + self.patch_size
        width = self.max_x + self.patch_size
        return np.zeros((height, width, 1), dtype=dtype)

    def get_geo(self):
        patch_id = f'{0:010d}-{0:010d}'
        # in training and validation set patches with no BUA were not downloaded -> top left patch may not be available
        if self._is_valid_patch_id(patch_id):
            _, transform, crs = self._get_sentinel1_data(patch_id)
        else:
            # use first patch and covert transform to that of uupper left patch
            patch = self.samples[0]
            patch_id = patch['patch_id']
            i, j = patch_id.split('-')
            i, j = int(i), int(j)
            _, transform, crs = self._get_sentinel1_data(patch_id)
            x_spacing, x_whatever, x_start, y_whatever, y_spacing, y_start, *_ = transform
            x_start -= (x_spacing * j)
            y_start -= (y_spacing * i)
            transform = affine.Affine(x_spacing, x_whatever, x_start, y_whatever, y_spacing, y_start)
        return transform, crs

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'
