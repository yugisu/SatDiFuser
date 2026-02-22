import importlib
import warnings
from torch.utils.data import Dataset
import torch


def get_sample_center_latlon(sample):
    """
    Extract the WGS84 (lat, lon) centroid of the first band in a geobench
    ``Sample``.  Returns ``(None, None)`` when the band has no geotransform
    or CRS (e.g. a dataset that was not geo-referenced).

    Requires ``rasterio`` (bundled with geobench).
    """
    try:
        from rasterio.warp import transform as rio_transform
        from rasterio.crs import CRS

        band = sample.bands[0]
        if band.transform is None or band.crs is None:
            return None, None

        h, w = band.data.shape[:2]
        # Centre pixel in the band's native CRS using the affine transform:
        #   X = c + col*a + row*b,  Y = f + row*e + col*d
        cx = band.transform.c + (w / 2) * band.transform.a + (h / 2) * band.transform.b
        cy = band.transform.f + (h / 2) * band.transform.e + (w / 2) * band.transform.d

        wgs84 = CRS.from_epsg(4326)
        lons, lats = rio_transform(band.crs, wgs84, [cx], [cy])
        return float(lats[0]), float(lons[0])
    except Exception as exc:
        warnings.warn(f"get_sample_center_latlon: could not extract coordinates — {exc}")
        return None, None

DATASET_REGISTRY = {
    "meurosat": "m_eurosat.EuroSAT",
    "mbigearthnet": "m_bigearthnet.BigEarthNet",
    "mnz-cattle": "m_nz_cattle.NZCattle",
    "mchesapeake": "m_chesapeake.ChesaPeake",
    "mcashew": "m_cashew_plant.CashewPlant",
}


def get_dataset_class(dataset_name):
    """
    Retrieve dataset class from registry.
    """
    if dataset_name not in DATASET_REGISTRY:
        valid_datasets = sorted(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not implemented. Valid datasets: {valid_datasets}"
        )
    
    module_path, class_name = DATASET_REGISTRY[dataset_name].rsplit(".", 1)
    module = importlib.import_module(f"datasets.{module_path}")
    return getattr(module, class_name)


def get_datasets(cfg):
    """
    Get train, validation, and test dataset splits based on config.
    """
    if not hasattr(cfg, "dataset_name") or not cfg.dataset_name:
        raise AttributeError("cfg must have a valid 'dataset_name' attribute")
    dataset_cls = get_dataset_class(cfg.dataset_name)
    try:
        splits = dataset_cls.get_splits(cfg)
        if not isinstance(splits, (tuple, list)) or len(splits) != 3:
            raise ValueError(
                f"{dataset_cls.__name__}.get_splits must return a tuple of (train, val, test)"
            )
        dataset_train, dataset_val, dataset_test = splits         
        return dataset_train, dataset_val, dataset_test
    
    except Exception as e:
        raise ValueError(f"Failed to get splits for dataset '{cfg.dataset_name}': {str(e)}")


class GeoBenchSubset(Dataset):
    def __init__(self, full_dataset, indices, repeat=1):
        self.full_dataset = full_dataset
        self.indices = indices
        self.repeat = repeat

    def __getitem__(self, idx):
        idx = self.indices[idx // self.repeat]
        return self.full_dataset[idx]

    def __len__(self):
        return len(self.indices) * self.repeat