from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import sys
import scipy
from pathlib import Path
from glob import glob

sys.path.append("/app")


class SurvFace(Dataset):
    def __init__(self, data_dir: str) -> None:
        super().__init__()
        # first gallery images are loaded
        # then probe images
        # then we load randomly sampled imposters
        self.image_paths = []
        mat_gallery = scipy.io.loadmat(str(Path(data_dir) / "gallery_img_ID_pairs.mat"))
        mat_probe = scipy.io.loadmat(Path(data_dir) / "mated_probe_img_ID_pairs.mat")

        self.gallery_ids = np.squeeze(mat_gallery["gallery_ids"])
        gallery_set = [
            image_name[0] for image_name in np.squeeze(mat_gallery["gallery_set"])
        ]
        self.gallery_template_ids = self.gallery_ids
        shift = 1 + np.max(self.gallery_ids)

        self.mated_probe_ids = np.squeeze(mat_probe["mated_probe_ids"])
        mated_probe_set = [
            image_name[0] for image_name in np.squeeze(mat_probe["mated_probe_set"])
        ]
        self.mated_probe_template_ids = self.mated_probe_ids + shift

        unmated_probe_set = np.load(Path(data_dir) / "unmated_probe_subset.npy")

        self.unmated_probe_ids = shift + np.arange(
            unmated_probe_set.shape[0]
        )  # generate fake ids that do not equal to gallery ids
        self.unmated_probe_template_ids = (
            self.unmated_probe_ids + 1 + np.max(self.mated_probe_template_ids)
        )

        self.image_short_names = []
        for image_name in gallery_set:
            self.image_paths.append(Path(data_dir) / "gallery" / image_name)
            self.image_short_names.append("gallery/" + image_name)
        for image_name in mated_probe_set:
            self.image_paths.append(Path(data_dir) / "mated_probe" / image_name)
            self.image_short_names.append("mated_probe/" + image_name)
        for image_name in unmated_probe_set:
            self.image_paths.append(Path(data_dir) / "unmated_probe" / image_name)
            self.image_short_names.append("unmated_probe/" + image_name)

    def __getitem__(self, index):
        image_path = str(self.image_paths[index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
        image = (image - 127.5) * 0.0078125
        return image.astype("float32")

    def __len__(self):
        return len(self.image_paths)
