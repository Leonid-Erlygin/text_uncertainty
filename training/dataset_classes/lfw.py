from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms


class LFW_aligned_images(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        # self.norm_image = norm_image
        self.img_names = list(Path(dataset_path).rglob("*.jpg"))
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        img = self.img_names[index]
        img = cv2.imread(str(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = (img - 127.5) * 0.0078125
        # if self.norm_image:
        #     img = (
        #         img - np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        #     ) / np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        sample = self.transform(img)
        return sample  # img.astype("float32")

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":
    ds = LFW_aligned_images("/app/datasets/lfw/data_aligned_112_112", norm_image=True)
    t = ds[0]
    x = 1
