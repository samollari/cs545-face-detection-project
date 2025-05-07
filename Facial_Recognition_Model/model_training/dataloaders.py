import zipfile
import io
import torch
from torch.utils.data import Dataset
from PIL import Image
import threading

class ZipImageDataset(Dataset):
    def __init__(self, zip_path, root_in_zip=None, transform=None):
        self.zip_path = zip_path
        self.root_in_zip = root_in_zip
        self.transform = transform
        self.image_info = []

        with zipfile.ZipFile(self.zip_path, 'r') as archive:
            for file in archive.namelist():
                if file.endswith(('.jpg', '.jpeg', '.png')) and len(file.split('/')) >= 2:
                    parts = file.split('/')
                    if root_in_zip:
                        try:
                            root_index = parts.index(root_in_zip)
                            label = parts[root_index + 1]
                        except (ValueError, IndexError):
                            continue
                    else:
                        label = parts[0]

                    self.image_info.append((file, label))

        self.classes = sorted(set(label for _, label in self.image_info))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.image_info]

        # thread-local zip file
        self._thread_local = threading.local()

    def _get_archive(self):
        if not hasattr(self._thread_local, "archive"):
            self._thread_local.archive = zipfile.ZipFile(self.zip_path, 'r')
        return self._thread_local.archive

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, label = self.samples[index]
        archive = self._get_archive()
        image_data = archive.read(file_name)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class LFWPairDataset(Dataset):
    def __init__(self, pairs, zip_path, transform=None):
        """
        Args:
            pairs (list): List of (path1, path2, label) tuples.
            zip_path (str): Path to the zip file containing images.
            transform (callable, optional): Optional transform to apply to both images.
        """
        self.pairs = pairs
        self.zip_path = zip_path
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]

        # Reopen ZIP file for each __getitem__ call (safe for multiprocessing)
        with zipfile.ZipFile(self.zip_path, 'r') as archive:
            with archive.open(path1) as file1:
                img1 = Image.open(file1).convert('RGB')
            with archive.open(path2) as file2:
                img2 = Image.open(file2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)