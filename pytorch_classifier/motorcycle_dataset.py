import os.path as osp
import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from random import randint, shuffle
import pickle


class MotorcycleDataset(Dataset):
    """ Custom dataset
        Args:
            root_dir (string): Path to the directory with data.
            data_type (string): train/test/val
            transform (callable, optional): Optional transform to be applied on a sample.
        """

    def __init__(self, root_dir, data_type, transform=None):
        # Get paths to images and annotations
        # We don't load in the actual data right now to save memory
        all_files = glob.glob(osp.join(root_dir, "*.pkl"))

        assert data_type in ("train", "test", "val"), "Invalid data type"
        if data_type == "train":
            self.files = all_files[:int(0.8 * len(all_files))]
        elif data_type == "test":
            self.files = all_files[int(
                0.8 * len(all_files)): int(0.9 * len(all_files))]
        elif data_type == "val":
            self.files = all_files[int(0.9 * len(all_files)):]

        self.classes = ("ON", "OFF")

        # Save the given transform
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the image
        pkl_file = self.files[idx]

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            im = data["image"]
            label = data["label"]

        # Return the sample, potentially transformed
        sample = {"image": im, "label": label}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_class_names(self):
        return self.classes


class Flip(object):
    """Randomly horizontally flips the image in the sample.
        Args:
            probability (float): Probability of this transform being applied, on a scale of 0 - 1.
    """

    def __init__(self, probability=0.5):
        self.probability = probability * 100

    def __call__(self, sample):
        im = sample["image"]

        if randint(0, 100) <= self.probability:
            im = cv2.flip(im, 1)

        return {"image": im, "label": sample["label"]}


class RandomCrop(object):
    """Randomly crops the image in the sample.
        Args:
            output_size (tuple or int): Desired output size. If int, square crop is applied.
            probability (float): Probability of the crop being off-center, on a scale from 0 - 1.
    """

    def __init__(self, output_size, probability=0.5):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            assert len(output_size) == 2
            self.output_size = output_size

        self.probability = probability * 100

    def __call__(self, sample):
        im = sample["image"]
        w, h = im.shape[:2]
        new_w, new_h = self.output_size

        if randint(0, 100) <= self.probability:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
        else:
            top = h - new_h
            left = w - new_w

        im = im[left: left + new_w, top: top + new_h]

        return {"image": im, "label": sample["label"]}


class Resize(object):
    """Resizes the image in the sample.
        Args:
            output_size (tuple or int): Desired output size. If int, square aspect ratio is applied.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            assert len(output_size == 2)
            self.output_size = output_size

    def __call__(self, sample):
        im = sample["image"]
        im = cv2.resize(im, self.output_size)
        return {"image": im, "label": sample["label"]}


class Normalize(object):
    """Normalizes the pixel values of the sample image."""

    def __call__(self, sample):
        im = sample["image"]
        w, h = im.shape[:2]
        norm_img = np.zeros((w, h))

        result = cv2.normalize(im, norm_img, 0, 255, cv2.NORM_MINMAX)
        result = result / 255.0

        return {"image": result,
                "label": sample["label"]}


class ToTensor(object):
    """Converts ndarrays in sample to Tensor"""

    def __call__(self, sample):
        im = sample["image"]

        # cv2 images are width x height x color
        # torch images are color x height x width
        im = np.transpose(im, (2, 1, 0))
        return {"image": torch.from_numpy(im).float(),
                "label": sample["label"]}


def show_label_batch(classes, sample_batched):
    """Show image with labels for a batch of samples.
        Args:
            classes (dict): Maps class indices to names.
            sample_batched: Batch from dataloader.
    """
    num_samples = int(sample_batched["image"].size()[0])

    for i in range(num_samples):
        im = sample_batched["image"][i].numpy()
        im = np.transpose(im, (2, 1, 0))
        cv2.imshow("", im)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == ord("q"):
            print()
            exit()


TrainPipeline = transforms.Compose([
    Flip(), RandomCrop(380), Resize(128), Normalize(), ToTensor()
])

TestPipeline = transforms.Compose([
    Resize(128), Normalize(), ToTensor()
])


if __name__ == "__main__":
    root_dir = "C:\\Users\\jewik\\GitRepos\\pytorch_classifier\\data"
    motorcycle_dataset = MotorcycleDataset(
        root_dir, "train", transform=TrainPipeline)
    classes = motorcycle_dataset.get_class_names()
    # num_batches = 3

    dataloader = DataLoader(
        motorcycle_dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print()
        print(f"Batch #{i_batch + 1} shape:", sample_batched["image"].size())
        print(f"Labels: {sample_batched['label']}")
        # show_label_batch(classes, sample_batched)

        # if i_batch == num_batches:
        #     break

    print()
