# Data loader

# Libraries
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class DistributionDataset(IterableDataset):
    """Wrapper around the torch.distributions.Distribution class for Pytorch lighlight dataset

    Stolen from https://github.com/cheind/score-matching/blob/main/diffusion/examples/toy_matching.py
    """

    def __init__(self, dist):
        self.dist = dist

    def __iter__(self):
        while True:
            yield self.dist.sample(), 1


class FiniteDistributionDataset(Dataset):
    """Dataset which using num_samples from a distribution as a dataset"""

    def __init__(self, dist, num_samples, batch_size=None):
        super().__init__()
        self.num_samples = num_samples
        if batch_size:
            self.dataset = torch.concat([
                dist.sample((batch_size,)) for _ in range(int(num_samples / batch_size))
            ])
        else:
            self.dataset = torch.concat([
                dist.sample((num_samples,))
            ])

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset[index], 1


class TransformedTensorDataset(TensorDataset):
    """Wraps a TensorDataset to apply a transform function on the fly."""

    def __init__(self, *tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.transform:
            return self.transform(*data)
        return data

class DistributionDataModule(LightningDataModule):
    """Converts DistributionDataset to LightningDataModule"""

    def __init__(self, dist, **kwargs):
        super().__init__()
        self.dataloader_kwargs = kwargs
        self.dist = dist

    def setup(self, stage=None):
        self.ds = DistributionDataset(self.dist)

    def train_dataloader(self):
        return DataLoader(self.ds, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.ds, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.ds, **self.dataloader_kwargs)


class FiniteDistributionDataModule(LightningDataModule):
    """Converts FiniteDistributionDataset to LightningDataModule"""

    def __init__(self,
                 dist,
                 batched: bool = False,
                 dataset_length: int = 60000,
                 **kwargs
                 ):
        """Constructor

        Args:
            dist (torch.distributions.Distribution): the distribution at stake
            batched (bool): Whether to sample the distribution in multiple small batches stacked.
                This has proven to be important with scikit-learn based distributions
            dataset_length (int): Length of the dataset
            kwargs (dict): Arguments for Dataloader
        """

        super().__init__()
        self.dataloader_kwargs = kwargs
        self.dataset_length = dataset_length
        self.batched = batched
        self.dist = dist

    def setup(self, stage=None):
        self.ds = FiniteDistributionDataset(self.dist, num_samples=self.dataset_length,
                                            batch_size=self.dataloader_kwargs['batch_size'] if self.batched else None)

    def train_dataloader(self):
        return DataLoader(self.ds, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.ds, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.ds, **self.dataloader_kwargs)


class TensorDataModule(LightningDataModule):
    """Transforms TensorDataset to LightningDataModule"""

    def __init__(self, *data, test_size=0.2, shuffle=True, transform=None, **kwargs):
        super().__init__()
        self.dataloader_kwargs = kwargs
        self.test_size = test_size
        self.shuffle = shuffle
        self.transform = transform
        if len(data) == 1:
            data_split = train_test_split(data[0], test_size=self.test_size, shuffle=self.shuffle)
            self.data_train = [data_split[0], torch.empty((data_split[0].shape[0],))]
            self.data_val = [data_split[1], torch.empty((data_split[1].shape[0],))]
        else:
            data_split = train_test_split(
                *data, test_size=self.test_size, shuffle=self.shuffle)
            self.data_train = [data_split[2 * i] for i in range(len(data))]
            self.data_val = [data_split[2 * i + 1] for i in range(len(data))]

    def setup(self, stage=None):
        self.ds_train = TransformedTensorDataset(*self.data_train, transform=self.transform)
        self.ds_val = TransformedTensorDataset(*self.data_val, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.ds_train, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.ds_val, **self.dataloader_kwargs)

    def refresh_dataset(self, transform):
        self.data_train = transform(*self.data_train)
        self.data_val = transform(*self.data_val)
        self.setup()