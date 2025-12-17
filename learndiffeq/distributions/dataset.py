# Sample from a distribution defined by a dataset

import torch


class DatasetDistributionWrapper(torch.nn.Module):
    """Wrap a dataset inside a distribution object

    You can use this with the same API as the original distribution
    """

    def __init__(self, dist, *datasets, temp=1.0):
        """Constructor for DatasetDistributionWrapper

        Args:
                * dist (Distribution): Base distribution
                * datasets (torch.Tensor): Datasets
                * temp (float): Tenperature (default is 1.0)
        """
        super().__init__()
        self.dist = dist
        self.dist_has_log_prob = hasattr(dist, 'log_prob')
        self.n_datasets = len(datasets)
        self.len_dataset = datasets[0].shape[0]
        self.datasets_dtypes = {}
        for i, d in enumerate(datasets):
            self.register_buffer('dataset_' + str(i), d, persistent=False)
            self.datasets_dtypes['dataset_' + str(i)] = d.dtype
        self.register_buffer('temp', torch.tensor(temp), persistent=False)

    def sample(self, sample_shape):
        idx = torch.randint(low=0, high=self.len_dataset, size=sample_shape)
        if self.n_datasets == 1:
            return self.dataset_0[idx]
        else:
            return tuple([getattr(self, 'dataset_' + str(i))[idx] for i in range(self.n_datasets)])

    def log_prob(self, *values):
        if self.dist_has_log_prob:
            return self.dist.log_prob(*values)
        else:
            return -self.dist.U(*values) / self.temp

    def U(self, *values):
        if self.dist_has_log_prob:
            return -self.temp * self.dist.log_prob(*values)
        else:
            return self.dist.U(*values)

    def _apply(self, fn):
        super()._apply(fn)
        for dataset_name, dtype in self.datasets_dtypes.items():
            dataset = getattr(self, dataset_name)
            if dataset is not None and dataset.dtype != dtype:
                setattr(self, dataset_name, dataset.to(dtype=dtype))
        return self