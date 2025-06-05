from torch.utils.data import Dataset
import lightning as L

class MO810Dataset(Dataset):
    """
    Abstract base class for dataset in MO810 course works.
    
    This class defines the interface that all MO810-compatible datasets should implement,
    including methods for downloading, accessing, and representing samples.
    """

    def download_data(self):
        """
        Downloads the dataset. Must be implemented in subclasses.
        """
        raise NotImplemented

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset. Must be implemented in subclasses.
        """
        raise NotImplemented
    
    def __getitem__(self, index):
        """
        Fetches the sample at the given index. Must be implemented in subclasses.

        Args:
            index (int): Index of the sample.

        Returns:
            Sample from the dataset.
        """
        raise NotImplemented
    
    def __str__(self):
        """
        Returns:
            str: Human-readable description of the dataset.
        """
        raise NotImplemented

class MO810DataModule(L.LightningDataModule):
    """
    Abstract base class for PyTorch Lightning DataModules in the MO810 course work.
    
    Provides the expected interface for training, validation, and test dataloaders.
    """

    def train_dataloader(self, fraction=None, samples_per_class=None):
        """
        Returns:
            DataLoader: Training dataloader. Should support sampling options.
        """
        raise NotImplemented

    def val_dataloader(self):
        """
        Returns:
            DataLoader: Validation dataloader.
        """
        raise NotImplemented

    def test_dataloader(self):
        """
        Returns:
            DataLoader: Test dataloader.
        """
        raise NotImplemented