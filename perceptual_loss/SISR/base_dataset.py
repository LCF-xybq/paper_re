from torch.utils.data import Dataset
from abc import abstractmethod
import copy

class BaseDataset(Dataset):
    def __init__(self, pipeline, test_mode=False):
        super(BaseDataset, self).__init__()
        self.test_mode = test_mode
        self.pipeline = pipeline

    @abstractmethod
    def load_annotations(self):
        pass

    def prepare_train_data(self, idx):
        """
            idx (int): Index of the training batch data.
        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)

        return self.prepare_train_data(idx)