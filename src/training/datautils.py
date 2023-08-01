import torch
from torch.utils.data import Dataset


class MyTrainDataset(Dataset):
    def __init__(self, path: str):
        """
        Args:
            path: path to tokenized dataset.
        """
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, ix: int):
        return {key: arr[ix] for key, arr in self.data.items()}
