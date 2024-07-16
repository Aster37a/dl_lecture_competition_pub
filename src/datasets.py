import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.interpolate import interp1d

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 200, augment: bool = False) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_rate = resample_rate
        self.augment = augment
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # リサンプリング
        self.X = self._resample_data(self.X)
        
        # データ拡張
        if self.augment:
            self.X = self._augment_data(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def _resample_data(self, data: torch.Tensor) -> torch.Tensor:
        current_rate = 200
        target_samples = int(self.resample_rate * (data.shape[2] / current_rate))
        
        resampled_data = np.array([self._resample_single(x.numpy(), target_samples) for x in data])
        
        return torch.tensor(resampled_data).float()

    def _resample_single(self, x, target_samples):
        time_points = np.linspace(0, 1, x.shape[1])
        interp_func = interp1d(time_points, x, kind='linear', axis=1)
        new_time_points = np.linspace(0, 1, target_samples)
        return interp_func(new_time_points)

    def _augment_data(self, data: torch.Tensor) -> torch.Tensor:
        noise_factor = 0.1
        augmented_data = data + noise_factor * torch.randn_like(data)
        return augmented_data






'''
class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
'''
        

'''
# src/datasets.py
import os
import torch
from torch.utils.data import Dataset

class ThingsMEGDataset(Dataset):
    def __init__(self, split, data_dir):
        self.split = split
        self.data_dir = data_dir
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        if split != 'test':
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            self.num_classes = len(torch.unique(self.y))
        else:
            self.y = None
            self.num_classes = 0

        self.seq_len = self.X.shape[1]
        self.num_channels = self.X.shape[2]

        # チャネル数と長さを入れ替える (もし必要であれば)
        self.X = self.X.transpose(1, 2)

        self.mean = self.X.mean()
        self.std = self.X.std()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = (self.X[idx] - self.mean) / self.std  # 正規化
        y = self.y[idx] if self.y is not None else -1
        return X, y, idx

    def get_mean_std(self):
        return self.mean, self.std
'''

'''
# src/datasets.py
import os
import torch
from torch.utils.data import Dataset

class ThingsMEGDataset(Dataset):
    def __init__(self, split, data_dir):
        self.split = split
        self.data_dir = data_dir
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        if split != 'test':
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            self.num_classes = len(torch.unique(self.y))
        else:
            self.y = None
            self.num_classes = 0

        self.seq_len = self.X.shape[2]
        self.num_channels = self.X.shape[1]

        self.mean = self.X.mean()
        self.std = self.X.std()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = (self.X[idx] - self.mean) / self.std  # 正規化
        X = self.augment_data(X)  # データ拡張を適用
        y = self.y[idx] if self.y is not None else -1
        return X, y, idx

    def augment_data(self, X):
        if self.split == 'train':
            # ノイズの追加
            noise = torch.randn_like(X) * 0.01
            X = X + noise
            # デバッグ用の出力
            #print(f"Before shift, X shape: {X.shape}")
            # 時間シフトの追加
            shift = np.random.randint(1, X.shape[1])
            X = torch.roll(X, shifts=shift, dims=1)
            #print(f"After shift, X shape: {X.shape}")
        return X

    def get_mean_std(self):
        return self.mean, self.std
'''


"""
# src/datasets.py
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class ThingsMEGDataset(Dataset):
    def __init__(self, split, data_dir):
        self.split = split
        self.data_dir = data_dir
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        if split != 'test':
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            self.num_classes = len(torch.unique(self.y))
        else:
            self.y = None
            self.num_classes = 0

        self.seq_len = self.X.shape[1]
        self.num_channels = self.X.shape[2]

        self.mean = self.X.mean()
        self.std = self.X.std()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = (self.X[idx] - self.mean) / self.std  # 正規化
        X = self.augment_data(X)  # データ拡張を適用
        y = self.y[idx] if self.y is not None else -1
        return X, y, idx

    def augment_data(self, X):
        if self.split == 'train':
            # ノイズの追加
            noise = torch.randn_like(X) * 0.01
            X = X + noise
            # 時間シフトの追加
            shift = np.random.randint(1, X.shape[1])
            X = torch.roll(X, shifts=shift, dims=0)
        return X

    def get_mean_std(self):
        return self.mean, self.std
"""

'''
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class ThingsMEGDataset(Dataset):
    def __init__(self, split, data_dir):
        self.split = split
        self.data_dir = data_dir
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        if split != 'test':
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            self.num_classes = len(torch.unique(self.y))
        else:
            self.y = None
            self.num_classes = 0

        self.seq_len = self.X.shape[1]
        self.num_channels = self.X.shape[2]

        self.mean = self.X.mean()
        self.std = self.X.std()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = (self.X[idx] - self.mean) / self.std  # 正規化
        X = self.augment_data(X)  # データ拡張を適用
        y = self.y[idx] if self.y is not None else -1
        return X, y, idx

    def augment_data(self, X):
        if self.split == 'train':
            # ノイズの追加
            noise = torch.randn_like(X) * 0.01
            X = X + noise
            # 時間シフトの追加
            shift = np.random.randint(1, X.shape[1])
            X = torch.roll(X, shifts=shift, dims=0)
        return X

    def get_mean_std(self):
        return self.mean, self.std
'''

'''
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        self.mean = self.X.mean()
        self.std = self.X.std()
        self.X = (self.X - self.mean) / self.std  # 正規化

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        y = self.y[i] if hasattr(self, "y") else -1
        return X, y, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
'''

'''
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        self.mean = self.X.mean()
        self.std = self.X.std()
        self.X = (self.X - self.mean) / self.std  # 正規化

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.split == "train":
            X = self.augment_data(X)
        y = self.y[i] if hasattr(self, "y") else -1
        return X, y, self.subject_idxs[i]
    
    def augment_data(self, X):
        noise = torch.randn_like(X) * 0.01
        X = X + noise
        return X
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
'''

'''
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        self.mean = self.X.mean()
        self.std = self.X.std()
        self.X = (self.X - self.mean) / self.std  # 正規化

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.split == "train":
            X = self.augment_data(X)
        if self.split in ["train", "val"]:
            y = self.y[i]
            return X, y, self.subject_idxs[i]
        return X, self.subject_idxs[i]
    
    def augment_data(self, X):
        # ノイズの追加
        noise = torch.randn_like(X) * 0.01
        X = X + noise
        # 時間軸方向のシフト
        shift = np.random.randint(X.size(1))
        X = torch.roll(X, shift, dims=1)
        return X
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
'''

'''
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        self.mean = self.X.mean()
        self.std = self.X.std()
        self.X = (self.X - self.mean) / self.std  # 正規化

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.split == "train":
            X = self.augment_data(X)
        if self.split in ["train", "val"]:
            y = self.y[i]
            return X, y, self.subject_idxs[i]
        return X, self.subject_idxs[i]
    
    def augment_data(self, X):
        # ノイズの追加
        noise = torch.randn_like(X) * 0.01
        X = X + noise
        # 時間軸方向のシフト
        shift = np.random.randint(X.size(1))
        X = torch.roll(X, shift, dims=1)
        return X
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
'''

'''
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import torchvision.transforms as transforms

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", normalize: bool = False) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        if normalize:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((271, 281)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # mean and std for normalization
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((271, 281)),
                transforms.ToTensor()
            ])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        X = self.transform(X)
        
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
'''

"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.transform:
            X = self.transform(X)
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
"""

"""
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.transform:
            X = self.transform(X)
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
"""


"""
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.transform:
            X = self.transform(X)
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
"""

'''
import os
import torch

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.transform:
            X = self.transform(X)
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
'''


'''
import os
import torch
from torch.utils.data import Dataset  # この行を追加

class ThingsMEGDataset(Dataset):
    def __init__(self, split, data_dir, transform=None, has_labels=True):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self.has_labels = has_labels
        self.data = self.load_data()
        if self.has_labels:
            self.labels = self.load_labels()
            self.num_classes = len(torch.unique(self.labels))  # クラス数を計算
        else:
            self.num_classes = None

    def load_data(self):
        data_path = os.path.join(self.data_dir, f"{self.split}_X.pt")
        return torch.load(data_path)

    def load_labels(self):
        labels_path = os.path.join(self.data_dir, f"{self.split}_y.pt")
        return torch.load(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        if self.has_labels:
            y = self.labels[idx]
            return x, y
        else:
            return x
'''


'''これは動く
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.transform:
            X = self.transform(X)
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
'''

'''
import os
import torch
from torch.utils.data import Dataset

class ThingsMEGDataset(Dataset):
    def __init__(self, split, data_dir):
        self.data_dir = data_dir
        self.split = split
        self.data = self.load_data()
        self.labels = self.load_labels()
        self.num_classes = 1854  # ラベルの範囲に基づいて設定

    def load_data(self):
        data_path = os.path.join(self.data_dir, f"{self.split}_X.pt")
        return torch.load(data_path)

    def load_labels(self):
        labels_path = os.path.join(self.data_dir, f"{self.split}_y.pt")
        if os.path.exists(labels_path):
            return torch.load(labels_path)
        else:
            # ラベルファイルが存在しない場合はNoneを返す
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return X, y, idx  # データ、ラベル、インデックスを返す
        else:
            return X, idx  # データとインデックスを返す

    def verify_labels(self):
        if self.labels is not None:
            print(f"Labels range from {self.labels.min()} to {self.labels.max()}")
            assert all(0 <= label < self.num_classes for label in self.labels), "Invalid label values"
'''

'''
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from scipy.signal import resample, butter, filtfilt

class ThingsMEGDataset(Dataset):
    def __init__(self, split, data_dir):
        self.split = split
        self.data_dir = data_dir
        self.data = self.load_data()
        self.labels = self.load_labels()
        self.subjects = self.load_subjects()  # 被験者情報の読み込み
        self.num_classes = len(np.unique(self.labels))
        
        # フィルタリング設定
        self.fs = 200  # サンプリングレート
        self.lowcut = 0.1  # Low cut-off frequency
        self.highcut = 50  # High cut-off frequency
        self.order = 5  # フィルタのオーダー

    def load_data(self):
        data_path = os.path.join(self.data_dir, f"{self.split}_X.pt")
        return torch.load(data_path)
    
    def load_labels(self):
        labels_path = os.path.join(self.data_dir, f"{self.split}_y.pt")
        return torch.load(labels_path)
    
    def load_subjects(self):
        subjects_path = os.path.join(self.data_dir, f"{self.split}_subjects.pt")
        return torch.load(subjects_path)
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def preprocess(self, x):
        # リサンプリング
        x = resample(x, num=200, axis=-1)  # サンプリングレート200Hzにリサンプリング
        
        # フィルタリング
        x = self.butter_bandpass_filter(x, self.lowcut, self.highcut, self.fs, self.order)
        
        # スケーリング
        x = (x - np.mean(x)) / np.std(x)
        
        # ベースライン補正
        x = x - np.mean(x[:, :20], axis=1, keepdims=True)
        
        return x
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        subject = self.subjects[idx]

        # 前処理を適用
        x = self.preprocess(x)
        
        return torch.tensor(x, dtype=torch.float32), y, subject
'''


'''リサンプリングの実装
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.interpolate import interp1d

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 200) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_rate = resample_rate
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()  # データ型をfloatに変換
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # リサンプリング
        self.X = self._resample_data(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def _resample_data(self, data: torch.Tensor) -> torch.Tensor:
        current_rate = 200
        target_samples = int(self.resample_rate * (data.shape[2] / current_rate))
        
        resampled_data = []
        for x in data:
            resampled_data.append(self._resample_single(x.numpy(), target_samples))
        
        return torch.tensor(np.array(resampled_data)).float()  # データ型をfloatに変換

    def _resample_single(self, x, target_samples):
        time_points = np.linspace(0, 1, x.shape[1])
        interp_func = interp1d(time_points, x, kind='linear', axis=1)
        new_time_points = np.linspace(0, 1, target_samples)
        return interp_func(new_time_points)
'''

"""フィルタリングの実装(微妙)
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 200, lowcut: float = 0.5, highcut: float = 40.0) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_rate = resample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()  # データ型をfloatに変換
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # リサンプリング
        self.X = self._resample_data(self.X)
        
        # フィルタリング
        self.X = self._filter_data(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def _resample_data(self, data: torch.Tensor) -> torch.Tensor:
        current_rate = 200
        target_samples = int(self.resample_rate * (data.shape[2] / current_rate))
        
        resampled_data = []
        for x in data:
            resampled_data.append(self._resample_single(x.numpy(), target_samples))
        
        return torch.tensor(np.array(resampled_data)).float()  # データ型をfloatに変換

    def _resample_single(self, x, target_samples):
        time_points = np.linspace(0, 1, x.shape[1])
        interp_func = interp1d(time_points, x, kind='linear', axis=1)
        new_time_points = np.linspace(0, 1, target_samples)
        return interp_func(new_time_points)

    def _filter_data(self, data: torch.Tensor) -> torch.Tensor:
        nyquist = 0.5 * self.resample_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(2, [low, high], btype='band')  # フィルタの順序を2に変更

        filtered_data = []
        for x in data:
            filtered_data.append(filtfilt(b, a, x, axis=1))
        
        return torch.tensor(np.array(filtered_data)).float()

"""

'''
#リサンプリング、ベースライン補正、スケーリングを実装
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.interpolate import interp1d

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 200, scale: bool = True) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_rate = resample_rate
        self.scale = scale
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # リサンプリング
        self.X = self._resample_data(self.X)
        
        # ベースライン補正
        self.X = self._baseline_correct(self.X)
        
        # スケーリング
        if self.scale:
            self.X = self._scale_data(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def _resample_data(self, data: torch.Tensor) -> torch.Tensor:
        current_rate = 200
        target_samples = int(self.resample_rate * (data.shape[2] / current_rate))
        
        resampled_data = np.array([self._resample_single(x.numpy(), target_samples) for x in data])
        
        return torch.tensor(resampled_data).float()

    def _resample_single(self, x, target_samples):
        time_points = np.linspace(0, 1, x.shape[1])
        interp_func = interp1d(time_points, x, kind='linear', axis=1)
        new_time_points = np.linspace(0, 1, target_samples)
        return interp_func(new_time_points)

    def _baseline_correct(self, data: torch.Tensor) -> torch.Tensor:
        baseline_data = []
        for x in data:
            baseline = x[:, :int(0.2 * x.shape[1])].mean(axis=1, keepdims=True)
            baseline_data.append(x - baseline)
        
        return torch.stack(baseline_data)

    def _scale_data(self, data: torch.Tensor) -> torch.Tensor:
        scaled_data = []
        for x in data:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            scaled_data.append((x - mean) / std)
        
        return torch.stack(scaled_data)
'''

'''
#基準をクリアしたコード
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.interpolate import interp1d

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 200, scale: bool = True, augment: bool = False) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_rate = resample_rate
        self.scale = scale
        self.augment = augment
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # リサンプリング
        self.X = self._resample_data(self.X)
        
        # ベースライン補正
        self.X = self._baseline_correct(self.X)
        
        # スケーリング
        if self.scale:
            self.X = self._scale_data(self.X)
        
        # データ拡張
        if self.augment and split == "train":
            self.X = self._augment_data(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def _resample_data(self, data: torch.Tensor) -> torch.Tensor:
        current_rate = 200
        target_samples = int(self.resample_rate * (data.shape[2] / current_rate))
        
        resampled_data = np.array([self._resample_single(x.numpy(), target_samples) for x in data])
        
        return torch.tensor(resampled_data).float()

    def _resample_single(self, x, target_samples):
        time_points = np.linspace(0, 1, x.shape[1])
        interp_func = interp1d(time_points, x, kind='linear', axis=1)
        new_time_points = np.linspace(0, 1, target_samples)
        return interp_func(new_time_points)

    def _baseline_correct(self, data: torch.Tensor) -> torch.Tensor:
        baseline_data = []
        for x in data:
            baseline = x[:, :int(0.2 * x.shape[1])].mean(axis=1, keepdims=True)
            baseline_data.append(x - baseline)
        
        return torch.stack(baseline_data)

    def _scale_data(self, data: torch.Tensor) -> torch.Tensor:
        scaled_data = []
        for x in data:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            scaled_data.append((x - mean) / std)
        
        return torch.stack(scaled_data)

    def _augment_data(self, data: torch.Tensor) -> torch.Tensor:
        augmented_data = []
        for x in data:
            noise = torch.randn_like(x) * 0.01  # ノイズの追加
            augmented_data.append(x + noise)
        
        return torch.stack(augmented_data)
'''

'''
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.interpolate import interp1d

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 200, augment: bool = False) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_rate = resample_rate
        self.augment = augment
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # リサンプリング
        self.X = self._resample_data(self.X)
        
        # データ拡張
        if self.augment:
            self.X = self._augment_data(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def _resample_data(self, data: torch.Tensor) -> torch.Tensor:
        current_rate = 200
        target_samples = int(self.resample_rate * (data.shape[2] / current_rate))
        
        resampled_data = np.array([self._resample_single(x.numpy(), target_samples) for x in data])
        
        return torch.tensor(resampled_data).float()

    def _resample_single(self, x, target_samples):
        time_points = np.linspace(0, 1, x.shape[1])
        interp_func = interp1d(time_points, x, kind='linear', axis=1)
        new_time_points = np.linspace(0, 1, target_samples)
        return interp_func(new_time_points)

    def _augment_data(self, data: torch.Tensor) -> torch.Tensor:
        noise_factor = 0.1
        augmented_data = data + noise_factor * torch.randn_like(data)
        return augmented_data
'''

'''
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.interpolate import interp1d

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 200, scale: bool = True, augment: bool = False) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_rate = resample_rate
        self.scale = scale
        self.augment = augment
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).float()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # リサンプリング
        self.X = self._resample_data(self.X)
        
        # ベースライン補正
        self.X = self._baseline_correct(self.X)
        
        # スケーリング
        if self.scale:
            self.X = self._scale_data(self.X)
        
        # データ拡張
        if self.augment and split == "train":
            self.X = self._augment_data(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def _resample_data(self, data: torch.Tensor) -> torch.Tensor:
        current_rate = 200
        target_samples = int(self.resample_rate * (data.shape[2] / current_rate))
        
        resampled_data = np.array([self._resample_single(x.numpy(), target_samples) for x in data])
        
        return torch.tensor(resampled_data).float()

    def _resample_single(self, x, target_samples):
        time_points = np.linspace(0, 1, x.shape[1])
        interp_func = interp1d(time_points, x, kind='linear', axis=1)
        new_time_points = np.linspace(0, 1, target_samples)
        return interp_func(new_time_points)

    def _baseline_correct(self, data: torch.Tensor) -> torch.Tensor:
        baseline_data = []
        for x in data:
            baseline = x[:, :int(0.2 * x.shape[1])].mean(axis=1, keepdims=True)
            baseline_data.append(x - baseline)
        
        return torch.stack(baseline_data)

    def _scale_data(self, data: torch.Tensor) -> torch.Tensor:
        scaled_data = []
        for x in data:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            scaled_data.append((x - mean) / std)
        
        return torch.stack(scaled_data)

    def _augment_data(self, data: torch.Tensor) -> torch.Tensor:
        augmented_data = []
        for x in data:
            noise = torch.randn_like(x) * 0.01  # ノイズの追加
            augmented_data.append(x + noise)
        
        return torch.stack(augmented_data)
'''