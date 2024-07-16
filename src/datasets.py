import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

import mne
from sklearn.preprocessing import RobustScaler


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_freq=120) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_freq = resample_freq
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        # データの前処理
        self.preprocess_data()

    # 前処理用関数の追加
    def preprocess_data(self):
        self.X = self.X.numpy()
        for i in range(len(self.X)):
            data = self.X[i]
            data = self.resample(data, self.resample_freq)
            data = self.epoch_and_baseline_correct(data)
            data = self.robust_scale_and_clip(data)
            self.X[i] = data
        self.X = torch.tensor(self.X)

    # リサンプリング
    def resample(self, data, resample_freq): # resample_freq: リサンプリング周波数=120Hz
        original_freq = 1200 # 元のサンプリング周波数
        info = mne.create_info(ch_names=data.shape[0], sfreq=original_freq, ch_types="grad")
        raw = mne.io.RawArray(data, info)
        raw.resample(resample_freq)
        return raw.get_data()

    # ベースライン補正
    def epoch_and_baseline_correct(self, data):
        original_freq = 1200  # 元のサンプリング周波数
        resample_freq = self.resample_freq
        tmin = -0.5  # エポックの開始時間(s)
        tmax = 1.0  # エポックの終了時間(s)
        baseline = (None, 0)  # ベースライン

        info = mne.create_info(ch_names=data.shape[0], sfreq=original_freq, ch_types="grad")
        raw = mne.io.RawArray(data, info)
        
        events = np.array([[int((i+0.5) * original_freq), 0, 1] for i in range(len(data[0]) // original_freq)])  # ダミーイベントによるエポック分割
        epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin, tmax=tmax, baseline=baseline, preload=True) # ベースライン補正
        epochs.resample(resample_freq)
        return epochs.get_data()[0]
    
    # ロバストスケーリングとクリッピング
    def robust_scale_and_clip(self, data):
        scaler = RobustScaler()
        data = scaler.fit_transform(data.T).T
        data = np.clip(data, -20, 20)
        return data

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