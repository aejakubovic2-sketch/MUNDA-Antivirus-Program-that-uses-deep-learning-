"""
malconv2_model.py
MalConv2 — end-to-end deep learning malware detector that reads raw bytes.
Based on: "Malware Detection by Eating a Whole EXE" (Raff et al.)
with Global Channel Gating (GCG) attention mechanism.

No feature engineering needed — feed the raw file bytes directly.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CACHE = os.path.join(os.path.dirname(__file__), 'data', 'malconv2')
MODEL_FILE  = 'malconv2_pretrained.pt'
HUGGINGFACE_REPO = 'FutureComputing4AI/MalConvGCT'

# Max file size to feed into the model (2 MB default, increase for larger files)
MAX_FILE_BYTES = 2_000_000
PAD_VALUE      = 256   # out-of-range byte used for padding


# ── Model Architecture ───────────────────────────────────────────────────────

class GlobalChannelGating(nn.Module):
    """
    GCG attention: learns which byte positions and channels matter most.
    Allows the model to focus on malicious regions anywhere in the file.
    """

    def __init__(self, input_channels: int, num_filters: int, window: int):
        super().__init__()
        self.conv_h = nn.Conv1d(
            input_channels, num_filters, window, padding=window // 2, bias=True
        )
        self.conv_g = nn.Conv1d(
            input_channels, num_filters, window, padding=window // 2, bias=True
        )

    def forward(self, x):
        # x: (batch, emb_size, seq_len) after embedding
        H = self.conv_h(x)                  # base features
        G = torch.sigmoid(self.conv_g(x))   # gating weights
        return H * G                         # gated output


class MalConv2(nn.Module):
    """
    MalConv2 with Global Channel Gating.
    Input:  raw byte sequence, shape (batch, seq_len), values 0-256 (256=pad)
    Output: scalar probability per sample (batch,)
    """

    def __init__(
        self,
        num_filters:  int = 128,
        filter_size:  int = 500,
        emb_size:     int = 8,
        max_len:      int = MAX_FILE_BYTES,
    ):
        super().__init__()
        self.max_len   = max_len
        self.emb_size  = emb_size

        # Embedding: 257 values (0-255 bytes + 256 pad) → 8-dim vector
        self.embedding = nn.Embedding(257, emb_size, padding_idx=256)

        # GCG convolutional block
        self.gcg = GlobalChannelGating(emb_size, num_filters, filter_size)

        # Classifier head
        self.fc1 = nn.Linear(num_filters, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) int64 tensor of byte values (0-256)
        returns: (batch,) float tensor of malware probabilities
        """
        # Truncate / pad to max_len
        if x.shape[1] > self.max_len:
            x = x[:, :self.max_len]

        # Embed bytes → (batch, seq_len, emb_size)
        emb = self.embedding(x)

        # Transpose for Conv1d → (batch, emb_size, seq_len)
        emb = emb.transpose(1, 2)

        # GCG attention + convolution → (batch, num_filters, seq_len)
        features = self.gcg(emb)

        # Global max pooling → (batch, num_filters)
        features = features.max(dim=2).values

        # Classification head
        out = F.relu(self.fc1(self.dropout(features)))
        out = torch.sigmoid(self.fc2(out)).squeeze(1)
        return out


# ── Public Class ─────────────────────────────────────────────────────────────

class MalConv2Detector:
    """
    High-level wrapper around MalConv2.
    Handles loading, preprocessing, and prediction.
    """

    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None

    def predict(self, filepath: str) -> float:
        """
        Scan a raw binary file.
        Returns float in [0,1] — probability of being malware.
        """
        model = self._load_model()
        tensor = self._file_to_tensor(filepath)
        with torch.no_grad():
            score = model(tensor).item()
        return float(np.clip(score, 0.0, 1.0))

    def predict_bytes(self, raw_bytes: bytes) -> float:
        """
        Scan raw bytes directly (useful for in-memory scanning).
        """
        model = self._load_model()
        tensor = self._bytes_to_tensor(raw_bytes)
        with torch.no_grad():
            score = model(tensor).item()
        return float(np.clip(score, 0.0, 1.0))

    # ── Internal helpers ─────────────────────────────────

    def _load_model(self) -> MalConv2:
        if self._model is None:
            model_path = os.path.join(MODEL_CACHE, MODEL_FILE)
            if not os.path.isfile(model_path):
                self._download_model(model_path)
            self._model = self._load_weights(model_path)
            self._model.eval()
        return self._model

    def _load_weights(self, path: str) -> MalConv2:
        model = MalConv2()
        state = torch.load(path, map_location=self.device, weights_only=True)
        # Support both raw state_dict and wrapped checkpoint
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        return model.to(self.device)

    def _file_to_tensor(self, filepath: str) -> torch.Tensor:
        with open(filepath, 'rb') as f:
            raw = f.read(MAX_FILE_BYTES)
        return self._bytes_to_tensor(raw)

    def _bytes_to_tensor(self, raw: bytes) -> torch.Tensor:
        arr = np.frombuffer(raw, dtype=np.uint8).copy()
        arr = arr[:MAX_FILE_BYTES]
        # Pad to fixed length for batch consistency
        if len(arr) < MAX_FILE_BYTES:
            pad = np.full(MAX_FILE_BYTES - len(arr), PAD_VALUE, dtype=np.int64)
            arr = np.concatenate([arr.astype(np.int64), pad])
        tensor = torch.tensor(arr, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        return tensor.to(self.device)

    def _download_model(self, dest_path: str):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            print(f"[MalConv2] Downloading pretrained weights from HuggingFace...")
            hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=MODEL_FILE,
                local_dir=MODEL_CACHE,
            )
            print(f"[MalConv2] Downloaded → {dest_path}")
        except Exception as e:
            raise RuntimeError(
                "Could not download the MalConv2 pretrained model.\n"
                "Please install the dependencies, check your network connection, "
                "or place a trained checkpoint at:\n"
                f"  {dest_path}\n"
                f"Original error: {e}"
            ) from e

    def is_available(self) -> bool:
        return os.path.isfile(os.path.join(MODEL_CACHE, MODEL_FILE))

    def ensure_available(self) -> None:
        """Download and load the checkpoint if it is not already cached."""
        self._load_model()

    # ── Training ─────────────────────────────────────────

    def train(self, filepaths: list, labels: list,
              epochs: int = 10, batch_size: int = 32,
              lr: float = 1e-3) -> None:
        """
        Fine-tune MalConv2 on custom data.

        Args:
            filepaths: list of file paths to training samples
            labels:    list of ints (0=benign, 1=malware)
            epochs:    number of training epochs
            batch_size: mini-batch size
            lr:        learning rate
        """
        from torch.utils.data import DataLoader, Dataset

        class BinaryDataset(Dataset):
            def __init__(self, paths, lbls):
                self.paths = paths
                self.lbls  = lbls
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, i):
                with open(self.paths[i], 'rb') as f:
                    raw = f.read(MAX_FILE_BYTES)
                arr = np.frombuffer(raw, dtype=np.uint8).copy().astype(np.int64)
                if len(arr) < MAX_FILE_BYTES:
                    pad = np.full(MAX_FILE_BYTES - len(arr), PAD_VALUE, dtype=np.int64)
                    arr = np.concatenate([arr, pad])
                return torch.tensor(arr), torch.tensor(self.lbls[i], dtype=torch.float32)

        model = self._load_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        loader    = DataLoader(BinaryDataset(filepaths, labels),
                               batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                preds = model(X)
                loss  = criterion(preds, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[MalConv2] Epoch {epoch+1}/{epochs} — "
                  f"loss: {total_loss/len(loader):.4f}")

        save_path = os.path.join(MODEL_CACHE, MODEL_FILE)
        torch.save(model.state_dict(), save_path)
        print(f"[MalConv2] Model saved → {save_path}")
