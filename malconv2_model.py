"""
malconv2_model.py
MalConv2 / MalConvGCT raw-byte malware detector.

This wrapper loads the public checkpoint from the official MalConv2 repository:
https://github.com/FutureComputing4AI/MalConv2
"""

import os
import sys
import json
import argparse
from urllib.request import urlretrieve

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise

    TORCH_IMPORT_ERROR = exc

    class _MissingTorch:
        Tensor = object

        class cuda:
            @staticmethod
            def is_available():
                return False

    class _MissingNN:
        Module = object

    torch = _MissingTorch()
    nn = _MissingNN()
    F = None

MODEL_CACHE = os.path.join(os.path.dirname(__file__), 'data', 'malconv2')
MODEL_FILE = 'malconvGCT_nocat.checkpoint'
LEGACY_MODEL_FILE = 'malconv2_pretrained.pt'
MODEL_URL = (
    'https://github.com/FutureComputing4AI/MalConv2/raw/main/'
    'malconvGCT_nocat.checkpoint'
)

MAX_FILE_BYTES = int(os.environ.get('MUNDA_MALCONV2_MAX_BYTES', 2_000_000))
PAD_VALUE = 0


def _require_torch():
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MalConv2 requires PyTorch, but it is not installed. "
            "Install project dependencies with `python -m pip install -r "
            "requirements.txt` and then run `python main.py --download-models`."
        ) from TORCH_IMPORT_ERROR


# ── Official MalConv2 Architecture ──────────────────────────────────────────

class LowMemConvBase(nn.Module):
    """
    Inference version of the fixed-memory pooling used by the official
    MalConv2/MalConvGCT implementation.
    """

    def __init__(self, chunk_size=65536, min_chunk_size=1024):
        super().__init__()
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.receptive_field = None

    def process_range(self, x, **kwargs):
        raise NotImplementedError

    def determine_rf(self):
        if self.receptive_field is not None:
            return self.receptive_field, self.stride, self.out_channels

        cur_device = next(self.embd.parameters()).device
        min_rf = 1
        max_rf = self.chunk_size

        with torch.no_grad():
            tmp = torch.zeros((1, max_rf), dtype=torch.long, device=cur_device)
            while True:
                test_size = (min_rf + max_rf) // 2
                try:
                    self.process_range(tmp[:, 0:test_size])
                    is_valid = True
                except Exception:
                    is_valid = False

                if is_valid:
                    max_rf = test_size
                else:
                    min_rf = test_size + 1

                if max_rf == min_rf:
                    self.receptive_field = min_rf
                    out_shape = self.process_range(tmp).shape
                    self.stride = self.chunk_size // out_shape[2]
                    self.out_channels = out_shape[1]
                    break

        return self.receptive_field, self.stride, self.out_channels

    def seq2fix(self, x, pr_args=None):
        pr_args = pr_args or {}
        receptive_window, stride, out_channels = self.determine_rf()

        if x.shape[1] < receptive_window:
            x = F.pad(x, (0, receptive_window - x.shape[1]), value=PAD_VALUE)

        batch_size = x.shape[0]
        length = x.shape[1]
        winner_values = np.zeros((batch_size, out_channels)) - 1.0
        winner_indices = np.zeros((batch_size, out_channels), dtype=np.int64)
        cur_device = next(self.embd.parameters()).device

        start = 0
        end = min(self.chunk_size, length)
        with torch.no_grad():
            while start < end and (end - start) >= max(self.min_chunk_size, receptive_window):
                x_sub = x[:, start:end].to(cur_device)
                activs = self.process_range(x_sub.long(), **pr_args)
                activ_win, activ_idx = F.max_pool1d(
                    activs,
                    kernel_size=activs.shape[2],
                    return_indices=True,
                )
                activ_win = activ_win.cpu().numpy()[:, :, 0]
                activ_idx = activ_idx.cpu().numpy()[:, :, 0]
                selected = winner_values < activ_win
                winner_indices[selected] = activ_idx[selected] * stride + start
                winner_values[selected] = activ_win[selected]
                start = end
                end = min(start + self.chunk_size, length)

        final_indices = [np.unique(winner_indices[b, :]) for b in range(batch_size)]
        chunks = [
            [
                x[b:b + 1, max(i - receptive_window, 0):min(i + receptive_window, length)]
                for i in final_indices[b]
            ]
            for b in range(batch_size)
        ]
        chunks = [torch.cat(c, dim=1)[0, :] for c in chunks]
        x_selected = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)
        x_selected = x_selected.to(cur_device)
        x_selected = self.process_range(x_selected.long(), **pr_args)
        x_selected = self.pooling(x_selected)
        return x_selected.view(x_selected.size(0), -1)


class MalConvML(LowMemConvBase):
    def __init__(
        self,
        out_size=2,
        channels=128,
        window_size=512,
        stride=512,
        layers=1,
        embd_size=8,
    ):
        super().__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=PAD_VALUE)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embd_size, channels * 2, window_size, stride=stride, bias=True)] +
            [
                nn.Conv1d(channels, channels * 2, window_size, stride=1, bias=True)
                for _ in range(layers - 1)
            ]
        )
        self.convs_1 = nn.ModuleList(
            [nn.Conv1d(channels, channels, 1, bias=True) for _ in range(layers)]
        )
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)

    def process_range(self, x):
        x = self.embd(x)
        x = x.permute(0, 2, 1).contiguous()
        for conv_glu, conv_share in zip(self.convs, self.convs_1):
            x = F.leaky_relu(conv_share(F.glu(conv_glu(x.contiguous()), dim=1)))
        return x

    def forward(self, x):
        post_conv = self.seq2fix(x)
        penult = F.relu(self.fc_1(post_conv))
        return self.fc_2(penult), penult, post_conv


class MalConvGCT(LowMemConvBase):
    def __init__(
        self,
        out_size=2,
        channels=256,
        window_size=256,
        stride=64,
        layers=1,
        embd_size=8,
        low_mem=True,
    ):
        super().__init__()
        self.low_mem = low_mem
        self.embd = nn.Embedding(257, embd_size, padding_idx=PAD_VALUE)
        self.context_net = MalConvML(
            out_size=channels,
            channels=channels,
            window_size=window_size,
            stride=stride,
            layers=layers,
            embd_size=embd_size,
        )
        self.convs = nn.ModuleList(
            [nn.Conv1d(embd_size, channels * 2, window_size, stride=stride, bias=True)] +
            [
                nn.Conv1d(channels, channels * 2, window_size, stride=1, bias=True)
                for _ in range(layers - 1)
            ]
        )
        self.linear_atn = nn.ModuleList(
            [nn.Linear(channels, channels) for _ in range(layers)]
        )
        self.convs_share = nn.ModuleList(
            [nn.Conv1d(channels, channels, 1, bias=True) for _ in range(layers)]
        )
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)

    def determine_rf(self):
        return self.context_net.determine_rf()

    def process_range(self, x, gct=None):
        if gct is None:
            raise RuntimeError("No global context provided")

        x = self.embd(x)
        x = x.permute(0, 2, 1)

        for conv_glu, linear_cntx, conv_share in zip(
            self.convs,
            self.linear_atn,
            self.convs_share,
        ):
            x = F.glu(conv_glu(x), dim=1)
            x = F.leaky_relu(conv_share(x))
            batch_size = x.shape[0]
            channels = x.shape[1]

            context = torch.tanh(linear_cntx(gct)).unsqueeze(dim=2)
            x_tmp = x.view(1, batch_size * channels, -1)
            x_tmp = F.conv1d(x_tmp, context, groups=batch_size)
            gates = torch.sigmoid(x_tmp.view(batch_size, 1, -1))
            x = x * gates

        return x

    def forward(self, x):
        global_context = self.context_net.seq2fix(x)
        post_conv = self.seq2fix(x, pr_args={'gct': global_context})
        penult = F.leaky_relu(self.fc_1(post_conv))
        return self.fc_2(penult), penult, post_conv


# ── Public Detector ─────────────────────────────────────────────────────────

class MalConv2Detector:
    """
    High-level wrapper around the official MalConv2 checkpoint.
    """

    def __init__(self, device: str = None, auto_download: bool = True):
        self.device = self._resolve_device(device)
        self.auto_download = auto_download
        self._model = None

    def predict(self, filepath: str) -> float:
        model = self._load_model()
        tensor = self._file_to_tensor(filepath)
        with torch.no_grad():
            logits, _penult, _post_conv = model(tensor)
            score = F.softmax(logits, dim=-1)[0, 1].item()
        return float(np.clip(score, 0.0, 1.0))

    def predict_bytes(self, raw_bytes: bytes) -> float:
        model = self._load_model()
        tensor = self._bytes_to_tensor(raw_bytes)
        with torch.no_grad():
            logits, _penult, _post_conv = model(tensor)
            score = F.softmax(logits, dim=-1)[0, 1].item()
        return float(np.clip(score, 0.0, 1.0))

    def _load_model(self) -> MalConvGCT:
        _require_torch()
        if self._model is None:
            model_path = self._model_path()
            if not os.path.isfile(model_path):
                if not self.auto_download:
                    raise FileNotFoundError(
                        "MalConv2 checkpoint is not downloaded yet. Run "
                        "`python main.py --download-models` or place "
                        f"{MODEL_FILE} at {model_path}."
                    )
                self._download_model(model_path)
            self._model = self._load_weights(model_path)
            self._model.eval()
        return self._model

    def _load_weights(self, path: str) -> MalConvGCT:
        checkpoint = self._safe_torch_load(path)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                "MalConv2 checkpoint has an unsupported format. Expected a "
                "state_dict or a dict containing `model_state_dict`."
            )

        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            model = MalConvGCT(
                channels=checkpoint.get('channels', 256),
                window_size=checkpoint.get('filter_size', 256),
                stride=checkpoint.get('stride', 64),
                embd_size=checkpoint.get('embd_dim', 8),
            )
        elif 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
            model = MalConvGCT()
        else:
            state = checkpoint
            model = MalConvGCT()

        if not isinstance(state, dict):
            raise RuntimeError(
                "MalConv2 checkpoint state is invalid. Expected a PyTorch "
                "state_dict mapping parameter names to tensors."
            )

        incompatible = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, 'missing_keys', incompatible[0]))
        unexpected = list(getattr(incompatible, 'unexpected_keys', incompatible[1]))
        if missing:
            raise RuntimeError(
                "MalConv2 checkpoint is missing required weights: "
                f"{', '.join(missing)}"
            )
        if unexpected:
            print(
                "[MalConv2] Ignoring unused checkpoint weights: "
                f"{', '.join(unexpected)}",
                file=sys.stderr,
            )
        return model.to(self.device)

    def _safe_torch_load(self, path: str):
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)
        except Exception as e:
            # Some older public checkpoints include metadata that PyTorch's
            # weights-only unpickler may reject. The checkpoint path is explicit
            # and user-controlled, so fall back to normal loading for that case.
            if 'weights_only' in str(e) or 'Weights only load failed' in str(e):
                return torch.load(path, map_location=self.device, weights_only=False)
            raise

    def _file_to_tensor(self, filepath: str) -> torch.Tensor:
        with open(filepath, 'rb') as f:
            raw = f.read(MAX_FILE_BYTES)
        return self._bytes_to_tensor(raw)

    def _bytes_to_tensor(self, raw: bytes) -> torch.Tensor:
        # Official loader maps byte 0..255 to token 1..256 and reserves 0 for pad.
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1
        tensor = torch.tensor(arr[:MAX_FILE_BYTES], dtype=torch.long).unsqueeze(0)
        return tensor.to(self.device)

    def _download_model(self, dest_path: str):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            print("[MalConv2] Downloading official checkpoint from GitHub...")
            urlretrieve(MODEL_URL, dest_path)
            if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
                raise RuntimeError("downloaded checkpoint is empty")
            print(f"[MalConv2] Downloaded -> {dest_path}")
        except Exception as e:
            try:
                if os.path.isfile(dest_path):
                    os.remove(dest_path)
            except OSError:
                pass
            raise RuntimeError(
                "Could not download the official MalConv2 checkpoint.\n"
                "Please check your network connection or manually download:\n"
                f"  {MODEL_URL}\n"
                "and place it at:\n"
                f"  {dest_path}\n"
                f"Original error: {e}"
            ) from e

    def _model_path(self) -> str:
        official_path = os.path.join(MODEL_CACHE, MODEL_FILE)
        legacy_path = os.path.join(MODEL_CACHE, LEGACY_MODEL_FILE)
        if os.path.isfile(official_path) or not os.path.isfile(legacy_path):
            return official_path
        return legacy_path

    def is_available(self) -> bool:
        return TORCH_IMPORT_ERROR is None and os.path.isfile(self._model_path())

    def ensure_available(self) -> None:
        self._load_model()

    @staticmethod
    def _resolve_device(device: str = None) -> str:
        device = device or os.environ.get('MUNDA_MALCONV2_DEVICE')
        if device:
            normalized = device.lower()
            if normalized.startswith('cuda'):
                _require_torch()
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA was requested for MalConv2, but PyTorch does "
                        "not report an available CUDA device."
                    )
            return device
        return 'cuda' if TORCH_IMPORT_ERROR is None and torch.cuda.is_available() else 'cpu'

    # ── Training ─────────────────────────────────────────

    def train(self, filepaths: list, labels: list,
              epochs: int = 10, batch_size: int = 4,
              lr: float = 1e-4) -> None:
        """
        Fine-tune MalConv2 on custom raw binary data.

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
                self.lbls = lbls

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, i):
                with open(self.paths[i], 'rb') as f:
                    raw = f.read(MAX_FILE_BYTES)
                arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1
                return torch.tensor(arr), torch.tensor(self.lbls[i], dtype=torch.long)

        def pad_collate(batch):
            vectors = [x[0] for x in batch]
            labels_batch = torch.stack([x[1] for x in batch])
            return torch.nn.utils.rnn.pad_sequence(vectors, batch_first=True), labels_batch

        model = self._load_model()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            BinaryDataset(filepaths, labels),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pad_collate,
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _penult, _post_conv = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[MalConv2] Epoch {epoch + 1}/{epochs} - "
                  f"loss: {total_loss / len(loader):.4f}")

        save_path = os.path.join(MODEL_CACHE, MODEL_FILE)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'channels': 256,
                'filter_size': 256,
                'stride': 64,
                'embd_dim': 8,
            },
            save_path,
        )
        print(f"[MalConv2] Model saved -> {save_path}")


def _main():
    parser = argparse.ArgumentParser(description='Run MalConv2 on one file')
    parser.add_argument('--predict-json', metavar='FILE')
    parser.add_argument('--device', default=None, help='Torch device, e.g. cpu or cuda')
    parser.add_argument(
        '--ensure-available',
        action='store_true',
        help='Download/load the checkpoint and exit',
    )
    args = parser.parse_args()

    try:
        detector = MalConv2Detector(device=args.device)
        if args.ensure_available:
            detector.ensure_available()
            print(json.dumps({'available': True, 'device': detector.device}))
            return

        if args.predict_json:
            score = detector.predict(args.predict_json)
            print(json.dumps({'score': score}))
            return

        parser.print_help()
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    _main()
