import torch
import torch.nn.functional as F
import math
from torch import nn

class FretNet(nn.Module):
    """
    FretNet-style architecture for discrete guitar tablature prediction only.

    from cwitkowitz/guitar-transcription-continuous:
      - relative pitch-deviation head removed
      - onset-detection head removed

     CNN encoder (identical to FretNet/TabCNN) + a per-string fret-classification head.

     Input
     ----------
     feats : Tensor (B, T, C, F, W)
         B - batch size
         T - number of sequence steps you choose to process together (can be 1)
         C - number of input feature channels (e.g. 1 for a CQT magnitude)
         F - number of frequency bins (this is `dim_in` below)
         W - context window width: number of adjacent input frames stacked
             together per prediction, same convention as TabCNN (`frame_width`)

     Output
     ----------
     logits : Tensor (B, T, n_strings, n_classes)
         Raw (unnormalized) per-string, per-fret-class scores.
     """

    def __init__(self, dim_in, in_channels, frame_width, n_strings=6,
                 n_classes=21, model_complexity=1, device='cpu'):
        """
        Parameters
        ----------
        dim_in : int
          Number of frequency bins in the input features (F above)
        in_channels : int
          Number of channels in the input features (C above)
        frame_width : int
          Number of frames stacked per input window (W above) -- must match
          whatever width you use when you build feats from audio
        n_strings : int
          Number of guitar strings (6)
        n_classes : int
          Number of fret classes per string, i.e. GOATFrameDataset.n_classes
          (0 = muted, 1 = open, 2 = fret 1, ..., matches `_fret_to_class`)
        model_complexity : int
          Scales the number of conv filters, as in the original FretNet
        """

        super().__init__()

        self.dim_in = dim_in
        self.in_channels = in_channels
        self.frame_width = frame_width
        self.n_strings = n_strings
        self.n_classes = n_classes
        self.device = device
        self.max_grad_norm = 1.0  # Prevent exploding gradients

        # Number of filters for each convolutional block
        nf1 = 16 * model_complexity
        nf2 = 32 * model_complexity
        nf3 = 48 * model_complexity

        # Kernel size for each convolutional block
        ks1 = (3, 3)
        ks2 = ks1
        ks3 = ks2

        # Padding amount for each convolutional block
        pd1 = (1, 1)
        pd2 = (1, 0)
        pd3 = pd2

        # Reduction size for each pooling operation
        rd2 = (2, 1)
        rd3 = rd2

        # Dropout percentages for each dropout operation
        dp2 = 0.5
        dp3 = 0.25
        dpx = 0.10

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, nf1, ks1, padding=pd1),
            nn.BatchNorm2d(nf1),
            nn.ReLU(),
            nn.Conv2d(nf1, nf1, ks1, padding=pd1),
            nn.BatchNorm2d(nf1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(nf1, nf2, ks2, padding=pd2),
            nn.BatchNorm2d(nf2),
            nn.ReLU(),
            nn.Conv2d(nf2, nf2, ks2, padding=pd2),
            nn.BatchNorm2d(nf2),
            nn.ReLU()
        )

        self.pool2 = nn.Sequential(
            nn.MaxPool2d(rd2),
            nn.Dropout(dp2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(nf2, nf3, ks3, padding=pd3),
            nn.BatchNorm2d(nf3),
            nn.ReLU(),
            nn.Conv2d(nf3, nf3, ks3, padding=pd3),
            nn.BatchNorm2d(nf3),
            nn.ReLU()
        )

        self.pool3 = nn.Sequential(
            nn.MaxPool2d(rd3),
            nn.Dropout(dp3)
        )

        def pooling_reduction(dim_in, times=1):
            # Define a simple recursive function to compute dimensionality after all pooling operations
            return dim_in if times <= 0 else pooling_reduction(math.ceil(dim_in / 2), times - 1)

        # Compute the dimensionality of feature embeddings
        features_dim_in = nf3 * pooling_reduction(dim_in, times=2)
        # Reduce the dimensionality by half before feeding to output layers
        features_dim_int = features_dim_in // 2

        # Initialize the discrete tablature estimation head
        self.tablature_head = nn.Sequential(
            nn.Linear(features_dim_in, features_dim_int),
            nn.ReLU(),
            nn.Dropout(dpx),
            nn.Linear(features_dim_int, n_strings * n_classes)
        )

    def train_step(self, optimizer, batch):
        optimizer.zero_grad()
        target, input = batch

        logits = self.forward(input)
        loss = self.get_loss(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()
        return loss.item()

    def val_step(self, batch):
        target, input = batch

        logits = self.forward(input)
        loss = self.get_loss(logits, target)
        acc = self.avg_acc(logits, target)
        return loss.item(), acc

    def avg_acc(self, logits: torch.Tensor, target_ids: torch.Tensor) -> float:

        if target_ids.dim() == 4: # if (B, T, n_strings, n_classes)
            target_ids = target_ids.argmax(dim=-1)  # (B, T, n_strings)

        pred_cls = logits.argmax(dim=-1)                   # (B, T, 6)
        return (pred_cls == target_ids).float().mean().item()

    def forward(self, feats):
        """
        Perform the main processing steps for FretNet.

        Parameters
        ----------
        feats : Tensor (B x T x C x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          C - number of channels in features
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        logits : Tensor (B, T, n_strings, n_classes)
        """

        batch_size, seq_len = feats.size(0), feats.size(1)

        # Collapse the T (sequence) axis into the batch axis, as in TabCNN/FretNet
        feats = feats.reshape(-1, self.in_channels, self.dim_in, self.frame_width)

        # Shared conv encoder
        embeddings = self.pool3(self.conv3(self.pool2(self.conv2(self.conv1(feats)))))
        embeddings = embeddings.flatten(1)

        # Restore the (B, T, ...) shape
        embeddings = embeddings.view(batch_size, seq_len, -1)

        logits = self.tablature_head(embeddings)
        logits = logits.view(batch_size, seq_len, self.n_strings, self.n_classes)

        return logits

    def get_loss(self, logits, target):
        """
        Cross-entropy loss computed directly against a one-hot fret-class target,

        Parameters
        ----------
        logits : Tensor (B, T, n_strings, n_classes)
          Raw scores from forward()
        target : Tensor (B, T, n_strings, n_classes) or (T, n_strings, n_classes)
          One-hot ground-truth fret classes (unbatched inputs are auto-unsqueezed)

        Returns
        ----------
        loss : Tensor (scalar)
          Cross-entropy summed across strings, averaged across batch and time
        """

        target_idx = target.argmax(dim=-1)  # (B, T, n_strings)

        # Flatten string axis into the batch so F.cross_entropy sees (N, n_classes)
        logits_flat = logits.reshape(-1, self.n_classes)
        target_flat = target_idx.reshape(-1)

        loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
        loss = loss.view(target_idx.shape)  # (B, T, n_strings)

        loss = loss.sum(dim=-1)  # sum across strings, like amt_tools' LogisticBank
        loss = loss.mean()  # average across batch and time

        return loss

    @torch.no_grad()
    def predict(self, logits):
        """
        Convert raw logits into predicted fret-class indices.

        logits : Tensor (B, T, n_strings, n_classes)
        returns : Tensor (B, T, n_strings) of class indices, directly comparable
                  to `target.argmax(dim=-1)` or usable with
                  GOATFrameDataset.decode_tab_target (after indexing one T-step)
        """
        return logits.argmax(dim=-1)


class TabCNN(nn.Module):
    """
    Architecture mirror of the original Keras model     
    from andywiggins/tab-cnn:
        Conv2d(32,3×3) → Conv2d(64,3×3) → Conv2d(64,3×3)
        → MaxPool(2×2) → Dropout(0.25)
        → Flatten → Linear(128) → Dropout(0.5)
        → Linear(6×24) → reshape (6,24) → per-string softmax
    """

    NUM_STRINGS = 6

    def __init__(self, dim_in, n_classes=22, con_win_size=9, device=torch.device('cpu')):
        super().__init__()
        freq_bins = dim_in
        self.device = device
        self.n_classes = n_classes
        self.max_grad_norm = 1.0  # Prevent exploding gradients

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)

        # compute flat size after conv stack + pool
        flat = self._flat_size(freq_bins, con_win_size)
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, self.NUM_STRINGS * self.n_classes)

    def _flat_size(self, freq_bins, con_win_size):
        """Dry-run to compute the flatten dimension."""
        with torch.no_grad():
            x = torch.zeros(1, 1, freq_bins, con_win_size)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
        return x.numel()

    def avg_acc(self, pred, target):
        pred_cls = pred.argmax(dim=-1)  # (B, 6)
        target_cls = target.argmax(dim=-1)  # (B, 6)
        return (pred_cls == target_cls).float().mean().item()

    # ─────────────────────────────────────────────
    #  Loss  (catcross_by_string → sum of 6 CE)
    # ─────────────────────────────────────────────

    def catcross_by_string(self, pred, target):
        """
        pred   : (B, 6, 24)  probabilities
        target : (B, 6, 24)  one-hot
        Returns scalar loss = sum of per-string cross-entropy.
        """
        loss = 0.0
        for s in range(6):
            # F.cross_entropy expects class-index targets; convert from one-hot
            t = target[:, 0, s, :].argmax(dim=-1)
            # (B,)
            loss += F.cross_entropy(pred[:, s, :], t)  # log of softmax internally
        return loss

    def train_step(self, optimizer, batch):
        optimizer.zero_grad()
        target, input = batch
        logits = self.forward(input)
        loss = self.catcross_by_string(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()
        return loss.item()

    def val_step(self, batch):
        target, input = batch
        logits = self.forward(input)
        loss = self.catcross_by_string(logits, target)
        acc = self.avg_acc(logits, target)
        return loss.item(), acc

    def forward(self, x):
        # x: (B, 1, F, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, start_dim=1)   # (B, flat)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)                      # (B, 126)
        x = x.view(-1, self.NUM_STRINGS, self.n_classes)  # (B, 6, 24)
        #x = F.softmax(x, dim=-1)             # per-string softmax
        return x


