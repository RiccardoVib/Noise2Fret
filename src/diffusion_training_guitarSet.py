"""
Created on Tue Nov 2 08:14:08 2025

@author: Riccardo Simionato

"""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from CheckpointManager import DiffusionCheckpointManager
from U_NET_Token import TokenUNet
from DiffusionUtils import save_losses, plot_losses
from utils import write_json
import json
from DiffusionModel import DiffusionModel
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from tab_metrics import print_tab_metrics, tab_metrics

class CustomDataset(Dataset):
    def __init__(self, data_list, mode="tab", input_feature_type="cqt"):
        self.data_list = data_list
        self.mode = mode
        self.input_feature_type = input_feature_type
        self.max_events = 64
        self.n_strings = 6
        self.n_classes = 22

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = np.load(self.data_list[index])
        if self.input_feature_type == "cqt":
            input_features = data["cqt"]
        elif self.input_feature_type == "melspec":
            input_features = data["mel_spec"]

        if self.mode == "F0":
            note_gt = data["F0"]
            frame_gt = data["frame_F0"]

        elif self.mode == "tab":
            note_gt = data["tab"]
            frame_gt = data["frame_tab"]
        bpm = data["tempo"]
        stft = data["stft"]  # (T, F)
        sf = data["sf"]  # (T, 1)
        b = data["b"]  # (T, 1)
        audio = data["audio"]  # (T*hop,)
        frame_len = input_features.shape[0]
        note_len = note_gt.shape[0]
        audio_len = audio.shape[0]

        return input_features, frame_gt, note_gt, frame_len, note_len, bpm, stft, sf, b, audio, audio_len

def tab_pad_collate(batch):
    input_features, frame_tab, note_tab, frame_len, note_len, bpm, stft, sf, b, audio, audio_len = zip(*batch)
    batch_size = len(input_features)

    frame_maxlen, note_maxlen, audio_maxlen = np.max(frame_len), np.max(note_len), np.max(audio_len)

    frame_len = np.asarray(frame_len)
    note_len = np.asarray(note_len)
    bpm = np.asarray(bpm)
    for batch_n in range(batch_size):
        frame_padlen = frame_maxlen - frame_len[batch_n]
        note_padlen = note_maxlen - note_len[batch_n]
        audio_padlen = audio_maxlen - audio[batch_n].shape[0]

        padded_input_features = np.pad(
            input_features[batch_n], [(0, frame_padlen), (0, 0)], 'constant')
        padded_stft = np.pad(stft[batch_n], [(0, frame_padlen), (0, 0)], 'constant')
        padded_sf   = np.pad(sf[batch_n],   [(0, frame_padlen), (0, 0)], 'constant')
        padded_b    = np.pad(b[batch_n],    [(0, frame_padlen), (0, 0)], 'constant')
        padded_audio = np.pad(audio[batch_n], [(0, audio_padlen)], 'constant')


        padded_note_tab = np.pad(
            note_tab[batch_n], [(0, note_padlen), (0, 0), (0, 0)], 'constant')
        padded_frame_tab = np.pad(
            frame_tab[batch_n], [(0, frame_padlen), (0, 0), (0, 0)], 'constant')

        if batch_n == 0:
            padded_input_features_out = np.expand_dims(
                padded_input_features, axis=0)
            padded_note_tab_out = np.expand_dims(padded_note_tab, axis=0)
            padded_frame_tab_out = np.expand_dims(padded_frame_tab, axis=0)
            padded_stft_out  = np.expand_dims(padded_stft,  axis=0)
            padded_sf_out    = np.expand_dims(padded_sf,    axis=0)
            padded_b_out     = np.expand_dims(padded_b,     axis=0)
            padded_audio_out = np.expand_dims(padded_audio, axis=0)
        else:
            padded_input_features_out = np.append(
                padded_input_features_out, np.expand_dims(padded_input_features, axis=0), axis=0)
            padded_note_tab_out = np.append(
                padded_note_tab_out, np.expand_dims(padded_note_tab, axis=0), axis=0)
            padded_frame_tab_out = np.append(
                padded_frame_tab_out, np.expand_dims(padded_frame_tab, axis=0), axis=0)

            padded_stft_out  = np.append(padded_stft_out,  np.expand_dims(padded_stft,  axis=0), axis=0)
            padded_sf_out    = np.append(padded_sf_out,    np.expand_dims(padded_sf,    axis=0), axis=0)
            padded_b_out     = np.append(padded_b_out,     np.expand_dims(padded_b,     axis=0), axis=0)
            padded_audio_out = np.append(padded_audio_out, np.expand_dims(padded_audio, axis=0), axis=0)

    # reverse sort by length
    sort_idx = np.argsort(frame_len)[::-1]
    padded_input_features_out = np.take(
        padded_input_features_out, sort_idx, axis=0)
    padded_note_tab_out = np.take(padded_note_tab_out, sort_idx, axis=0)
    padded_frame_tab_out = np.take(padded_frame_tab_out, sort_idx, axis=0)
    frame_len = np.take(frame_len, sort_idx, axis=0)
    note_len = np.take(note_len, sort_idx, axis=0)
    audio_len = np.take(audio_len, sort_idx, axis=0)
    bpm = np.take(bpm, sort_idx, axis=0)

    padded_stft_out  = np.take(padded_stft_out,  sort_idx, axis=0)
    padded_sf_out    = np.take(padded_sf_out,    sort_idx, axis=0)
    padded_b_out     = np.take(padded_b_out,     sort_idx, axis=0)
    padded_audio_out = np.take(padded_audio_out, sort_idx, axis=0)

    return  (
        torch.from_numpy(padded_input_features_out).float(),
        torch.from_numpy(padded_frame_tab_out),
        torch.from_numpy(padded_note_tab_out),
        torch.from_numpy(frame_len),
        torch.from_numpy(note_len),
        torch.from_numpy(bpm),
        torch.from_numpy(padded_stft_out).float(),
        torch.from_numpy(padded_sf_out).float(),
        torch.from_numpy(padded_b_out).float(),
        torch.from_numpy(padded_audio_out).float(),
        torch.from_numpy(audio_len)
    )


def train_diffusion_model(data_dir, model_path, noise_steps, base_channels, embed_dim, inject_feature_dim,
                          batch_size, epochs=10, lr=1e-4, use_pre=False, losses_str=[""], train_model=True):
    """Train the diffusion model on a dataset."""

    data_path = os.path.join(data_dir / "GuitarSet/",
        "data", "npz", f"original", "split", "*.npz")
    data_list = np.array(glob.glob(data_path, recursive=True))
    train_ratio = 0.9
    test_num = 0
    dev_data_list = [datapath for datapath in data_list if not (
        os.path.split(datapath)[1].startswith(f"0{test_num}_"))]
    random.shuffle(dev_data_list)
    train_data_list = dev_data_list[:int(
        round(len(dev_data_list) * train_ratio))]
    valid_data_list = dev_data_list[int(
            round(len(dev_data_list) * train_ratio)):]

    dataset = CustomDataset(train_data_list)
    dataset_test = CustomDataset(valid_data_list)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tab_pad_collate,
        num_workers=4,
        pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=4,
        shuffle=False,
        collate_fn=tab_pad_collate,
        num_workers=4,
        pin_memory=False, drop_last=True)

    early_stopping_count = 0

    # Initialize checkpoint manager
    ckpt_manager = DiffusionCheckpointManager(model_path / "my_checkpoints")

    # Store the model params in a json file in model_dir
    model_params = {
        'input_size (T)': int(dataset.max_events),
        'input_size (F)': int(dataset.n_classes),
        'hidden_size': int(base_channels),
        'batch_size': int(batch_size),
        'inject_feature_size': int(inject_feature_dim),
        'losses_str': losses_str,
    }
    
    print(f"losses: {losses_str}")
    print(f"model_params: {model_params}")
    print(f"Saving model params in {model_path}")
    write_json(model_params, model_path / "params.json", False)

    # Define model components
    model = TokenUNet(in_channels=dataset.n_strings * embed_dim,
                      base_channels=base_channels,
                      inject_feature_dim=inject_feature_dim,
                      use_pre=use_pre,
                      max_len=dataset.max_events
                      )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print('\n batch_size: ', batch_size)
    print('\n hidden_size: ', base_channels)
    print('\n embed_dim: ', embed_dim)
    print('\n input_size (T): ', int(dataset.max_events))
    print('\n input_size (F): ', int(dataset.n_classes))
    print('\n inject_channels: ', base_channels)
    print('\n noise_steps: ', noise_steps)
    print('\n dataset len: ', len(dataset))
    print('\n epochs ', epochs)
    print('\n')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda available :', torch.cuda.is_available())

    model = model.to(device)
    print(all(p.is_cuda for p in model.parameters()))  # True if all params on GPU

    diffusion = DiffusionModel(model=model, noise_steps=noise_steps, embed_dim=embed_dim, n_classes=dataset.n_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(diffusion.encoder.parameters()) + list(diffusion.embeddings.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Load last checkpoint
    checkpoint = ckpt_manager.load_last_checkpoint(diffusion, optimizer, scheduler, device=device)
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        best_loss = checkpoint['best_val_loss']
        print(f"Loaded best model with metric: {best_loss}")
    else:
        print("Starting training from scratch")
        best_loss = float('inf')

    if train_model:
        train_losses, val_losses = [], []
        # Training loop
        for epoch in range(epochs):
            train_batches = 0
            train_loss, val_loss = 0, 0
            fret_train_loss, pc_train_loss, cof_train_loss, s_train_loss, h_train_loss = 0, 0, 0, 0, 0

            model.train()
            for cqt, frame_gt, token, frame_len, note_len, bpm, stft, sf, b, audio, audio_len in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}",  disable=True):
                cqt = cqt.to(diffusion.device)
                token = token.to(diffusion.device)
                audio = audio.to(diffusion.device).unsqueeze(-1)
                stft = stft.to(diffusion.device)
                sf = sf.to(diffusion.device)
                b = b.to(diffusion.device)
                features = torch.cat(
                    [stft, sf, b], dim=-1)
                
                loss, fret_loss, pc_loss, cof_loss, string_loss, hs_loss = diffusion.train_step(optimizer=optimizer, batch=[token, None, audio, features],
                                            losses_str=losses_str)
                train_loss += loss
                fret_train_loss += fret_loss
                pc_train_loss += pc_loss
                cof_train_loss += cof_loss
                s_train_loss += string_loss
                h_train_loss += hs_loss
                
                train_batches += 1

            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            print(f'Epoch {epoch + 1}: Fret Loss: {fret_train_loss/train_batches:.6f}, Pc Loss: {pc_train_loss/train_batches:.6f}, Cof Loss: {cof_train_loss/ train_batches:.6f}, String Loss: {s_train_loss/ train_batches:.6f}, HandSpan Loss: {h_train_loss/ train_batches:.6f}\n')

            # Validation phase
            if (epoch + 1) % 1 == 0:
                total_val_loss, total_acc, total_samples = 0, 0, 0
                val_batches = 0
                model.eval()
                with torch.no_grad():
                    for cqt, frame_gt, token, frame_len, note_len, bpm, stft, sf, b, audio, audio_len in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", disable=True):
                        cqt = cqt.to(diffusion.device)
                        token = token.to(diffusion.device)
                        audio = audio.to(diffusion.device).unsqueeze(-1)
                        stft = stft.to(diffusion.device)
                        sf = sf.to(diffusion.device)
                        b = b.to(diffusion.device)
                        features = torch.cat(
                            [stft, sf, b], dim=-1)
                
                        loss, acc = diffusion.val_step(batch=[token, None, audio, features])

                        total_val_loss += loss
                        total_acc += acc
                        val_batches += 1

                avg_val_loss = total_val_loss / val_batches
                avg_acc = total_acc / val_batches
                val_losses.append(avg_val_loss)

                print(
                    f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Acc: {avg_acc:.6f}')
                print(f'Learning Rate {optimizer.param_groups[0]["lr"]:.2e}')

                # Save latest checkpoint
                state_dict = {
                    'epoch': epoch,
                    'model_state_dict': diffusion.model.state_dict(),
                    'time_embedding_state_dict': diffusion.encoder.state_dict(),
                    'embedding_state_dict': diffusion.embeddings.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_loss
                }

                # Save last checkpoint
                ckpt_manager.save_last_checkpoint(state_dict)

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    # Save best checkpoint (assuming this is the best model so far)
                    state_dict = {
                        'epoch': epoch,
                        'model_state_dict': diffusion.model.state_dict(),
                        'time_embedding_state_dict': diffusion.encoder.state_dict(),
                        'embedding_state_dict': diffusion.embeddings.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'best_val_loss': best_loss
                    }
                    ckpt_manager.save_checkpoint(state_dict, is_best=True)
                    print(f"Epoch {epoch + 1}, Validation loss improved: ", best_loss)
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    print(f"Epoch {epoch + 1}, Validation loss did not improved.")
                    print(f"early_stopping_count: {early_stopping_count}")
                    if early_stopping_count == 70:
                        print(f'No improvements over 70 epochs -> stopping...')
                        break

                # Generate and visualize samples
                if ((epoch + 1) % 100 == 0 and epoch != epochs - 1) or epochs == 1:
                  
                    predicted_indices, predicted_tab = visualize_samples(token, None, audio, features, diffusion)

                    # decode whole batch at once (shape B, maxevents)
                    predicted_item, target_item = vectors_to_text_token(predicted_indices, token)

                    # Per item: keep as list[list[str]] — one inner list per chord event
                    output_path = model_path / (str(epoch) + "results.txt")
                    print_results(target_item, predicted_item, output_path)

            filename = model_path / ('losses.json')
            save_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)
            filename = model_path / ('loss_plot.png')
            plot_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)
            # Update learning rate scheduler with validation loss
            scheduler.step()
            
    # Load best checkpoint
    best_checkpoint = ckpt_manager.load_best_checkpoint(diffusion, device=device)
    if best_checkpoint:
        print(f"Loaded best model with metric: {best_checkpoint.get('best_val_loss', 0)}")

    losses_dict = {
        'best_val_loss': best_loss
    }
    filename = model_path / ('test_losses.txt')
    with open(filename, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {filename}")
        
    gt_chunks, pred_chunks = [], []
    with torch.no_grad():
        for cqt, frame_gt, token, frame_len, note_len, bpm, stft, sf, b, audio, audio_len in tqdm(val_loader, desc=f"Test", disable=True):
            cqt = cqt.to(diffusion.device)
            token = token.to(diffusion.device)
            audio = audio.to(diffusion.device).unsqueeze(-1)
            stft = stft.to(diffusion.device)
            sf = sf.to(diffusion.device)
            b = b.to(diffusion.device)
            features = torch.cat(
                [stft, sf, b], dim=-1)

            predicted_indices, predicted_tab = visualize_samples(token, None, audio, features, diffusion)
            predicted_item, target_item = vectors_to_text_token(predicted_indices, token)

            # normalise both to integer IDs (B, T, 6) before storing
            gt_ids = token.argmax(dim=-1).cpu() if token.ndim == 4 else token.cpu()
            pred_ids = predicted_indices.argmax(dim=-1).cpu() if predicted_indices.ndim == 4 else predicted_indices.cpu()
            gt_chunks.append(gt_ids)
            pred_chunks.append(pred_ids)
            
        all_gt = torch.cat(gt_chunks, dim=0)  # (N, T, 6)
        all_pred = torch.cat(pred_chunks, dim=0)  # (N, T, 6)
        
        np.savez(model_path/"predictions", gt=all_gt, pred=all_pred)
        print(f"Predictions cached → {model_path}")
   
   
    avg = tab_metrics(all_gt, all_pred)
    out_path = model_path / f"metrics_Test.txt"
    print_tab_metrics(avg, save_path=str(out_path), prefix="Test set")
        
    all_pred, all_gt = vectors_to_text_token(all_pred, all_gt)
    output_path = model_path / "predictions_TEST.txt"
    print_results(all_gt, all_pred, output_path)

    return 42

def vectors_to_text_token(predicted_indices, token):
    """
    predicted_indices : (B, T, 6)  argmax class indices from diffusion
    token             : (B, T, 6)  ground truth class indices  ← argmax of one-hot target

    Returns:
        predicted_decoded : list[list[list[str]]]  B × T × n_active_strings
        token_decoded     : list[list[list[str]]]  B × T × n_active_strings
    """
    if isinstance(predicted_indices, torch.Tensor):
        predicted_indices = predicted_indices.cpu()
    if isinstance(token, torch.Tensor):
        token = token.cpu()

    # token is (B, T, 6, 21) one-hot from dataset → need argmax to get indices
    # if token still comes as one-hot (B, T, 6, 21), reduce it first:
    if token.ndim == 4:
        token = token.argmax(dim=-1)  # (B, T, 6)

    def decode_indices(indices_BTx6):
        """(B, T, 6) → list[list[list[str]]]"""
        result = []
        for b in range(indices_BTx6.shape[0]):
            item = []
            for t in range(indices_BTx6.shape[1]):
                frame_tokens = []
                for s_idx in range(6):
                    cls = int(indices_BTx6[b, t, s_idx])
                    if cls == 0:
                        pass  # muted — skip or keep as needed
                    else:
                        frame_tokens.append(f"s{s_idx + 1}:f{cls - 1}")
                item.append(frame_tokens)
            result.append(item)
        return result

    predicted_decoded = decode_indices(predicted_indices)  # B × T × n_strings
    token_decoded = decode_indices(token)  # B × T × n_strings

    return predicted_decoded, token_decoded


def visualize_samples(inputs, prev_input, audio, cond, diffusion):
    """Visualize samples from the diffusion model."""
    z = diffusion.sample(input=inputs, prev_input=prev_input, audio=audio, cond=cond,
                         num_steps=diffusion.noise_steps)

    tab = diffusion.decode(z)
    tab_indices = tab.argmax(dim=-1)  # (B, T, 6)  — integer class per string
    return tab_indices, tab


def print_results(tokens, predicted_tokens, output_path):
    n_mismatch = 0
    examples = []
    for idx, (tgt, pred) in enumerate(zip(tokens, predicted_tokens)):
        target_set = set(frozenset(group) for group in tgt)
        predicted_set = set(frozenset(group) for group in pred)
        if target_set != predicted_set:
            n_mismatch += 1
        examples.append((idx, tgt, pred, target_set, predicted_set))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        if n_mismatch == 0:
            f.write(f"✓ All {len(tokens)} sampled frames match.\n")
        else:
            f.write(f"⚠ {n_mismatch}/{len(tokens)} frames have mismatch!\n")
        f.write('=' * 60 + '\n\n')

        for idx, tgt, pred, raw_s, dec_s in examples:
            f.write(f"Frame {idx}\n")
            f.write('-' * 40 + '\n')
            f.write('--- Target tokens ---\n')
            for j, group in enumerate(tgt):
                f.write(f"  event {j}  {' '.join(group)}\n")
            f.write('--- Predicted tokens ---\n')
            for j, group in enumerate(pred):
                f.write(f"  event {j}  {' '.join(group)}\n")
            f.write('--- Diff ---\n')
            f.write(f"  missing : {sorted(raw_s - dec_s)}\n")
            f.write(f"  extra   : {sorted(dec_s - raw_s)}\n")
            f.write('\n')

    return
    
#Example usage
if __name__ == "__main__":
    import os
    from pathlib import Path
    from utils import find_folder_upward

    current_dir = Path(os.getcwd())
    print(f"current_dir: {current_dir}")
    files_dir = find_folder_upward(folder_name="Files", start_path=current_dir)

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    n_batches = 32
    embed_dim = 32
    hidden_dim = 64
    noise_steps = 500
    epochs = 1000
    lr = 3e-4
    inject_feature_dim = 515
    losses_str = [""]
    
    model_name = "_".join(
            ['Audio2Tab', "GuitarSet", "H", str(hidden_dim), "I", str(inject_feature_dim), "U", str(use_pre), "_f1_p01_4"])#losses_str[0]])
    model_path = script_dir.parent.parent / "TrainedModels" / (model_name)

    print(f"model_name: {model_name}")
    print(f"model_path: {model_path}")

    train_diffusion_model(data_dir=files_dir,
                              model_path=model_path,
                              noise_steps=noise_steps,
                              base_channels=hidden_dim,
                              inject_feature_dim=inject_feature_dim,
                              embed_dim=embed_dim,
                              batch_size=n_batches,
                              use_pre=use_pre,
                              epochs=epochs,
                              lr=lr,
                              losses_str=losses_str,
                              train_model=True
                              )
