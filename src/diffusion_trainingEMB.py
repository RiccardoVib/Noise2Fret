"""
Created on Tue Nov 2 08:14:08 2025

@author: Riccardo Simionato

"""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from CheckpointManager import DiffusionCheckpointManager
from U_NET_Token import TokenUNet
from GOATDataset import GOATFrameDataset
from DiffusionUtils import save_losses, plot_losses
from utils import write_json, wait_for_allowed_time
import json
from DiffusionModel import DiffusionModel
from FeaturesExtractor import compute_audio_features
from tab_metrics import tab_metrics, print_tab_metrics
import numpy as np


def train_diffusion_model(data_dir, model_path, noise_steps, base_channels, inject_feature_dim, feat, embed_dim,
                          batch_size, use_pre, epochs=10, lr=1e-4, losses_str=[""], train_model=True):
    """Train the diffusion model on a dataset."""
    # Setup dataloader
    dataset = GOATFrameDataset(
        root_dir=data_dir / "GOAT",
        data_dir=data_dir / "train",
    )

    dataset_test = GOATFrameDataset(
        root_dir=data_dir / "GOAT",
        data_dir=data_dir / "test",
        max_events=dataset.max_events,
    )

    dataset_val = GOATFrameDataset(
        root_dir=data_dir / "GOAT",
        data_dir=data_dir / "Validation",
        max_events=dataset.max_events,
    )

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, drop_last=True,
                                                  num_workers=4, pin_memory=True)

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
        'feat': feat,
        'losses_str': losses_str,
    }
    print(f"feat: {feat}")
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

    diffusion = DiffusionModel(model=model, noise_steps=noise_steps, embed_dim=embed_dim).to(device)
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
            #wait_for_allowed_time()
            train_batches = 0
            train_loss, val_loss = 0, 0
            model.train()
            for audio, token, prev_token in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}",
                                                 disable=True):
                audio = audio.to(diffusion.device)
                token = token.to(diffusion.device)
                prev_token = prev_token.to(diffusion.device)
                # Compute audio features before training step
                if inject_feature_dim > 1:
                    features = compute_audio_features(audio, sr=16000)
                    if feat == "all":
                        features = torch.cat(
                            [features["cqt_mag"], features["stft_mag"], features["spectral_flux"],
                             features["brightness"]], dim=-1)
                    elif feat == "alls":
                        features = torch.cat(
                            [features["stft_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                    elif feat == "allc":
                        features = torch.cat(
                            [features["cqt_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                    elif feat == "stft":
                        features = features["stft_mag"]
                    elif feat == "sf":
                        features = features["spectral_flux"]
                    elif feat == "b":
                        features = features["brightness"]
                    elif feat == "cqt":
                        features = features["cqt_mag"]
                    elif feat == "stft+sf":
                        features = torch.cat(
                            [features["stft_mag"], features["spectral_flux"]], dim=-1)
                    elif feat == "stft+b":
                        features = torch.cat(
                            [features["stft_mag"], features["brightness"]], dim=-1)
                    elif feat == "sf+b":
                        features = torch.cat(
                            [features["spectral_flux"], features["brightness"]], dim=-1)
                else:
                    features = audio

                loss = diffusion.train_step(optimizer=optimizer, batch=[token, prev_token, audio, features],
                                            losses_str=losses_str)  # , criterion=loss_fn)
                train_loss += loss
                train_batches += 1

            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)

            # Validation phase
            if (epoch + 1) % 1 == 0:
                total_val_loss, total_acc, total_samples = 0, 0, 0
                val_batches = 0
                model.eval()
                with torch.no_grad():
                    for audio, token, prev_token in tqdm(test_dataloader, desc=f"Validation Epoch {epoch + 1}",
                                                         disable=True):
                        audio = audio.to(diffusion.device)
                        token = token.to(diffusion.device)
                        prev_token = prev_token.to(diffusion.device)
                        if inject_feature_dim > 1:
                            features = compute_audio_features(audio, sr=16000)
                            if feat == "all":
                                features = torch.cat(
                                    [features["cqt_mag"], features["stft_mag"], features["spectral_flux"],
                                     features["brightness"]], dim=-1)
                            elif feat == "alls":
                                features = torch.cat(
                                    [features["stft_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                            elif feat == "allc":
                                features = torch.cat(
                                    [features["cqt_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                            elif feat == "stft":
                                features = features["stft_mag"]
                            elif feat == "sf":
                                features = features["spectral_flux"]
                            elif feat == "b":
                                features = features["brightness"]
                            elif feat == "cqt":
                                features = features["cqt_mag"]
                            elif feat == "stft+sf":
                                features = torch.cat(
                                    [features["stft_mag"], features["spectral_flux"]], dim=-1)
                            elif feat == "stft+b":
                                features = torch.cat(
                                    [features["stft_mag"], features["brightness"]], dim=-1)
                            elif feat == "sf+b":
                                features = torch.cat(
                                    [features["spectral_flux"], features["brightness"]], dim=-1)
                        else:
                            features = audio

                        loss, acc = diffusion.val_step(batch=[token, prev_token, audio, features])

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
                    if inject_feature_dim > 1:
                        features = compute_audio_features(audio, sr=16000)
                        if feat == "all":
                            features = torch.cat(
                                [features["cqt_mag"], features["stft_mag"], features["spectral_flux"],
                                 features["brightness"]], dim=-1)
                        elif feat == "alls":
                            features = torch.cat(
                                [features["stft_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                        elif feat == "allc":
                            features = torch.cat(
                                [features["cqt_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                        elif feat == "stft":
                            features = features["stft_mag"]
                        elif feat == "sf":
                            features = features["spectral_flux"]
                        elif feat == "b":
                            features = features["brightness"]
                        elif feat == "cqt":
                            features = features["cqt_mag"]
                        elif feat == "stft+sf":
                            features = torch.cat(
                                [features["stft_mag"], features["spectral_flux"]], dim=-1)
                        elif feat == "stft+b":
                            features = torch.cat(
                                [features["stft_mag"], features["brightness"]], dim=-1)
                        elif feat == "sf+b":
                            features = torch.cat(
                                [features["spectral_flux"], features["brightness"]], dim=-1)
                    else:
                        features = audio

                    predicted_indices, predicted_tab = visualize_samples(token, prev_token, audio, features, diffusion)

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

    audios, tokens, predicted_tokens, top5_items = [], [], [], []
    # Visualize the diffusion process
    model.eval()

    gt_chunks, pred_chunks = [], []
    with torch.no_grad():
        for audio, token, prev_token in tqdm(test_dataloader, desc=f"Test",
                                             disable=True):
            audio = audio.to(diffusion.device)
            token = token.to(diffusion.device)
            prev_token = prev_token.to(diffusion.device)

            if inject_feature_dim > 1:
                features = compute_audio_features(audio, sr=16000)
                if feat == "all":
                    features = torch.cat(
                        [features["cqt_mag"], features["stft_mag"], features["spectral_flux"], features["brightness"]],
                        dim=-1)
                elif feat == "alls":
                    features = torch.cat(
                        [features["stft_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                elif feat == "allc":
                    features = torch.cat(
                        [features["cqt_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                elif feat == "stft":
                    features = features["stft_mag"]
                elif feat == "sf":
                    features = features["spectral_flux"]
                elif feat == "b":
                    features = features["brightness"]
                elif feat == "cqt":
                    features = features["cqt_mag"]
                elif feat == "stft+sf":
                    features = torch.cat(
                        [features["stft_mag"], features["spectral_flux"]], dim=-1)
                elif feat == "stft+b":
                    features = torch.cat(
                        [features["stft_mag"], features["brightness"]], dim=-1)
                elif feat == "sf+b":
                    features = torch.cat(
                        [features["spectral_flux"], features["brightness"]], dim=-1)
            else:
                features = audio

            predicted_indices, predicted_tab = visualize_samples(token, prev_token, audio, features, diffusion)
            predicted_item, target_item = vectors_to_text_token(predicted_indices, token)

            # normalise both to integer IDs (B, T, 6) before storing
            gt_ids = token.argmax(dim=-1).cpu() if token.ndim == 4 else token.cpu()
            pred_ids = predicted_indices.argmax(
                dim=-1).cpu() if predicted_indices.ndim == 4 else predicted_indices.cpu()
            gt_chunks.append(gt_ids)
            pred_chunks.append(pred_ids)

        all_gt = torch.cat(gt_chunks, dim=0)  # (N, T, 6)
        all_pred = torch.cat(pred_chunks, dim=0)  # (N, T, 6)
        np.savez(model_path / "predictions", gt=all_gt, pred=all_pred)
        print(f"Predictions cached → {model_path}")

    avg = tab_metrics(all_gt, all_pred)
    out_path = model_path / f"metrics_Test.txt"
    print_tab_metrics(avg, save_path=str(out_path), prefix="Test set")

    val_dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)
    with torch.no_grad():
        for audio, token, prev_token in tqdm(val_dataloader, desc=f"Test", disable=True):
            audio = audio.to(diffusion.device)
            token = token.to(diffusion.device)
            prev_token = prev_token.to(diffusion.device)

            if inject_feature_dim > 1:
                features = compute_audio_features(audio, sr=16000)
                if feat == "all":
                    features = torch.cat(
                        [features["cqt_mag"], features["stft_mag"], features["spectral_flux"], features["brightness"]],
                        dim=-1)
                elif feat == "alls":
                    features = torch.cat(
                        [features["stft_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                elif feat == "allc":
                    features = torch.cat(
                        [features["cqt_mag"], features["spectral_flux"], features["brightness"]], dim=-1)
                elif feat == "stft":
                    features = features["stft_mag"]
                elif feat == "sf":
                    features = features["spectral_flux"]
                elif feat == "b":
                    features = features["brightness"]
                elif feat == "cqt":
                    features = features["cqt_mag"]
                elif feat == "stft+sf":
                    features = torch.cat(
                        [features["stft_mag"], features["spectral_flux"]], dim=-1)
                elif feat == "stft+b":
                    features = torch.cat(
                        [features["stft_mag"], features["brightness"]], dim=-1)
                elif feat == "sf+b":
                    features = torch.cat(
                        [features["spectral_flux"], features["brightness"]], dim=-1)
            else:
                features = audio

            predicted_indices, predicted_tab = visualize_samples(token, prev_token, audio, features, diffusion)
            predicted_item, target_item = vectors_to_text_token(predicted_indices, token)
            predicted_tokens.append(predicted_item[0])
            tokens.append(target_item[0])

    predicted_tokens = predicted_tokens[:5]
    # audios = torch.cat(audios[:5], dim=0)
    tokens = tokens[:5]
    output_path = model_path / "predictions.txt"

    print_results(tokens, predicted_tokens, output_path)

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
# Example usage
if __name__ == "__main__":
    import os
    from pathlib import Path
    from utils import find_folder_upward

    current_dir = Path(os.getcwd())
    print(f"current_dir: {current_dir}")
    files_dir = find_folder_upward(folder_name="Files", start_path=current_dir)
    ROOT_DIR = files_dir / "GOAT_processed_0.1"


    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    n_batches = 128
    noise_steps = 500
    #noise_steps = 1
    epochs = 1000
    #epochs = 1
    lr = 3e-4
    inject_feature_dim = 515#, 514, 513
    use_pre = True
    embed_dim = 32
    hidden_dims = [64, 64, 64, 64, 64, 64]
    inject_feature_dims = [515, 513, 514, 514, 1, 1, 2]
    inject_feature_dims = [515]
    feats = ["all", "stft", "stft+sf", "stft+b", "sf", "b", "sf+b"]
    feats = ["all"]

    addtional_name = "_TabEmbPROVA01"

    from itertools import combinations

    elements = ["f", "p", "c", "s", "h"]
    elements = [""]

    losses_strs = []
    for r in range(1, len(elements) + 1):
        losses_strs.extend(list(c) for c in combinations(elements, r))

    to_remove = [
        ["f", "p", "c"],
        ["f", "p", "c", "s", "h"],
        ["f", "p"],
        ["f"],
    ]

    # Filter: keep combos not in the exclusion list
    # Use sorted() so order doesn't matter during comparison
    losses_strs = [c for c in losses_strs if sorted(c) not in [sorted(x) for x in to_remove]]


    for hidden_dim, feat, inject_feature_dim, losses_str in zip(hidden_dims, feats, inject_feature_dims, losses_strs):

        l = "".join(losses_str)
        model_name = "_".join(
            ['Audio2Tab', "H", str(hidden_dim), "I", str(inject_feature_dim), "U", str(use_pre),
             "feat", str(feat), l])
        model_path = script_dir.parent.parent / "TrainedModels" / (model_name + addtional_name)

        print(f"model_name: {model_name}")
        print(f"model_path: {model_path}")


        train_diffusion_model(data_dir=ROOT_DIR,
                              model_path=model_path,
                              noise_steps=noise_steps,
                              base_channels=hidden_dim,
                              inject_feature_dim=inject_feature_dim,
                              embed_dim=embed_dim,
                              feat=feat,
                              batch_size=n_batches,
                              use_pre=use_pre,
                              epochs=epochs,
                              lr=lr,
                              losses_str=losses_str,
                              )
