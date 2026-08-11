import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from CheckpointManager import CheckpointManager
from GOATDataset import GOATFrameDataset
from utils import write_json
import json
from DiffusionUtils import save_losses, plot_losses
from FeaturesExtractor import compute_cqt, compute_hcqt
from tab_metrics import tab_metrics, print_tab_metrics
import numpy as np
from Baselines import FretNet, TabCNN
from AuxiliaryLoss import (
    pc_tokens_to_binary,
    fret_distance,
    cof_chord_distance,
    jaccard_tonal_distance,
    hand_span_penalty,
    string_activity_jaccard_loss,
)
from DiffusionModel import DiffusionModel


def train_model(data_dir, model_path, batch_size, baseline, epochs=10, lr=1e-4, train_model=True):
    """Train the model on a dataset."""
    # Setup dataloader
    dataset = GOATFrameDataset(
        root_dir=data_dir / "GOAT",
        data_dir=data_dir / "train",
        max_events=1,
        fs=22050,
        discard_multi_event=True
    )

    dataset_test = GOATFrameDataset(
        root_dir=data_dir / "GOAT",
        data_dir=data_dir / "test",
        max_events=dataset.max_events,
        fs=22050,
        discard_multi_event=True
    )

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, drop_last=True,
                                                  num_workers=4, pin_memory=True)

    early_stopping_count = 0

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(model_path / "my_checkpoints")

    # Store the model params in a json file in model_dir
    model_params = {
        'input_size (F)': int(dataset.n_classes),
        'batch_size': int(batch_size),
    }
    print(f"model_params: {model_params}")
    print(f"Saving model params in {model_path}")
    write_json(model_params, model_path / "params.json", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda available :', torch.cuda.is_available())

    # Define model components
    if baseline == "FRET":
        model = FretNet(dim_in=144, in_channels=6, frame_width=9, n_classes=dataset.n_classes, device=device)
        compute_feat = compute_hcqt
    elif baseline == "TabCNN":
        model = TabCNN(dim_in=192, n_classes=dataset.n_classes, device=device)
        compute_feat = compute_cqt
    else:
        model = None
    diffusion = DiffusionModel(model=model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print('\n batch_size: ', batch_size)
    print('\n input_size (F): ', int(dataset.n_classes))
    print('\n dataset len: ', len(dataset))
    print('\n epochs ', epochs)
    print('\n')

    model = model.to(device)
    print(all(p.is_cuda for p in model.parameters()))  # True if all params on GPU

    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Load last checkpoint
    checkpoint = ckpt_manager.load_last_checkpoint(model, optimizer, scheduler, device=device)
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
            model.train()
            for audio, token, prev_token in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}",
                                                 disable=True):
                audio = audio.to(model.device)
                token = token.to(model.device)

                # Compute audio features before training step
                features = compute_feat(audio)

                loss = model.train_step(optimizer=optimizer, batch=[token, features[..., :9]])

                train_loss += loss
                train_batches += 1

            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            print(f'Epoch {epoch + 1}: Loss: {avg_train_loss:.6f}\n')

            # Validation phase
            if (epoch + 1) % 1 == 0:
                total_val_loss, total_acc, total_samples = 0, 0, 0
                val_batches = 0
                model.eval()
                with torch.no_grad():
                    for audio, token, prev_token in tqdm(test_dataloader, desc=f"Validation Epoch {epoch + 1}",
                                                         disable=True):
                        audio = audio.to(model.device)
                        token = token.to(model.device)
                        features = compute_feat(audio)

                        loss, acc = model.val_step(batch=[token, features[..., :9]])

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
                    'model_state_dict': model.state_dict(),
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
                        'model_state_dict': model.state_dict(),
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

            filename = model_path / ('losses.json')
            save_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)
            filename = model_path / ('loss_plot.png')
            plot_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)
            # Update learning rate scheduler with validation loss
            scheduler.step()

    # Load best checkpoint
    best_checkpoint = ckpt_manager.load_best_checkpoint(model, device=device)
    if best_checkpoint:
        print(f"Loaded best model with metric: {best_checkpoint.get('best_val_loss', 0)}")

    losses_dict = {
        'best_val_loss': best_loss
    }
    filename = model_path / ('test_losses.txt')
    with open(filename, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {filename}")


    # Visualize the diffusion process
    model.eval()
    gt_chunks, pred_chunks = [], []
    val_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True)
                       
    with torch.no_grad():
        for audio, token, prev_token in tqdm(val_dataloader, desc=f"Test",
                                             disable=True):
            audio = audio.to(model.device)
            token = token.to(model.device)
            features = compute_feat(audio)

            logits = model(features[..., :9])

            # normalise both to integer IDs (B, T, 6) before storing
            gt_ids = token.argmax(dim=-1).cpu() if token.ndim == 4 else token.cpu()
            pred_ids = logits.argmax(
                dim=-1).cpu() if logits.ndim == 4 else logits.cpu()
            gt_chunks.append(gt_ids)
            pred_chunks.append(pred_ids)

        all_gt = torch.cat(gt_chunks, dim=0)  # (N, T, 6)
        all_pred = torch.cat(pred_chunks, dim=0)  # (N, T, 6)
        np.savez(model_path / "predictions", gt=all_gt, pred=all_pred)
        print(f"Predictions cached → {model_path}")

    avg = tab_metrics(all_gt, all_pred)
    out_path = model_path / f"metrics_Test.txt"
    print_tab_metrics(avg, save_path=str(out_path), prefix="Test set")
                        
    #print examples
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


def print_aux_metrics_table(metrics_dict, save_path, title="Auxiliary metrics", mode="w"):
    metric_col_width = max(len("Metric"), max(len(k) for k in metrics_dict.keys()))
    value_col_width = max(len("Value"), 12)

    line = f"+-{'-' * metric_col_width}-+-{'-' * value_col_width}-+\n"

    with open(save_path, mode, encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(line)
        f.write(f"| {'Metric'.ljust(metric_col_width)} | {'Value'.ljust(value_col_width)} |\n")
        f.write(line)

        for key, value in metrics_dict.items():
            value_str = f"{value:.6f}" if isinstance(value, float) else str(value)
            f.write(f"| {key.ljust(metric_col_width)} | {value_str.ljust(value_col_width)} |\n")

        f.write(line)
        f.write("\n")
        
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
    ROOT_DIR = files_dir / "Clean_GOAT_processed_0.1"

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    n_batches = 128
    epochs = 0#1000
    lr = 3e-4
    baselines = ["TabCNN", "FRET"]
    baseline = baselines[1]
    model_name = "_".join(
            ['Audio2Tab', baseline])
    model_path = script_dir.parent.parent / "TrainedModels" / (model_name)

    print(f"model_name: {model_name}")
    print(f"model_path: {model_path}")

    train_model(data_dir=ROOT_DIR,
                model_path=model_path,
                batch_size=n_batches,
                epochs=epochs,
                lr=lr,
                baseline=baseline
                )
