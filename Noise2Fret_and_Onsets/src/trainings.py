"""
Created on Tue Jun 23 2026

@author: Riccardo Simionato

"""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from CheckpointManager import DiffusionCheckpointManager
from U_NET_Token_Masked import TokenUNet
from EventCountHead import EventCountHead
from OnsetsHead import (OnsetHead, compute_onset_pos_weight, decode_onset_times,
                        onset_metrics, print_onset_metrics)
from Dataset import GOATFrameDataset
from DiffusionUtils import save_losses, plot_losses
from utils import write_json
import json
from DiffusionModel import DiffusionModel
from FeaturesExtractor import compute_audio_features
from tab_metrics import tab_metrics, print_tab_metrics
import numpy as np
import time
from contextlib import ExitStack
from ema import EMA

from count_loss_utils import compute_count_class_counts_from_dataloader, compute_inverse_freq_class_weights
#from length_bucketing import LengthBucketBatchSampler, event_counts_for_dataset, trimming_collate, build_bucket_boundaries, bucket_report, length_to_bucket
import torch.nn.functional as F

STAGE_ORDER = ("unet", "count", "onset")

def stage_weights_path(model_path, stage):
    d = model_path / "stage_weights"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stage}.pt"


def component_state(diffusion, stage):
    """CPU snapshot of the components this stage owns."""
    return {
        name: {k: v.detach().cpu().clone()
               for k, v in diffusion.component(name).state_dict().items()}
        for name in DiffusionModel.STAGE_COMPONENTS[stage]
    }


def load_component_state(diffusion, state, device):
    for name, sd in state.items():
        diffusion.component(name).load_state_dict(
            {k: v.to(device) for k, v in sd.items()}
        )


def save_stage_weights(model_path, stage, ema_state, raw_state, metric, epoch):
    torch.save(
        {"stage": stage, "ema": ema_state, "raw": raw_state,
         "metric": float(metric), "epoch": int(epoch)},
        stage_weights_path(model_path, stage),
    )


def load_stage_weights(model_path, stage, diffusion, device, which="ema"):
    """Load one stage's components into `diffusion`. Returns the blob or None."""
    path = stage_weights_path(model_path, stage)
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu")
    load_component_state(diffusion, blob[which], device)
    return blob


def seed_from_previous_stages(diffusion, model_path, device, current_stage):
    """Bring in every component trained by an earlier stage.

    Other stages contribute their EMA weights (what was measured and what you
    would deploy). The current stage, if it has been run before, contributes
    its raw weights, so re-running it continues rather than warm-starting from
    an average.
    """
    notes = []
    for st in STAGE_ORDER:
        which = "raw" if st == current_stage else "ema"
        blob = load_stage_weights(model_path, st, diffusion, device, which=which)
        if blob is not None:
            notes.append(f"{st}:{which}(epoch {blob['epoch']}, metric {blob['metric']:.6f})")
    return notes


def report_drift(before, after, stage, tol=0.0):
    """Warn if a component that should be frozen moved.

    A parameter delta means the freeze leaked. A buffer delta with parameters
    unchanged means a normalization layer is still in train mode and its
    running statistics drifted — which changes eval-time output even though
    the weights are identical.
    """
    owned = set(DiffusionModel.STAGE_COMPONENTS[stage])
    lines = []
    for name, (p_after, b_after) in after.items():
        p_before, b_before = before[name]
        dp, db = abs(p_after - p_before), abs(b_after - b_before)
        if name in owned:
            continue
        if dp > tol or db > tol:
            lines.append(f"  {name}: |Δparams|={dp:.6e}  |Δbuffers|={db:.6e}")
    if lines:
        print(f"⚠ frozen components drifted during the '{stage}' stage:")
        print("\n".join(lines))
    return not lines


def train_diffusion_model(data_dir, model_path, noise_steps, base_channels, inject_feature_dim, embed_dim, audio_emb,
                          batch_size, epochs=10, lr=1e-4, train_model=True,
                          train_count_head=False, train_onset_head=False, gt_onset_anneal_epochs=100,
                          onset_source="gt", time_bias_mode="soft_cumulative",
                          count_source_for_onset="gt", gt_count_anneal_epochs=0):
    """Train the diffusion model on a dataset.

    Stages are mutually exclusive — exactly one component gets gradients:
        train_count_head=True   -> count head only
        train_onset_head=True   -> onset head only
        neither                 -> U-Net + embeddings (both heads frozen)

    Recommended order: count -> onset -> unet. The U-Net stage conditions its
    cross-attention on onsets, so the onset head should already be usable
    before it starts; `gt_onset_anneal_epochs` linearly moves the conditioning
    from ground-truth onsets to predicted ones over that many epochs so the
    U-Net never sees a distribution at test time it was not trained on.
    """
    assert onset_source in ("pred", "gt"), onset_source
    assert not (train_count_head and train_onset_head), "train one head at a time"
    stage = "count" if train_count_head else ("onset" if train_onset_head else "unet")
    print(f"training stage: {stage}  |  onset conditioning: {onset_source}"
          f"  |  cross-attn bias: {time_bias_mode}")

    # Setup dataloader
    dataset = GOATFrameDataset(
        root_dir=data_dir,
        data_dir=data_dir / "train",
        random_crop_lengths=False
    )

    dataset_test = GOATFrameDataset(
        root_dir=data_dir,
        data_dir=data_dir / "test",
        max_events=dataset.max_events,
        random_crop_lengths=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda available :', torch.cuda.is_available())
    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    early_stopping_count = 0

    ZERO_EVENTS_IMPOSSIBLE = True  # class 0 never occurs in your data
    NUM_COUNT_CLASSES = dataset.max_events if ZERO_EVENTS_IMPOSSIBLE else dataset.max_events + 1
    count_smoothing = 0.0
    class_counts = compute_count_class_counts_from_dataloader(
            train_dataloader, num_classes=NUM_COUNT_CLASSES, zero_events_impossible=ZERO_EVENTS_IMPOSSIBLE)
    print("count class distribution (train):", class_counts.tolist())
    count_class_weights = compute_inverse_freq_class_weights(class_counts, scheme="sqrt_inv", max_ratio=5.0)
    print("count class weights:", count_class_weights.tolist())


    # onset frames are ~1-2% positives; without pos_weight the head converges
    # to "never an onset" and stays there
    onset_pos_weight = compute_onset_pos_weight(train_dataloader)
    print("onset pos_weight:", float(onset_pos_weight))

    # Define model components
    model = TokenUNet(in_channels=dataset.n_strings*embed_dim,
                      base_channels=base_channels,
                      audio_embed_dim=audio_emb,
                      inject_feature_dim=inject_feature_dim,
                      max_len=dataset.max_events,
                      dropout=0.1,
                      zero_pad_output=False,
                      time_bias_mode=time_bias_mode,
                      use_cross_attn=False
                      )

    count_head = EventCountHead(audio_ch=audio_emb, spectral_ch=base_channels, hidden=base_channels, max_events=NUM_COUNT_CLASSES)

    onset_head = OnsetHead(
        audio_embed_dim=audio_emb,
        inject_feature_dim=inject_feature_dim - 1,
        hidden=base_channels,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
    )

    # Store the model params in a json file in model_dir
    model_params = {
        'input_size (T)': int(dataset.max_events),
        'hidden_size': int(base_channels),
        'batch_size': int(batch_size),
        'inject_feature_size': int(inject_feature_dim),
        'embed_dim': embed_dim,
        'audio_emb': audio_emb,
        'spectral_ch': count_head.spectral_ch,
        'count_head_hidden': count_head.hidden,
        'vocab_size': len(dataset.vocab),
        'noise_steps': noise_steps,
        'onset_hidden': onset_head.hidden,
        'onset_frames': int(dataset.n_onset_frames),
        'onset_pos_weight': float(onset_pos_weight),
        'time_bias_mode': time_bias_mode,
    }

    print(f"model_params: {model_params}")
    print(f"Saving model params in {model_path}")
    write_json(model_params, model_path / "params.json", False)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print('\n batch_size: ', batch_size)
    print('\n hidden_size: ', base_channels)
    print('\n embed_dim: ', embed_dim)
    print('\n audio embed_dim: ', count_head.audio_ch)
    print('\n spectral_ch: ', count_head.spectral_ch)
    print('\n count event size: ', count_head.hidden)
    print('\n input_size (T): ', int(dataset.max_events))
    print('\n vocab size: ', len(dataset.vocab))
    print('\n inject_channels: ', base_channels)
    print('\n noise_steps: ', noise_steps)
    print('\n dataset len: ', len(dataset))
    print('\n epochs ', epochs)
    print('\n')


    model = model.to(device)
    print(all(p.is_cuda for p in model.parameters()))  # True if all params on GPU


    diffusion = DiffusionModel(model=model, count_head=count_head, onset_head=onset_head, noise_steps=noise_steps,
                               embed_dim=embed_dim, vocab_size=len(dataset.vocab), stage=stage,
                               zero_events_impossible=ZERO_EVENTS_IMPOSSIBLE,
                               count_smoothing=count_smoothing, count_class_weights=count_class_weights,
                               onset_pos_weight=onset_pos_weight,
                               gt_onset_prob=0.0 if onset_source == "pred" else 1.0,
                               force_gt_onsets=(onset_source == "gt"),
                               onset_count_source="count_head",
                               count_source_for_onset=count_source_for_onset,
                               gt_count_prob=1.0,
                               ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(diffusion.embeddings.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    optimizer_count_head = torch.optim.AdamW(
        list(count_head.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    # Define the scheduler
    scheduler_count_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_count_head, T_max=epochs
    )

    optimizer_onset_head = torch.optim.AdamW(
        list(onset_head.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    scheduler_onset_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_onset_head, T_max=epochs
    )

    # Initialize checkpoint manager — ONE DIRECTORY PER STAGE, so a later
    # stage's "best" can never overwrite an earlier stage's.
    ckpt_manager = DiffusionCheckpointManager(model_path / "my_checkpoints" / stage)

    # one EMA per component, keyed by the attribute name on DiffusionModel
    emas = {
        "model":      EMA(diffusion.model,      decay=0.999, warmup=True).to(device),
        "embeddings": EMA(diffusion.embeddings, decay=0.999, warmup=True).to(device),
        "count_head": EMA(diffusion.count_head, decay=0.999, warmup=True).to(device),
        "onset_head": EMA(diffusion.onset_head, decay=0.999, warmup=True).to(device),
    }
    # only the components this stage actually trains
    active_emas = [(name, emas[name], diffusion.component(name))
                   for name in DiffusionModel.STAGE_COMPONENTS[stage]]
    print("EMA tracked this stage:", [n for n, _, _ in active_emas])

    if train_model:
        # Load last checkpoint of THIS stage (optimizers, schedulers, EMA, epoch)
        checkpoint = ckpt_manager.load_last_checkpoint(diffusion, optimizer, optimizer_count_head, scheduler,
                                                       scheduler_count_head, device=device)
        start_epoch = 0
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
            best_monitored = checkpoint.get('best_monitored', float('inf'))
            print(f"Loaded best monitored metric for stage '{stage}': {best_monitored}")
            # an EMA is only meaningful for the stage that trained its component
            if checkpoint.get('ema_stage') == stage:
                for name, ema, _ in active_emas:
                    if f"ema_{name}" in checkpoint:
                        ema.load_state_dict(checkpoint[f"ema_{name}"])
        else:
            print("No checkpoint for this stage — seeding components from finished stages")
            best_monitored = float('inf')

        # Bring in every component an earlier stage already trained. This is
        # what replaces "resume from the last checkpoint of whatever ran
        # before": a stage inherits the best weights each other stage measured,
        # not whatever happened to be on disk last.
        notes = seed_from_previous_stages(diffusion, model_path, device, stage)
        print("seeded from stage weights: " + (", ".join(notes) if notes else "(nothing on disk)"))

        fp_stage_start = diffusion.component_fingerprint()

        train_losses, val_losses = [], []
        # Training loop
        for epoch in range(start_epoch, epochs):
            train_loss, val_loss, train_count_loss, train_onset_loss, train_batches = 0, 0, 0, 0, 0
            diffusion.train()
            # teacher forcing on ground-truth onsets, annealed to 0
            if stage == "unet" and onset_source == "pred":
                p_gt = max(0.0, 1.0 - epoch / max(gt_onset_anneal_epochs, 1))
                diffusion.set_gt_onset_prob(p_gt)
            # teacher forcing on the ground-truth count fed to the onset head
            if stage == "onset" and count_source_for_onset == "mix":
                diffusion.set_gt_count_prob(max(0.0, 1.0 - epoch / max(gt_count_anneal_epochs, 1)))

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}",
                                                          disable=True):
                audio = batch["audio"].to(diffusion.device)
                token = batch["tab_tokens"].to(diffusion.device)
                tab_mask = batch["tab_mask"].to(diffusion.device)

                onset_frames = batch["onset_frames"].to(diffusion.device)
                onset_frame_mask = batch["onset_frame_mask"].to(diffusion.device)
                onset_times = batch["onset_times"].to(diffusion.device)
                onset_times_mask = batch["onset_times_mask"].to(diffusion.device)

                # Compute audio features before training step
                features = compute_audio_features(audio, sr=16000)
                features = torch.cat(
                                [features["stft_mag"], features["spectral_flux"], features["brightness"]], dim=-1)

                loss, count_loss, onset_loss = diffusion.train_step(
                    optimizer=optimizer,
                    optimizer_count_head=optimizer_count_head,
                    optimizer_onset_head=optimizer_onset_head,
                    batch=[token, audio, features, tab_mask,
                           onset_frames, onset_frame_mask, onset_times, onset_times_mask])

                # right after train_step, which does the .step()
                for _name, ema, module in active_emas:
                    ema.update(module)

                train_loss += loss
                train_count_loss += count_loss
                train_onset_loss += onset_loss
                train_batches += 1

            avg_train_loss = train_loss / train_batches
            avg_train_count_loss = train_count_loss / train_batches
            avg_train_onset_loss = train_onset_loss / train_batches
            train_losses.append(avg_train_loss)

            # Validation phase
            if (epoch + 1) % 1 == 0:
                total_val_loss, total_acc, total_samples, total_val_count_loss, val_batches = 0, 0, 0, 0, 0
                total_val_onset_loss = 0
                onset_f1_num, onset_f1_den = 0.0, 0
                diffusion.eval()
                # only the active stage's EMAs are applied — the frozen
                # components already hold their own stage's averaged weights
                with torch.no_grad(), ExitStack() as ema_stack:
                    for _name, ema, module in active_emas:
                        ema_stack.enter_context(ema.average_parameters(module))
                    for batch in tqdm(test_dataloader, desc=f"Validation Epoch {epoch + 1}",
                                                                  disable=True):
                        audio = batch["audio"].to(diffusion.device)
                        token = batch["tab_tokens"].to(diffusion.device)
                        tab_mask = batch["tab_mask"].to(diffusion.device)

                        onset_frames = batch["onset_frames"].to(diffusion.device)
                        onset_frame_mask = batch["onset_frame_mask"].to(diffusion.device)
                        onset_times = batch["onset_times"].to(diffusion.device)
                        onset_times_mask = batch["onset_times_mask"].to(diffusion.device)

                        features = compute_audio_features(audio, sr=16000)
                        features = torch.cat(
                                        [features["stft_mag"], features["spectral_flux"], features["brightness"]], dim=-1)


                        loss, acc, count_loss, onset_loss = diffusion.val_step(
                            batch=[token, audio, features, tab_mask,
                                   onset_frames, onset_frame_mask, onset_times, onset_times_mask])


                        # BCE is not comparable across pos_weight settings — track F1
                        pred_t, pred_m = diffusion.predict_onsets(
                            audio, features, max_events=dataset.max_events)
                        if pred_t is not None:
                            om = onset_metrics(pred_t.cpu(), pred_m.cpu(),
                                               onset_times.cpu(), onset_times_mask.cpu(),
                                               window_sec=dataset.window_sec, tolerance_sec=0.05)
                            onset_f1_num += om["f1"]
                            onset_f1_den += 1

                        total_val_loss += loss
                        total_val_count_loss += count_loss
                        total_val_onset_loss += onset_loss
                        total_acc += acc
                        val_batches += 1

                    # Captured INSIDE the EMA context: these are the weights the
                    # numbers below were measured on, and the ones every later
                    # stage and the final evaluation will load.
                    ema_component_state = component_state(diffusion, stage)

                avg_val_loss = total_val_loss / val_batches
                avg_acc = total_acc / val_batches
                avg_count_loss = total_val_count_loss / val_batches
                avg_onset_loss = total_val_onset_loss / val_batches
                avg_onset_f1 = onset_f1_num / max(onset_f1_den, 1)
                val_losses.append(avg_val_loss)

                print(
                    f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Acc: {avg_acc:.6f}, Count loss: {avg_count_loss:.6f}, Onset loss: {avg_onset_loss:.6f}, Onset F1: {avg_onset_f1:.4f}')
                print(f'Learning Rate {optimizer.param_groups[0]["lr"]:.2e}')
                if stage == "unet":
                    print(f'p(ground-truth onset conditioning): {diffusion.gt_onset_prob:.3f}')

                # frozen components must be bit-identical; catch a leak on the
                # epoch it happens, not three stages later
                report_drift(fp_stage_start, diffusion.component_fingerprint(), stage)

                def build_state_dict():
                    return {
                        'epoch': epoch,
                        'stage': stage,
                        'model_state_dict': diffusion.model.state_dict(),
                        'embedding_state_dict': diffusion.embeddings.state_dict(),
                        'count_head_state_dict': diffusion.count_head.state_dict(),
                        'onset_head_state_dict': diffusion.onset_head.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'optimizer_count_head_dict': optimizer_count_head.state_dict(),
                        'optimizer_onset_head_dict': optimizer_onset_head.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scheduler_count_head_dict': scheduler_count_head.state_dict(),
                        'scheduler_onset_head_dict': scheduler_onset_head.state_dict(),
                        **{f"ema_{name}": ema.state_dict() for name, ema, _ in active_emas},
                        'ema_stage': stage,   # an EMA is only valid for its own stage
                        'train_loss': avg_train_loss,
                        'train_count_loss': avg_train_count_loss,
                        'train_onset_loss': avg_train_onset_loss,
                        'val_loss': avg_val_loss,
                        'val_acc': avg_acc,
                        'val_count_loss': avg_count_loss,
                        'val_onset_loss': avg_onset_loss,
                        'val_onset_f1': avg_onset_f1,
                        'best_monitored': best_monitored,
                    }

                # model selection follows whichever component is actually training.
                # One monitored quantity per stage, compared against its own
                # running best — the old code compared 1 - F1 against a BCE value.
                if stage == "count":
                    monitored, patience = avg_count_loss, 100
                elif stage == "onset":
                    # select on 1 - F1, not BCE: the loss keeps creeping down long
                    # after localization has stopped improving
                    monitored, patience = 1.0 - avg_onset_f1, 100
                else:
                    monitored, patience = avg_val_loss, 70

                if monitored < best_monitored:
                    best_monitored = monitored
                    ckpt_manager.save_checkpoint(build_state_dict(), is_best=True)
                    # the stage-scoped copy: only this stage's components
                    save_stage_weights(model_path, stage,
                                       ema_state=ema_component_state,
                                       raw_state=component_state(diffusion, stage),
                                       metric=monitored, epoch=epoch)
                    print(f"Epoch {epoch + 1}, Validation metric improved: {monitored:.6f}")
                    early_stopping_count = 0

                else:
                    early_stopping_count += 1
                    print(f"Epoch {epoch + 1}, Validation metric did not improve.")
                    print(f"early_stopping_count: {early_stopping_count}")
                    if early_stopping_count == patience:
                        print(f'No improvements over {patience} epochs -> stopping...')
                        break

                # Save last checkpoint
                ckpt_manager.save_last_checkpoint(build_state_dict())

            filename = model_path / ('losses.json')
            save_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)
            filename = model_path / ('loss_plot.png')
            plot_losses(train_losses=train_losses, val_losses=val_losses, filename=filename)
            # Update learning rate scheduler with validation loss
            if stage == "count":
                scheduler_count_head.step()
            elif stage == "onset":
                scheduler_onset_head.step()
            else:
                scheduler.step()

        ok = report_drift(fp_stage_start, diffusion.component_fingerprint(), stage)
        if ok:
            print(f"✓ stage '{stage}' left every other component untouched")

    # ── Load the best weights of EVERY stage ──────────────────────────────
    # Each stage contributes its own EMA-averaged best. No EMA context is
    # needed below: the averaged weights are already in the modules, so there
    # is no cross-stage EMA state to reload.
    loaded = {}
    for st in STAGE_ORDER:
        blob = load_stage_weights(model_path, st, diffusion, device, which="ema")
        if blob is not None:
            loaded[st] = blob
            print(f"Loaded best '{st}' weights: epoch {blob['epoch']}, metric {blob['metric']:.6f}")

    if loaded:
        losses_dict = {f"best_{st}_metric": blob["metric"] for st, blob in loaded.items()}
        losses_dict.update({f"best_{st}_epoch": blob["epoch"] for st, blob in loaded.items()})
        filename = model_path / ('test_losses.txt')
        with open(filename, 'w') as f:
            json.dump(losses_dict, f)
        print(f"Losses saved to {filename}")

    # Visualize the diffusion process
    diffusion.eval()
    gt_chunks, pred_chunks, mask_chunks, counts, counts_target = [], [], [], [], []
    onset_pred_t, onset_pred_m, onset_gt_t, onset_gt_m = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Test", disable=True):

            # start_time = time.time()
            audio = batch["audio"].to(diffusion.device)
            token = batch["tab_tokens"].to(diffusion.device)
            tab_mask = batch["tab_mask"].to(diffusion.device)

            features = compute_audio_features(audio, sr=16000)
            features = torch.cat(
                                [features["stft_mag"], features["spectral_flux"], features["brightness"]], dim=-1)


            onset_frames = batch["onset_frames"].to(diffusion.device)
            onset_times = batch["onset_times"].to(diffusion.device)
            onset_times_mask = batch["onset_times_mask"].to(diffusion.device)
            gt_cond = gt_onset_cond(onset_source, onset_frames, onset_times,
                                        onset_times_mask, tab_mask, dataset.max_events)
            predicted_indices, predicted_tab = visualize_samples(token, audio, features, tab_mask, diffusion,
                                                                     onset_cond=gt_cond)

            count_logits = diffusion.count_head(audio, features[..., :-1])
            pt, pm = diffusion.predict_onsets(audio, features, max_events=dataset.max_events)
            onset_pred_t.append(pt.cpu())
            onset_pred_m.append(pm.cpu())
            onset_gt_t.append(batch["onset_times"])
            onset_gt_m.append(batch["onset_times_mask"])

            gt_chunk = token.view(token.shape[0], token.shape[1] // 6, -1).cpu()
            gt_chunks.append(gt_chunk)
            pred_chunk = predicted_indices.cpu()
            pred_chunks.append(pred_chunk)
            mask = ~tab_mask.view(tab_mask.shape[0], tab_mask.shape[1] // 6, 6).cpu()
            mask_chunks.append(mask)

            count = torch.argmax(count_logits, dim=-1) + 1 if ZERO_EVENTS_IMPOSSIBLE else torch.argmax(count_logits, dim=-1)
            counts.append(count)
            n_real = (~tab_mask).view(tab_mask.shape[0], -1).sum(dim=1) // 6  # (B,)
            counts_target.append(n_real.long())

        all_gt = torch.cat(gt_chunks, dim=0)  # (N, T, 6)
        all_pred = torch.cat(pred_chunks, dim=0)  # (N, T, 6)
        all_event_masks = torch.cat(mask_chunks, dim=0)  # (N, T, 6)
        counts = torch.cat(counts, dim=0)  # (N, T, 6)
        counts_target = torch.cat(counts_target, dim=0)  # (N, T, 6)

        np.savez(model_path / "predictions",
                     gt=all_gt.numpy(), pred=all_pred.numpy(),
                     event_mask=all_event_masks[..., 0].numpy(),
                     onset_gt_t=torch.cat(onset_gt_t).numpy(),
                     onset_gt_m=torch.cat(onset_gt_m).numpy(),
                     onset_pred_t=torch.cat(onset_pred_t).numpy(),
                     onset_pred_m=torch.cat(onset_pred_m).numpy(),
                     counts=counts.cpu().numpy(),
                     counts_target=counts_target.cpu().numpy(),
                     window_sec=np.float64(dataset.window_sec))
        print(f"Predictions cached → {model_path}")

    to_metric = lambda x: torch.where(x <= 1, torch.zeros_like(x), x - 1)
    avg = tab_metrics(to_metric(all_gt[all_event_masks]).reshape(-1, 1, 6),
                          to_metric(all_pred[all_event_masks]).reshape(-1, 1, 6))
    out_path = model_path / f"metrics_Test.txt"
    print_tab_metrics(avg, save_path=str(out_path), prefix="Test set")

    om = onset_metrics(torch.cat(onset_pred_t), torch.cat(onset_pred_m),
                           torch.cat(onset_gt_t), torch.cat(onset_gt_m),
                           window_sec=dataset.window_sec, tolerance_sec=0.05)
    print_onset_metrics(om, save_path=str(model_path / "metrics_onset_Test.txt"), prefix="Test set")

    all_pred, all_gt = vectors_to_text_token(all_pred, all_gt, dataset, ~all_event_masks)
    output_path = model_path / "predictions_TEST.txt"
    print_results(all_gt, all_pred, output_path)

    output_path = model_path / "predictionsCounts_TEST.txt"
    print_results_count(counts_target, counts, output_path, n_classes=NUM_COUNT_CLASSES, count_offset=1 if ZERO_EVENTS_IMPOSSIBLE else 0)

    return 42


def vectors_to_text_token(predicted_indices, token, dataset, tab_mask=None):
    """
    predicted_indices : (B, T, 6)  argmax class indices from diffusion
    token             : (B, T, 6)  ground truth class indices  ← argmax of one-hot target
    tab_mask:           (B, T, 6), True for non-PAD string positions

    Returns:
        predicted_decoded : list[list[list[str]]]  B × T × n_active_strings
        token_decoded     : list[list[list[str]]]  B × T × n_active_strings
    """
    if isinstance(predicted_indices, torch.Tensor):
        predicted_indices = predicted_indices.cpu()
    if isinstance(token, torch.Tensor):
        token = token.cpu()
    if isinstance(tab_mask, torch.Tensor):
        tab_mask = tab_mask.detach().cpu()

    b, t, s = predicted_indices.shape
    token = token.view(b, t, s)

    predicted_decoded = [
        dataset.decode_token_ids(predicted_indices[i].reshape(-1))
        for i in range(predicted_indices.shape[0])
    ]
    token_decoded = [
        dataset.decode_token_ids(token[i].reshape(-1))
        for i in range(token.shape[0])
    ]

    if tab_mask is not None:
        event_mask = ~tab_mask.view(b, t, s).any(dim=-1)

        predicted_decoded = [
            [
                event
                for event, keep in zip(
                    predicted_decoded[i],
                    event_mask[i].tolist()
                )
                if keep
            ]
            for i in range(b)
        ]

        token_decoded = [
            [
                event
                for event, keep in zip(
                    token_decoded[i],
                    event_mask[i].tolist()
                )
                if keep
            ]
            for i in range(b)
        ]

    return predicted_decoded, token_decoded


def gt_onset_cond(onset_source, onset_frames, onset_times, onset_times_mask,
                  tab_mask, max_events):
    """Ground-truth conditioning dict, or None to let the onset head decide."""
    if onset_source != "gt":
        return None
    n_real = (~tab_mask).view(tab_mask.shape[0], -1).sum(dim=1) // 6
    return {"prob": onset_frames.float(), "n_events": n_real.long(),
            "times": onset_times, "valid": onset_times_mask.bool(),
            "n_slots": max_events}

def visualize_samples(inputs, audio, cond, tab_mask, diffusion, onset_cond=None):
    """Visualize samples from the diffusion model.

    onset_cond: pass a ground-truth conditioning dict here to sample under the
    oracle. Left as None, sample() calls the onset head itself.
    """
    z = diffusion.sample(input=inputs, audio=audio, cond=cond, tab_mask=tab_mask,
                         num_steps=diffusion.noise_steps, onset_cond=onset_cond)
    tab = diffusion.decode(z)
    tab_indices = tab.argmax(dim=-1)    # (B, T, 6)  — integer class per string

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


def print_results_count(target_counts, predicted_counts, output_path,
                        n_classes, count_offset=1):
    """
    Print target and predicted event counts for each frame.

    Parameters
    ----------
    target_counts : Sequence[int]
        Ground-truth number of events per frame, in REAL count space
        (i.e. 1..max_count when zero events are impossible).
    predicted_counts : Sequence[int]
        Predicted number of events per frame, also in REAL count space.
    output_path : str
        Path of the output text file.
    n_classes : int
        Number of confusion-matrix classes = number of count head outputs.
        With zero_events_impossible=True this is max_count (NOT max_count+1).
    count_offset : int
        Real count of class index 0. Pass 1 when zero_events_impossible=True
        (class 0 means "1 event"), 0 otherwise. Both target_counts and
        predicted_counts get this subtracted before indexing, which is what
        keeps a true count of `max_count` from running off the end of the
        matrix, and stops a permanently-empty "0 events" row appearing.
    """

    n_frames = max(len(target_counts), len(predicted_counts))
    n_mismatch = 0
    examples = []
    valid_pairs = []
    n_out_of_range = 0

    for idx in range(n_frames):
        target = int(target_counts[idx]) if idx < len(target_counts) else None
        predicted = int(predicted_counts[idx]) if idx < len(predicted_counts) else None

        is_mismatch = target != predicted
        if is_mismatch:
            n_mismatch += 1

        examples.append((idx, target, predicted, is_mismatch))

        if target is not None and predicted is not None:
            # real count -> class index
            t_idx = target - count_offset
            p_idx = predicted - count_offset
            if 0 <= t_idx < n_classes and 0 <= p_idx < n_classes:
                valid_pairs.append((t_idx, p_idx))
            else:
                # should never fire; if it does, n_classes or count_offset
                # disagrees with how the counts were produced upstream
                n_out_of_range += 1

    if valid_pairs:
        confusion_mat = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
        for t_idx, p_idx in valid_pairs:
            confusion_mat[t_idx][p_idx] += 1

        correct = sum(confusion_mat[i][i] for i in range(n_classes))
        total = sum(sum(row) for row in confusion_mat)
        accuracy = correct / total if total > 0 else 0.0

        # off-by-one rate: for an ordinal target this matters more than
        # raw accuracy, since adjacent confusions are the expected failure
        near = sum(
            confusion_mat[i][j]
            for i in range(n_classes)
            for j in range(n_classes)
            if abs(i - j) <= 1
        )
        mae = sum(
            abs(i - j) * confusion_mat[i][j]
            for i in range(n_classes)
            for j in range(n_classes)
        ) / total if total > 0 else 0.0
    else:
        confusion_mat = None
        accuracy = 0.0
        near = 0
        mae = 0.0
        total = 0
        correct = 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")

        if n_mismatch == 0:
            f.write(f"✓ All {n_frames} frames match.\n")
        else:
            f.write(f"⚠ {n_mismatch}/{n_frames} frames have mismatches.\n")

        if n_out_of_range:
            f.write(f"⚠ {n_out_of_range} pairs fell outside "
                    f"[{count_offset}, {count_offset + n_classes - 1}] and were skipped.\n")

        f.write("=" * 60 + "\n\n")

        if confusion_mat is not None:
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {accuracy:.4f} ({int(accuracy * 100)}%)\n")
            f.write(f"Within +/-1: {near / total:.4f} ({int(near / total * 100)}%)\n")
            f.write(f"Mean absolute error: {mae:.4f}\n")
            f.write(f"Total frames: {total}\n")
            f.write(f"Correct: {correct}\n")
            f.write(f"Incorrect: {total - correct}\n\n")

            # axes labelled with REAL counts, not class indices
            f.write("Target \\ Pred  ")
            for j in range(n_classes):
                f.write(f"{j + count_offset:6d}")
            f.write("\n")

            for i in range(n_classes):
                f.write(f"{i + count_offset:13d} ")
                for j in range(n_classes):
                    f.write(f"{confusion_mat[i][j]:6d}")
                f.write("\n")

            f.write("\n")

            f.write("PER-CLASS STATISTICS\n")
            f.write("-" * 40 + "\n")
            for i in range(n_classes):
                row_sum = sum(confusion_mat[i])
                col_sum = sum(confusion_mat[j][i] for j in range(n_classes))
                correct_class = confusion_mat[i][i]

                recall = correct_class / row_sum if row_sum > 0 else 0.0
                precision = correct_class / col_sum if col_sum > 0 else 0.0

                # label by real count so this lines up with the matrix axes
                f.write(f"Count {i + count_offset}: support={row_sum:4d}, TP={correct_class:4d}, "
                        f"precision={precision:.3f}, recall={recall:.3f}\n")

            f.write("\n" + "=" * 60 + "\n\n")

        for idx, target, predicted, is_mismatch in examples:
            f.write(f"Frame {idx}\n")
            f.write("-" * 40 + "\n")

            if is_mismatch:
                if target is None:
                    f.write("⚠ MISMATCH: missing target value\n")
                elif predicted is None:
                    f.write("⚠ MISMATCH: missing predicted value\n")
                else:
                    difference = predicted - target
                    f.write("⚠ MISMATCH\n")
                    f.write(f"  target     : {target}\n")
                    f.write(f"  predicted  : {predicted}\n")
                    f.write(f"  difference : {difference:+d} (abs {abs(difference)})\n")
            else:
                f.write(f"✓ match: {target}\n")

            f.write("\n")
    return


# Example usage
if __name__ == "__main__":

    import os
    from pathlib import Path
    from utils import find_folder_upward

    current_dir = Path(os.getcwd())
    print(f"current_dir: {current_dir}")
    files_dir = find_folder_upward(folder_name="Files", start_path=current_dir)


    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    n_batches = 128
    embed_dim = 32
    audio_emb = 64
    hidden_dims = [64]
    noise_steps = 20#100#500
    #noise_steps = 1
    epochs = 1000
    #epochs = 1
    lr = 3e-4
    inject_feature_dim = 515

    model_type = "Unet"
    feat = "alls"
    addtional_name = "_full_vocab_20_noCA"
    ROOT_DIR = files_dir / "Clean_GOAT_processed_1"
    #ROOT_DIR = files_dir / "Clean_GuitarSet_processed_1"

    for hidden_dim in hidden_dims:

        model_name = "_".join(
            ["H", str(hidden_dim), "E", str(embed_dim)])
        model_path = script_dir.parent.parent / "TrainedModels" / (model_name + addtional_name)

        print(f"model_name: {model_name}")
        print(f"model_path: {model_path}")
        train_diffusion_model(data_dir=ROOT_DIR,
                              model_path=model_path,
                              noise_steps=noise_steps,
                              base_channels=hidden_dim,
                              inject_feature_dim=inject_feature_dim,
                              embed_dim=embed_dim,
                              audio_emb=audio_emb,
                              batch_size=n_batches,
                              epochs=epochs,
                              onset_source="pred",
                              count_source_for_onset="pred",
                              gt_onset_anneal_epochs=0,#100,
                              lr=lr,
                              train_model=True,
                              train_count_head=True,
                              train_onset_head=False
                              )

        train_diffusion_model(data_dir=ROOT_DIR,
                              model_path=model_path,
                              noise_steps=noise_steps,
                              base_channels=hidden_dim,
                              inject_feature_dim=inject_feature_dim,
                              embed_dim=embed_dim,
                              audio_emb=audio_emb,
                              batch_size=n_batches,
                              epochs=epochs,
                              onset_source="pred",
                              count_source_for_onset="pred",
                              gt_onset_anneal_epochs=0,#100,
                              lr=lr,
                              train_model=True,
                              train_count_head=False,
                              train_onset_head=True
                              )
        train_diffusion_model(data_dir=ROOT_DIR,
                              model_path=model_path,
                              noise_steps=noise_steps,
                              base_channels=hidden_dim,
                              inject_feature_dim=inject_feature_dim,
                              embed_dim=embed_dim,
                              audio_emb=audio_emb,
                              batch_size=n_batches,
                              epochs=epochs,
                              onset_source="pred",#"pred",
                              count_source_for_onset="pred",#"pred",
                              gt_onset_anneal_epochs=0,#0
                              lr=lr,
                              train_model=True,
                              train_count_head=False,
                              train_onset_head=False
                              )