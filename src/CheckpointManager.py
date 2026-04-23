import os
import torch
import shutil
from pathlib import Path

class DiffusionCheckpointManager:
    """
    PyTorch model checkpoint management utility.
    Handles saving and loading of both latest and best model checkpoints.
    """

    def __init__(self, checkpoint_dir="checkpoints"):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir (str): Directory to store checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_path = self.checkpoint_dir / "best_checkpoint.pth"
        self.last_path = self.checkpoint_dir / "last_checkpoint.pth"

    def save_checkpoint(self, state_dict, is_best=False, filename=None):
        """
        Save model checkpoint.

        Args:
            state_dict (dict): Model state dictionary containing:
                - 'epoch': Current epoch number
                - 'model_state_dict': Model weights
                - 'optimizer_state_dict': Optimizer state
                - 'scheduler_state_dict': Learning rate scheduler state (optional)
                - 'best_metric': Best validation metric value (optional)
                - Any other information to save
            is_best (bool): Whether this checkpoint has the best metric so far
            filename (str, optional): Custom filename for checkpoint
        """
        # Save the latest checkpoint
        if filename is None:
            filename = self.last_path

        torch.save(state_dict, filename)
        print(f"Checkpoint saved to {filename}")

        # If this is the best model, save a copy as the best checkpoint
        if is_best:
            self.save_best_checkpoint(filename)

    def save_last_checkpoint(self, state_dict):
        """
        Save the latest checkpoint.

        Args:
            state_dict (dict): Model state dictionary
        """
        self.save_checkpoint(state_dict, filename=self.last_path)
        print(f"Last checkpoint saved to {self.last_path}")

    def save_best_checkpoint(self, src_path=None):
        """
        Save the best checkpoint.

        Args:
            src_path (str or Path, optional): Source checkpoint path to copy from.
                If None, uses the last saved checkpoint.
        """
        src_path = src_path or self.last_path
        shutil.copyfile(src_path, self.best_path)
        print(f"Best checkpoint saved to {self.best_path}")

    def load_checkpoint(self, filepath, diffusion, optimizer=None, scheduler=None, device='cpu'):
        """
        Load a checkpoint.

        Args:
            filepath (str or Path): Path to the checkpoint file
            diffusion (nn.Module): diffusion to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            device (str): Device to load the model to ('cpu', 'cuda', etc.)

        Returns:
            dict: Checkpoint contents
        """
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}")
            return None

        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        print("Checkpoint keys:", checkpoint.keys())

        # Load model state
        if 'model_state_dict' in checkpoint:
            diffusion.model.load_state_dict(checkpoint['model_state_dict'])
            diffusion.encoder.load_state_dict(checkpoint['time_embedding_state_dict'])
            diffusion.embeddings.load_state_dict(checkpoint['embedding_state_dict'])
            print("Weights loaded successfully!")
        else:
            print("No 'model_state_dict' found in checkpoint")

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint

    def load_last_checkpoint(self, model, optimizer=None, scheduler=None, device='cpu'):
        """
        Load the latest checkpoint.

        Args:
            model (nn.Module): Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            device (str): Device to load the model to

        Returns:
            dict: Checkpoint contents or None if no checkpoint found
        """
        return self.load_checkpoint(self.last_path, model, optimizer, scheduler, device)

    def load_best_checkpoint(self, model, optimizer=None, scheduler=None, device='cpu'):
        """
        Load the best checkpoint.

        Args:
            model (nn.Module): Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            device (str): Device to load the model to

        Returns:
            dict: Checkpoint contents or None if no checkpoint found
        """
        return self.load_checkpoint(self.best_path, model, optimizer, scheduler, device)

    def checkpoint_exists(self, checkpoint_type="last"):
        """
        Check if a checkpoint exists.

        Args:
            checkpoint_type (str): Type of checkpoint ('last' or 'best')

        Returns:
            bool: True if checkpoint exists, False otherwise
        """
        if checkpoint_type.lower() == "last":
            return os.path.exists(self.last_path)
        elif checkpoint_type.lower() == "best":
            return os.path.exists(self.best_path)
        else:
            raise ValueError("checkpoint_type must be 'last' or 'best'")

class CheckpointManager:
    """
    PyTorch model checkpoint management utility.
    Handles saving and loading of both latest and best model checkpoints.
    """

    def __init__(self, checkpoint_dir="checkpoints"):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir (str): Directory to store checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_path = self.checkpoint_dir / "best_checkpoint.pth"
        self.last_path = self.checkpoint_dir / "last_checkpoint.pth"

    def save_checkpoint(self, state_dict, is_best=False, filename=None):
        """
        Save model checkpoint.

        Args:
            state_dict (dict): Model state dictionary containing:
                - 'epoch': Current epoch number
                - 'model_state_dict': Model weights
                - 'optimizer_state_dict': Optimizer state
                - 'scheduler_state_dict': Learning rate scheduler state (optional)
                - 'best_metric': Best validation metric value (optional)
                - Any other information to save
            is_best (bool): Whether this checkpoint has the best metric so far
            filename (str, optional): Custom filename for checkpoint
        """
        # Save the latest checkpoint
        if filename is None:
            filename = self.last_path

        torch.save(state_dict, filename)
        print(f"Checkpoint saved to {filename}")

        # If this is the best model, save a copy as the best checkpoint
        if is_best:
            self.save_best_checkpoint(filename)

    def save_last_checkpoint(self, state_dict):
        """
        Save the latest checkpoint.

        Args:
            state_dict (dict): Model state dictionary
        """
        self.save_checkpoint(state_dict, filename=self.last_path)
        print(f"Last checkpoint saved to {self.last_path}")

    def save_best_checkpoint(self, src_path=None):
        """
        Save the best checkpoint.

        Args:
            src_path (str or Path, optional): Source checkpoint path to copy from.
                If None, uses the last saved checkpoint.
        """
        src_path = src_path or self.last_path
        shutil.copyfile(src_path, self.best_path)
        print(f"Best checkpoint saved to {self.best_path}")

    def load_checkpoint(self, filepath, model, optimizer=None, scheduler=None, device='cpu'):
        """
        Load a checkpoint.

        Args:
            filepath (str or Path): Path to the checkpoint file
            model (nn.Module): model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            device (str): Device to load the model to ('cpu', 'cuda', etc.)

        Returns:
            dict: Checkpoint contents
        """
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}")
            return None

        print(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        print("Checkpoint keys:", checkpoint.keys())

        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Weights loaded successfully!")
        else:
            print("No 'model_state_dict' found in checkpoint")

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint

    def load_last_checkpoint(self, model, optimizer=None, scheduler=None, device='cpu'):
        """
        Load the latest checkpoint.

        Args:
            model (nn.Module): Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            device (str): Device to load the model to

        Returns:
            dict: Checkpoint contents or None if no checkpoint found
        """
        return self.load_checkpoint(self.last_path, model, optimizer, scheduler, device)

    def load_best_checkpoint(self, model, optimizer=None, scheduler=None, device='cpu'):
        """
        Load the best checkpoint.

        Args:
            model (nn.Module): Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            device (str): Device to load the model to

        Returns:
            dict: Checkpoint contents or None if no checkpoint found
        """
        return self.load_checkpoint(self.best_path, model, optimizer, scheduler, device)

    def checkpoint_exists(self, checkpoint_type="last"):
        """
        Check if a checkpoint exists.

        Args:
            checkpoint_type (str): Type of checkpoint ('last' or 'best')

        Returns:
            bool: True if checkpoint exists, False otherwise
        """
        if checkpoint_type.lower() == "last":
            return os.path.exists(self.last_path)
        elif checkpoint_type.lower() == "best":
            return os.path.exists(self.best_path)
        else:
            raise ValueError("checkpoint_type must be 'last' or 'best'")