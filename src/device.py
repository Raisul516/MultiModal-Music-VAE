import torch


def get_device():
    """Return a torch.device: CUDA if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch, device):
    """Move any torch.Tensor values in a batch dict to `device` in-place.

    Non-tensor values are left unchanged.
    """
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch
