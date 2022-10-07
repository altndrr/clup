"""Module containing utils functions about data."""

from torch.utils.data.dataloader import default_collate


def id_collate(batch):
    """Collate data from a dataset returning also the id of its samples."""
    updated_batch = []
    ids = []

    for b in batch:
        updated_batch.append(b[:-1])
        ids.append(b[-1])

    return ids, default_collate(updated_batch)
