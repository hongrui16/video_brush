import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


from dataloader.dataset.dataset_WebVid10M import WebVid10M
from dataloader.dataset.dataset_SAV import SAV

    


def build_dataloader(dataset_name, **kwargs):
    num_workers = kwargs.get("num_workers", 4)
    train_batch_size = kwargs.get("train_batch_size", 4)
    val_batch_size = kwargs.get("val_batch_size", 4)
    image_finetune = kwargs.get("image_finetune", False)
    num_processes = kwargs.get("num_processes", 1)
    global_rank = kwargs.get("global_rank", 0)
    global_seed = kwargs.get("global_seed", 0)
    train_data = kwargs.get("train_data", {})

    if dataset_name == "WebVid10M":
        # Get the training dataset
        train_dataset = WebVid10M(**train_data, is_image=image_finetune)
        distributed_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_processes,
            rank=global_rank,
            shuffle=True,
            seed=global_seed,
        )
    
    elif dataset_name == "SAV":
        # Get the training dataset
        train_dataset = SAV(**train_data, is_image=image_finetune)
        distributed_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_processes,
            rank=global_rank,
            shuffle=True,
            seed=global_seed,
        )

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")# Path: video_inpainting/dataloader/build_dataloader.py



    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_dataloader, train_dataset