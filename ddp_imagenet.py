import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler

# Function to set up distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    

# Function to clean up after DDP training
def cleanup():
    dist.destroy_process_group()

# Define the model (e.g., ResNet18 from torchvision)
def create_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1000) # Change the output layer to have 1000 classes
    return model

def train(rank, world_size, data_dir, batch_size=32, epochs=10):
    setup(rank, world_size)

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset & DataLoader
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    # Set up the model, optimizer, and loss function
    torch.cuda.set_device(rank)
    model = create_model().to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle the dataset differently each epoch
        model.train()

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(rank), labels.to(rank)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(f'Epoch [{epoch}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    cleanup()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP Training Script")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training (default: 10)")

    args = parser.parse_args()

    # Get rank and world_size from environment variables set by SLURM
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    
    train(rank, world_size, args.data_dir, args.batch_size, args.epochs)

