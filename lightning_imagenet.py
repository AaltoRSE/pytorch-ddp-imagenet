import os
from argparse import ArgumentParser
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models, transforms
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Accuracy, AUROC

# Define the Lightning module
class ImageClassificationModule(pl.LightningModule):
    def __init__(self, data_dir, batch_size=32, lr=1e-3, num_classes=1000):
        super(ImageClassificationModule, self).__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr

        # Model definition
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # Adjust final layer

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_auroc = AUROC(task='multiclass', num_classes=num_classes)
        self.train_accuracy_top1 = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.train_accuracy_top5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        
        # Calculate loss
        loss = self.criterion(outputs, labels)
        
        # Update metrics
        self.train_auroc.update(outputs, labels)
        self.train_accuracy_top1.update(outputs, labels)
        self.train_accuracy_top5.update(outputs, labels)
        
        # Log loss
        # self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        print(loss)

        return loss

    def on_train_epoch_end(self):
        # Compute and log epoch-level metrics
        self.log('train_auroc', self.train_auroc.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy_top1', self.train_accuracy_top1.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy_top5', self.train_accuracy_top5.compute(), on_epoch=True, prog_bar=True, logger=True)
        
        # Reset metrics for the next epoch
        self.train_auroc.reset()
        self.train_accuracy_top1.reset()
        self.train_accuracy_top5.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        # Data transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Dataset & DataLoader with DistributedSampler
        dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=8, pin_memory=True)
        return dataloader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Set up PyTorch Lightning model
    model = ImageClassificationModule(data_dir=args.data_dir, batch_size=args.batch_size, lr=args.lr)
    print("start training...")
    print(torch.cuda.device_count())
    # Set up the trainer with DDP strategy
    trainer = Trainer(
        max_epochs=args.epochs,
        strategy=DDPStrategy(find_unused_parameters=False),
        devices=torch.cuda.device_count(),
        logger=CSVLogger("logs/"),
        accelerator="gpu",
    )

    # Start training
    trainer.fit(model)
