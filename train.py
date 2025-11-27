import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import ResNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
)

LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 40
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
TRAIN_DATASET_DIR = "imagenette2/train"
Val_DATASET_DIR = "imagenette2/val"
losses = []

def train_fn(loader, model, optimizer, loss_fn, scaler,scheduler):
    loop = tqdm(loader)
    model.train()
    Total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        with torch.autocast(device_type='cuda',enabled=False):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        Total_loss += loss.item()

        losses.append(loss.item())


        loop.set_postfix(loss=loss.item())
    

    avg_loss = Total_loss / len(loader)
    scheduler.step(avg_loss)
    return avg_loss



def main():
    best_accu = 0
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)

    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ])

    model = ResNet(num_classes=10).to(DEVICE)
    _ = model(dummy)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_loader = get_loaders(
        TRAIN_DATASET_DIR, train_transform, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )
    val_loader = get_loaders(
        Val_DATASET_DIR, val_transform, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )

    scaler = torch.amp.GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    #load_checkpoint("my_checkpoint.pth.tar", model, optimizer)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler)
        acc , t = check_accuracy(val_loader, model)

        if acc > best_accu:

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint)
            
            best_accu = acc
            print(f"\n✅ Best threshold = {t:.1f}, Accuracy = {best_accu:.2f}%\n")
        
        



if __name__ == "__main__":
    main()
