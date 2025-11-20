import torch
from dataset import ImageNet
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None):
    print("=> Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])

    print("Checkpoint loaded successfully!")

def get_loaders(data_dir, transform, batch_size, num_workers, pin_memory):
    full_dataset = ImageNet(data_dir, transform=transform)
    

    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    return loader

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    best_acc = 0
    best_t = 0.5

    with torch.no_grad():
        
        num_correct = 0
        num_samples = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            output_vector = torch.softmax(model(x), dim=1)

            predicted_class = torch.argmax(output_vector, dim=1)

            num_correct += (predicted_class == y).sum()
            num_samples += predicted_class.size(0)
        
        acc = float(num_correct) / num_samples * 100
        print(f"Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc

    
    model.train()

    return best_acc, best_t
