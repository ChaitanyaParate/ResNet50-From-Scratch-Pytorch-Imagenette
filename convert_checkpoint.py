import torch
from model import ResNet

DEVICE = "cuda"

checkpoint = torch.load("my_checkpoint.pth.tar", map_location=DEVICE)



print("=> Checkpoint keys:", checkpoint.keys())

model = ResNet(num_classes=10).to(DEVICE)
print("=> Checkpoint state_dict keys:")
for k in checkpoint["state_dict"].keys():
    print(k)
    break  # remove this to see all if needed

print("\n=> Model state_dict keys:")
for k in model.state_dict().keys():
    print(k)
    break  # remove this to see all if needed

_ = model(torch.randn(2, 3, 224, 224).to(DEVICE))

model.load_state_dict(checkpoint["state_dict"])

torch.save(model.state_dict(), "ResNet.pth")
print("Converted to ResNet.pth successfully!")
