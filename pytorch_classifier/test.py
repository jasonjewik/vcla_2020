import torch
from torch.utils.data import DataLoader
from train_classifier import Net
from motorcycle_dataset import MotorcycleDataset, TestPipeline, show_label_batch

net = Net()
PATH = './motorcycle_net_epoch10.pth'
net.load_state_dict(torch.load(PATH))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

root_dir = "C:\\Users\\jewik\\GitRepos\\pytorch_classifier\\data"
test_dataset = MotorcycleDataset(root_dir, "test", transform=TestPipeline)
classes = test_dataset.get_class_names()
test_loader = DataLoader(test_dataset, batch_size=4,
                         shuffle=True, num_workers=0)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs = data["image"].to(device)
        labels = data["label"].to(device)
        outputs = net(inputs)
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}")
