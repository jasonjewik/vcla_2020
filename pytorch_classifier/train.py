import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from motorcycle_dataset import MotorcycleDataset, TrainPipeline, TestPipeline


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    root_dir = "C:\\Users\\jewik\\GitRepos\\pytorch_classifier\\data"

    train_dataset = MotorcycleDataset(
        root_dir, "train", transform=TrainPipeline)
    val_dataset = MotorcycleDataset(
        root_dir, "val", transform=TestPipeline)

    classes = train_dataset.get_class_names()
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        print(f"Epoch: {epoch + 1}")

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs = data["image"].to(device)
            labels = data["label"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                # loss
                print(f"Loss: {running_loss / 500 }")
                running_loss = 0.0

        # print validation data metrics every epoch
        correct, total = 0, 0
        true_positives, false_positives = 0, 0
        false_negatives = 0

        with torch.no_grad():
            for data in val_loader:
                inputs = data["image"].to(device)
                labels = data["label"].to(device)
                outputs = net(inputs)
                predicted = torch.max(outputs.data, 1)[1]

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for guess, truth in zip(predicted, labels):
                    g = guess.item()
                    t = truth.item()

                    if (g == 1 and t == 1):
                        true_positives += 1
                    if (g == 1 and t == 0):
                        false_positives += 1
                    if (g == 0 and t == 1):
                        false_negatives += 1

        print(f"Accuracy: {100 * correct / total}")
        print(
            f"Precision: {100 * true_positives / (true_positives + false_positives)}")
        print(
            f"Recall: {100 * true_positives / (true_positives + false_negatives)}")
        print()

        PATH = f"./motorcycle_net_epoch{epoch + 1}.pth"
        torch.save(net.state_dict(), PATH)

    print("Finished Training")
