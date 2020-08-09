import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from motorcycle_dataset import MotorcycleDataset, TrainPipeline, TestPipeline
import os
import os.path as osp
import torch.nn as nn
from net import Net

if __name__ == "__main__":
    root_dir = "../data"
    checkpoint_dir = "../checkpoints"

    if not osp.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    train_dataset = MotorcycleDataset(
        root_dir, "train", transform=TrainPipeline)
    print(f"Training samples: {len(train_dataset)}")
    val_dataset = MotorcycleDataset(
        root_dir, "val", transform=TestPipeline)
    print(f"Validation samples: {len(val_dataset)}")

    batch_size = 4
    num_epochs = 10
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")

    start_training = False
    while not start_training:
        user_input = input("Do you want to start training? [Y/n] ")
        if user_input == 'n':
            exit(1)
        else:
            start_training = True

    classes = train_dataset.get_class_names()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
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
                print(f"Loss: {(running_loss / 500):.3f}")
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

        print(f"Accuracy: {(100 * correct / total):.3f}%")
        print(
            f"Precision: {(100 * true_positives / (true_positives + false_positives)):.3f}%")
        print(
            f"Recall: {(100 * true_positives / (true_positives + false_negatives)):.3f}%")
        print()

        checkpoint = f"./motorcycle_net_epoch{str(epoch + 1).zfill(2)}.pth"
        torch.save(net.state_dict(), osp.join(checkpoint_dir, checkpoint))

    print("Finished Training")
    print()
