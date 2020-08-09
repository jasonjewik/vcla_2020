import torch
from torch.utils.data import DataLoader
from net import Net
from motorcycle_dataset import MotorcycleDataset, TestPipeline, show_label_batch

if __name__ == "__main__":
    net = Net()
    checkpoint_file = '..\\checkpoints\\motorcycle_net_epoch10.pth'
    net.load_state_dict(torch.load(checkpoint_file))
    print('Using', checkpoint_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    root_dir = "..\\data"
    test_dataset = MotorcycleDataset(root_dir, "test", transform=TestPipeline)
    classes = test_dataset.get_class_names()
    test_loader = DataLoader(test_dataset, batch_size=4,
                             shuffle=True, num_workers=0)
    print(f"Testing samples: {len(test_dataset)}")

    correct, total = 0, 0
    true_positives, false_positives = 0, 0
    false_negatives = 0

    with torch.no_grad():
        for data in test_loader:
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
