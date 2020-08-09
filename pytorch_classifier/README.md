# Logs

## Classifier training log
```
> python train_classifier.py
# of Training samples: 6219
# of Validation samples: 778
Batch size: 4
# of epochs: 10

Epoch: 1
Loss: 0.673
Loss: 0.535
Loss: 0.338
Accuracy: 67.224%
Precision: 67.374%
Recall: 98.259%

Epoch: 2
Loss: 0.271
Loss: 0.200
Loss: 0.140
Accuracy: 86.504%
Precision: 83.442%
Recall: 99.420%

Epoch: 3
Loss: 0.104
Loss: 0.086
Loss: 0.084
Accuracy: 88.046%
Precision: 85.099%
Recall: 99.420%

Epoch: 4
Loss: 0.063
Loss: 0.064
Loss: 0.047
Accuracy: 87.789%
Precision: 84.590%
Recall: 99.807%

Epoch: 5
Loss: 0.047
Loss: 0.056
Loss: 0.026
Accuracy: 87.275%
Precision: 83.929%
Recall: 100.000%

Epoch: 6
Loss: 0.038
Loss: 0.033
Loss: 0.031
Accuracy: 98.329%
Precision: 99.219%
Recall: 98.259%

Epoch: 7
Loss: 0.040
Loss: 0.022
Loss: 0.023
Accuracy: 98.715%
Recall: 99.807%

Epoch: 8
Loss: 0.016
Loss: 0.022
Loss: 0.031
Recall: 100.000%

Epoch: 9
Loss: 0.017
Loss: 0.015
Accuracy: 99.486%
Precision: 99.232%
Recall: 100.000%
Epoch: 10
Loss: 0.017
Loss: 0.018
Loss: 0.012
Accuracy: 99.486%
Precision: 99.232%
Recall: 100.000%

Finished Training
```

## Classifier testing Log
```
> python test_classifier.py
Using .\models\motorcycle_net_epoch10.pth
# of Testing samples: 777
Accuracy: 99.871%
Precision: 100.000%
Recall: 99.834%
```