# Siamese network for classifying unseen classes

## Questions for the experiments
1. What if one train a siamese network on 0-5 MNIST classes,
select representative images for 6-9 classes and test 6-9 samples?

2. If metrics are not too good how many samples needed to improve the results?

## Steps
### Defining regular model results
- train 3 conv layers with one layer for fine-tuning on 0-5 classes
- test metrics on 0-5 classes, 6-9 classes
- train 3 conv layers with one layer for fine-tuning on 0-9 classes
- test metrics on 0-5 classes, 6-9 classes

### Experimenting with siamese networks
- train 3 conv layers for siamese networks on 0-5 classes
- test metrics on 0-5 classes, 6-9 classes
- continue training on 6-9 samples (from train dataset) until equal metrics achieved

## Configurations
### Model structure
- Conv2D -

### Running settings
- loss
- optimizer 
- epochs
- batch size

## Results
### Defining regular model results
- regular model 0-5 train samples
  - 0-5 classes test metrics - TBU
  - 6-9 classes test metrics - TBU
- regular model 0-9 train classes
  - 0-5 classes test metrics - TBU
  - 6-9 classes test metrics - TBU

### Experimenting with siamese networks
- siamese network 0-5 train samples
  - 0-5 classes test metrics - TBU
  - 6-9 classes test metrics - TBU
- continue training 
  - 0-5 classes test metrics - TBU, epochs - ..., samples - ...
  - 6-9 classes test metrics - TBU, epochs - ..., samples - ...
