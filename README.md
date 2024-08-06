# Prog-SR
### Experimental environments:
PyTorch 2.2.0 + Python3.10;

### Datasets
For training, we use DIV2K and Flickr2K datasets. To accelerate image loading during training, each training image is evenly divided into four
parts, with each part becoming a separate, relatively smaller training image. 

For testing, we use five standard benchmark datasets, including Set5, Set14, BSDS100, Urban100, and the validation set of DIV2K.

### Train
The Python source files to run training for α<sub>1</sub> ~ α<sub>7</sub> approaches are stored in the subdirectory “Train”.
The definition files of the network model are stored in the subdirectory “ops”.

### Test
The well-trained checkpionts is stored in the subdirectory “Checkpoints”.
The Python source files to run testing for α<sub>1</sub> ~ α<sub>7</sub> approaches are stored in the subdirectory “Test”.

Any questions, please contact us at swzhou@hnu.edu.cn.