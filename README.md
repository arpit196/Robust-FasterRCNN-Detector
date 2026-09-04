# Robust Faster R-CNN LBIN normalization for domain and noise robust detector.
## Overview
This is an implementation of the Faster Robust R-CNN object detection model in TensorFlow 2.10 with Keras, using Python 3.10. 

> **Generalizable object detection under severe environmental noise, glare, and contrast shifts without retraining.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch / TF2](https://img.shields.io/badge/Framework-PyTorch%20%7C%20TensorFlow2-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Standard object detectors (like baseline Faster R-CNN with Batch Normalization) fail catastrophically under domain shifts—such as sudden contrast fluctuations, camera glare, or atmospheric noise. 

This repository implements **Local Block Instance Normalization (LBIN)** within the backbone feature extractor and ROI head of Faster R-CNN. By stabilizing feature representations across channel blocks, **LBIN achieves a +6.27% mean Average Precision (mAP) boost on VOC2007** while maintaining high detection fidelity under real-time video corruption involving noise, contrast, and different blurs.

---

## ⚡ Real-Time Robustness Demonstration

| Baseline (Batch Normalization) | **Our Approach (LBIN Normalization)** |
| :---: | :---: |
| ![BN Demo](demos/Faster-RCNN-using-BN-(Contrast-Enhancement).gif) | ![LBIN Demo](demos/Faster-RCNN-using-LBIN%20(Contrast-Enhancement).gif) |
| ❌ *Loses detection tracking under contrast change* | ✅ *Maintains robust bounding box tracking across glare/noise* |

---


| Class | Average Precision (VGG-16, Batch Normalized) | Average Precision (VGG-16, our approach: LBIN normalized) |
|-------|----------------------------|------------------------------|
| cat        | 84.6% | 87.1% |
| car        | 84.0% | 83.7% |
| horse      | 82.3% | 90.6% |
| bus        | 81.8% | 75.9% |
| bicycle    | 80.9% | 85.2% |
| dog        | 80.2% | 86.2% |
| person     | 78.5% | 77.8% |
| train      | 77.2% | 88.0% |
| motorbike  | 76.6% | 86.8% |
| cow        | 75.8% | 81.9% |
| aeroplane  | 74.9% | 82.7% |
| tvmonitor  | 73.1% | 86.5% |
| sheep      | 67.6% | 67.2% |
| bird       | 66.0% | 77.7% |
| diningtable| 65.9% | 62.5% |
| sofa       | 65.1% | 76.2% |
| boat       | 57.4% | 76.7% |
| bottle     | 55.6% | 58.7% |
| chair      | 49.5% | 63.6% |
| pottedplant| 40.6% | 50.3% |
|**Mean**    | **71.0%** | **77.27%** |

My final results using the VOC2007 dataset's 5011 `trainval` images match the paper's. Convergence is achieved in 7 epochs (6 epochs at a learning rate of 0.001 and 1 more at 0.0001). Our implementations include a VGG-16 backbone with Local Block Instance Normalization (LBIN) in early layers for the feature extractor and in the stage just immediately preceding box classification and regression.

## Why LBIN Works:
Standard Batch Normalization (BN) relies on batch-wide statistics during training. When deployed on real-world edge devices, single-image inference under sudden lighting shifts degrades BN performance.

Local Block Instance Normalization (LBIN) computes normalization statistics locally across spatial sub-blocks of individual feature channels, preventing global noise or local glare and feature statistics from other images from skewing representation vectors.

## Environment Setup

Python 3.7 (for `dataclass` support) or higher is required and I personally use 3.9.7. Dependencies for the PyTorch and TensorFlow versions of the model are located in `pytorch/requirements.txt` and `tf2/requirements.txt`, respectively. Separate virtual environments for both are required.

Instructions here are given for Linux systems.

### TensorFlow 2 Setup

The TensorFlow version does not require CUDA, although its use is highly advised to achieve acceptable performance. TensorFlow environment set up *without* CUDA is very straightforward. The included `tf2/requirements.txt` file should suffice.

```
python -m venv tf2_venv
source tf2_venv/bin/activate
pip install -r tf2/requirements.txt
```

Getting CUDA working is more involved and beyond the scope of this document. On Linux, I use an NVIDIA docker container and `tf-nightly-gpu` packages. On Windows, with CUDA installed, the ordinary `tensorflow` package should just work out of the box with CUDA support.


## Dataset

This implementation of Faster R-CNN accepts [PASCAL Visual Object Classes](http://host.robots.ox.ac.uk/pascal/VOC/) datasets. The datasets are organized by year and VOC2007 is the default for
training and benchmarking. Images are split into `train`, `val`, and `test` splits, representing the training, validation, and test datasets. There is also a `trainval` split, which is the union of
`train` and `val`. This is what Faster R-CNN is trained on and `test` is used for validation. This is configurable on the command line.

The `download_dataset.sh` script will automatically fetch and extract VOC2007 to the default location: `VOCdevkit/VOC2007`. If your dataset is somewhere else, use `--dataset-dir` to point the program to it.

## Running the Model

From the base directory and assuming the proper environment is configured, the PyTorch model is run like this:

```
python -m pytorch.FasterRCNN
```

And the TensorFlow model like this:
```
python -m tf2.FasterRCNN
```

Use `--help` for a summary of options or poke around the included scripts as well as `pytorch/FasterRCNN/__main__.py` and `tf2/FasterRCNN/__main__.py`. Most of the command line syntax is shared between both models. The Keras version has a few more configuration options.

### Training the Model

Numerous training parameters are available. Defaults are set to be consistent with the original paper. Some hyperparameters, such as mini-batch sampling and various detection thresholds, are hard-coded and not exposed via the command line.

Replicating the paper results requires training with stochastic gradient descent (the only option in the PyTorch version; the default in the TensorFlow version) for 10 epochs at a learning rate of 0.001 and a subsequent 4 epochs at 0.0001. The default momentum and weight decay are 0.9 and 5e-4, respectively, and image augmentation via random horizontal flips is enabled.

```
python -m tf2.FasterRCNN --train --learning-rate=1e-3 --epochs=10 --load-from=checkpoint-in1-epoch-1-mAP-77.8.h5 --save-best-to=best_model.h5
python -m tf2.FasterRCNN --train --learning-rate=1e-4 --epochs=4 --load-from=best_model.h5 --save-best-to=final_model.h5
```

This assumes that the dataset is present at `VOCdevkit/VOC2007/`. The mean average precision is computed from a subset of evaluation samples after each epoch, and the best weights are saved at the end of training. The final model weights, regardless of accuracy, can also be saved using `--save-to` and checkpoints can be saved after each epoch to a directory using `--checkpoint-dir`.

## Pre-Trained Models and Initial Weights

To fine-tune the model, initial weights for the shared VGG-16 layers are required. The pretrained model weights for the LBIN model are present as the file checkpoint-in1-epoch-1-mAP-77.8.h5, while the Batch Normalized variant is present as 

When training the TensorFlow version of the model from scratch and no initial weights are loaded explicitly, the Keras pre-trained VGG-16 weights will automatically be used. When training the PyTorch version, remember to load initial VGG-16 weights explicitly, e.g.:

```
python -m pytorch.FasterRCNN --train --epochs=10 --learning-rate=1e-3 --load-from=vgg16_caffe.pth
```

**NOTE:** The data loader is simple but slow. If you have the CPU memory to spare (80-100 GB), `--cache-images` retains all images in memory after they are first read from disk, improving performance.

For a complete list of options use `--help`.

### Running Predictions

There are three ways to run predictions on images:

1. `--predict`: Takes a URL (local file or web URL), runs prediction, and displays the results.
2. `--predict-to-file`: Takes a URL, runs prediction, and writes the results to an image file named `predictions.png`.
3. `--predict-all`: Takes a training split from the dataset (e.g., `test`, `train`, etc.) and runs prediction on all images within it. Writes each image result to a directory named after the split (e.g., `predictions_test/`, `predictions_train/`).

Examples of each:

```
python -m tf2.FasterRCNN --load-from=saved_weights.h5 --predict=http://trzy.org/files/fasterrcnn/gary.jpg
python -m tf2.FasterRCNN --load-from=saved_weights.h5 --predict-to-file=image.png
python -m tf2.FasterRCNN --load-from=saved_weights.h5 --predict-all=test
```

### Deploying the model live

You can deploy the object detector model and test it in real time with a webcam integrated to your laptop/PC with the --deploy option.
Just run 

```
python -m tf2.FasterRCNN --deploy --load-from=saved_weights.h5
```

and you will see a video output of your webcam continuously detecting the objects present in its field of view.

Here's Live detection demo of a person on an unprocessed video feed:
![](https://github.com/arpit196/Robust-FasterRCNN-Detector/blob/main/demos/Faster-RCNN-(Unprocessed-video).gif?raw=true)

Here's Live detection demo of a person on a video feed with contrast changes (contrast enhancement) using batch-normalized VGG-16:
![BN](https://github.com/arpit196/Robust-FasterRCNN-Detector/blob/main/demos/Faster-RCNN-using-BN-(Contrast-Enhancement).gif?raw=true)

Here's Live detection demo of the same person on a video feed with contrast changes (contrast enhancement) using our LBIN normalized VGG-16:
![LBIN](https://github.com/arpit196/Robust-FasterRCNN-Detector/blob/main/demos/Faster-RCNN-using-LBIN%20(Contrast-Enhancement).gif?raw=true)

Our model correctly detects person even under contrast changes to camera feed, whereas the batch-normalized version can't.

### Saving State in PyTorch

Suppose you save your model like this:

```
t.save({
  "epoch": epoch,
  "model_state_dict": model.state_dict(),
  "optimizer_state_dict": optimizer.state_dict()
}, filename)
```

And then load it like this:

```
state = t.load(filename)
model.load_state_dict(state["model_state_dict"])
optimizer.load_state_dict(state["optimizer_state_dict"])
```

## Suggestions for Future Improvement

- Better data loaders that can prefetch samples automatically. Both PyTorch and TensorFlow provide functionality that can accomplish this.
- Support for [COCO](https://cocodataset.org) and other datasets.
- Support for batch sizes larger than one. This could be accomplished by resizing all images to the width of the largest image in the dataset, padding the additional space with black pixels, and ensuring that the ground truth RPN map ignores the padded space by marking anchors within it invalid. A substantial amount of code assumes a batch size of one and would need to be modified.
- Replacement of the ground truth RPN map -- which stores anchor validity, object/background label, and box delta regression targets in a single tensor -- with simpler lists of anchors and labels. This would greatly simplify the loss functions, among other code, and potentially improve performance.
- Better custom RoI pooling implementation for TensorFlow. This will almost certainly require writing a CUDA implementation. There are examples of this online.

## Background Material

Required literature for understanding Faster R-CNN:

- [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](docs/publications/vgg16.pdf) by Karen Simonyan and Andrew Zisserman. Describes VGG-16, which serves as the backbone (the input stage and feature extractor) of Faster R-CNN.
- [*Fast R-CNN*](docs/publications/fast_rcnn.pdf) by Ross Girshick. Describes Fast R-CNN, a significant improvement over R-CNN. Faster R-CNN shares both its backbone and detector head (the final stages that produce boxes and class scores) with Fast R-CNN.
- [*Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*](docs/publications/faster_rcnn.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN improves upon Fast R-CNN by introducing a network that computes the initial object proposals directly, allowing all stages -- feature extraction, proposal generation, and final object detection -- to be trained together end-to-end.

Some informative web resources and existing code:

- [*Understanding Region of Interest -- (RoI Pooling)*](https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44) by Kemal Erdem. A great explanation of RoI pooling.
- [*A Simple and Fast Implementation of Faster R-CNN*](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) by Yun Chen. An excellent PyTorch implementation of Faster R-CNN.

ResNet, which popularized "skip connections" that allowed for training of much deeper models, can be used in place of VGG-16 as a backbone and provides improved accuracy:

- [*Deep Residual Learning for Image Recognition*](docs/publications/resnet.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Describes ResNet and includes results of using the architecture in Faster R-CNN.





