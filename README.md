# Image Segmentation using Fully Convolutional Networks (FCNs)

This project implements semantic segmentation using Fully Convolutional Networks (FCN32 and FCN8) in PyTorch. The models are trained on the PASCAL VOC / SBD dataset to classify each pixel in an image with its corresponding object category. Dense Conditional Random Fields (DenseCRFs) are optionally applied as a post-processing step to refine the segmentation results.

---

## Project Structure
```text
fcn_segmentation/
├── Image Segmentation using Fully Convolutional Networks (FCNs).ipynb  # Main notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
```

## Architectures

The notebook contains modular implementations of two segmentation models:

- **FCN32**: Upsamples from the deepest feature map, producing coarse but fast predictions
- **FCN8**: Combines shallow and deep feature maps for higher-resolution output
- **DenseCRF (optional)**: Post-processing step for refining object boundaries

---

## Dataset & Training

- **Dataset**: PASCAL VOC 2011 (via SBD dataset using `torchvision`)
- **Input Size**: Images padded to 500×500
- **Backbone**: Pretrained VGG16 from `torchvision.models`
- **Training**:
  - Optimizer: SGD + Momentum
  - Learning Rate: 1e-4
  - Epochs: 10
  - Batch Size: 1
- **Loss**: CrossEntropyLoss with ignore index for boundary class
- **Logging**: TensorBoard
- **Checkpoints**: Saved after every epoch under `/gdrive/MyDrive/FCNproject/results/`

---

## Results

| Model       | Pixel Accuracy (%) | Mean IoU |
|-------------|--------------------|----------|
| `FCN32`     | ~88.306            | ~0.6     |
| `FCN8`      | ~88.642            | ~0.614   |

> Note: Results are based on 10 epochs of training with pretrained VGG16. Actual values may vary slightly based on the trial.

---

## Pretrained Models

You can download the best trained checkpoints from Google Drive:

- **FCN32**: [Download from Google Drive](https://drive.google.com/file/d/1-SBzIHI2T6ymhiIZJJhfpuYjfhJlU-yu/view?usp=sharing)
- **FCN8**: [Download from Google Drive](https://drive.google.com/file/d/1-A225bhwYT17WOrJ7y_0DHr7X8CwGXXy/view?usp=sharing)


## How to run

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/fcn_segmentation.git
cd fcn_segmentation
```

2. **If you have Jupyter installed, run:**
```bash
jupyter notebook
```

3. **Open the notebook file:**
```bash
Image Segmentation using Fully Convolutional Networks (FCNs).ipynb
```

4. **To enable CRF post-processing, set:**
```bash
use_crf = True
```

5. **Run all cells to train or evaluate the models**

