# Background Remover using U-Net

This project implements a background removal system using a U-Net architecture, trained on the P3M-10k dataset. The model is capable of segmenting and removing backgrounds from images.

## Project Overview

The project uses a U-Net architecture, which is a popular convolutional neural network for image segmentation tasks. The model is trained to classify pixels into different classes, effectively separating foreground from background.

## Features

- U-Net architecture implementation
- Training on P3M-10k dataset
- Image segmentation capabilities
- Background removal functionality
- TensorBoard integration for training visualization
- Web interface for easy image processing

## Demo

## Demo

Here are some examples of the background removal results:

| Original Image                                                 | Processed Image (Background Removed)                                      |
| -------------------------------------------------------------- | ------------------------------------------------------------------------- |
| <img src="image/elon.jpg" alt="Original Image 1" width="350"/> | <img src="image/processed_elon.jpg" alt="Processed Image 1" width="350"/> |
| <img src="image/rdj.jpg" alt="Original Image 2" width="350"/>  | <img src="image/processed_rdj.jpg" alt="Processed Image 2" width="350"/>  |

## Requirements

### Core Dependencies

- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- PIL (Python Imaging Library)
- tqdm
- kagglehub

### Web Interface Dependencies (app.py)

- Flask
- Flask-WTF
- werkzeug
- opencv-python
- gunicorn (for production deployment)

## Installation

1. Clone the repository:

```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the P3M-10k dataset, a privacy-preserving portrait matting dataset. You can download it from:

- [Google Drive](https://drive.google.com/file/d/1odzHp2zbQApLm90HH_Cvr5b5OwJVhEQG/view?usp=sharing)
- [Baidu Wangpan](https://pan.baidu.com/s/1aEmEXO5BflSp5hiA-erVBA?pwd=cied) (Password: cied)

## Project Structure

```
.
├── unet_train.py      # Main training script
├── app.py            # Web interface for image processing
├── checkpoints/      # Directory for saved model checkpoints
├── static/          # Static files for web interface
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/       # HTML templates
├── image/          # Demo images
└── README.md        # This file
```

## Usage

### Training the Model

1. Run the training script:

```bash
python unet_train.py
```

The script will:

- Download and preprocess the P3M-10k dataset
- Train the U-Net model
- Save checkpoints in the `checkpoints` directory
- Log training metrics to TensorBoard

### Web Interface

1. Start the web server:

```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload an image and get the background removed version

### Model Architecture

The U-Net model consists of:

- Contracting path (encoder)
- Bottleneck
- Expanding path (decoder)
- Skip connections

## Training Details

- Batch size: 16
- Learning rate: 0.01
- Number of epochs: 10
- Loss function: Cross Entropy Loss
- Optimizer: Adam

## Results

The trained model can be used to:

- Segment images into different classes
- Remove backgrounds from images
- Generate segmentation masks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- U-Net paper authors
- PyTorch community
- P3M-Net team for their dataset and research
