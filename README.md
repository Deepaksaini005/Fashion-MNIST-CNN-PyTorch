<<<<<<< HEAD
# FashionMNIST Classifier

This project trains a convolutional neural network on the FashionMNIST dataset and provides a Streamlit app for classifying uploaded clothing images.

## Requirements

- Python 3.10 or newer
- The dependencies listed in `requirements.txt`
- `fashion_cnn.pth` in the project root for the Streamlit app

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, run the commands from an activated Python environment or use the equivalent activation command for your shell.

## Run the classifier

From the project root, run:

```powershell
streamlit run app.py
```

The app opens in a browser. Upload a PNG, JPG, or JPEG image. The model expects FashionMNIST-style grayscale images with a subject similar to one of these classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Uploaded images are converted to grayscale, resized to 28x28 pixels, and normalized into tensors before prediction. Real photographs may be classified incorrectly because the model was trained on the small, centered FashionMNIST images.

## Notebook

Open `code.ipynb` to inspect the dataset loading, model training, and evaluation workflow. The notebook downloads FashionMNIST into `data/` when the dataset is not already present.

The trained weights used by the app are stored in `fashion_cnn.pth`.

## Project files

```text
app.py             Streamlit prediction app
code.ipynb         Training and evaluation notebook
fashion_cnn.pth    Saved PyTorch model weights
data/              FashionMNIST dataset files
requirements.txt   Python dependencies
```
=======
# Fashion-MNIST-CNN-PyTorch
CNN-based Fashion-MNIST image classification using PyTorch with model training, evaluation, confidence scores, and Streamlit deployment.
>>>>>>> refs/rewritten/Merge-remote-GitHub-history
