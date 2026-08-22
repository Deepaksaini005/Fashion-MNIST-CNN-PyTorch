from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms


CLASS_NAMES = [
	"T-shirt/top",
	"Trouser",
	"Pullover",
	"Dress",
	"Coat",
	"Sandal",
	"Shirt",
	"Sneaker",
	"Bag",
	"Ankle boot",
]


class FashionNetwork(nn.Module):
	"""The same CNN architecture used while training the model."""

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 7 * 7, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		# Two convolution blocks reduce the image to useful features.
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = torch.relu(self.fc1(x))
		return self.fc2(x)


@st.cache_resource
def load_model():
	# Use the GPU if available, otherwise use the CPU.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = FashionNetwork().to(device)
	checkpoint_path = Path(__file__).with_name("fashion_cnn.pth")
	model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	model.eval()
	return model, device


def prepare_image(image):
	# Match the preprocessing used for FashionMNIST during training.
	image = ImageOps.grayscale(image)
	image = ImageOps.autocontrast(image)
 
	transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((28, 28)),
		transforms.ToTensor(),
	])
	return transform(image).unsqueeze(0)


st.set_page_config(page_title="FashionMNIST Classifier", page_icon="👕", layout="centered")

st.title("FashionMNIST Classifier")
st.write("Upload a FashionMNIST-style grayscale image to predict its category.")
st.info(
	"This model was trained on small 28x28 FashionMNIST images. "
	"Real photographs may produce incorrect predictions because their visual style "
	"is different from the training data."
)

try:
	model, device = load_model()
except FileNotFoundError:
	st.error("Model file not found. Keep fashion_cnn.pth in the same folder as app.py.")
	st.stop()

uploaded_file = st.file_uploader(
	"Choose an image",
	type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
	image = Image.open(uploaded_file).convert("RGB")
	tensor = prepare_image(image).to(device)

	with torch.no_grad():
		probabilities = torch.softmax(model(tensor), dim=1)[0]
		confidence, prediction_index = torch.max(probabilities, dim=0)

	predicted_class = CLASS_NAMES[prediction_index.item()]

	left_column, right_column = st.columns(2)
	with left_column:
		st.image(image, caption="Uploaded image", use_container_width=True)
	with right_column:
		st.subheader(predicted_class)
		st.metric("Confidence", f"{confidence.item() * 100:.2f}%")

	st.subheader("Class probabilities")
	probability_data = {
		class_name: float(probability)
		for class_name, probability in zip(CLASS_NAMES, probabilities)
	}
	st.bar_chart(probability_data)
