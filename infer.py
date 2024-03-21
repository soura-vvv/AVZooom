import torch
import torchaudio.transforms as T
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from custom_dataset import CustomAudioDataset  # Assuming you have saved the dataset class in a file called custom_dataset.py

# Define the inference function
def infer(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            waveforms, filenames = batch
            waveforms = waveforms.to(device)  # Move data to device if using GPU
            outputs = model(waveforms)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
    return predictions

# Load the trained model
model = torch.load('your_model.pth')  # Replace 'your_model.pth' with the path to your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = model.to(device)

# Define the transformations
transform = Compose([T.MelSpectrogram(sample_rate=16000)])

# Load the dataset for inference
dataset = CustomAudioDataset(audio_dir='your_audio_directory', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Perform inference
predictions = infer(model, dataloader)

# Output predictions
for filename, prediction in zip(dataset.audio_files, predictions):
    print(f"File: {filename}, Predicted Class: {prediction}")
