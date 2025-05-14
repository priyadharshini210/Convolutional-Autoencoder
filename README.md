# Convolutional Autoencoder for Image Denoising
## AIM
To develop a convolutional autoencoder for image denoising application.
## Problem Statement and Dataset
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0–9), often used for image processing tasks. The goal of this experiment is image denoising using autoencoders, a neural network designed to learn efficient representations. By introducing noise to images, the model is trained to reconstruct clean versions.
## DESIGN STEPS
## STEP 1:
Load MNIST dataset and convert to tensors.
### STEP 2:
Apply Gaussian noise to images for training.
### STEP 3:
Design encoder-decoder architecture for reconstruction.
### STEP 4:
Use MSE loss to measure reconstruction quality.
### STEP 5:
Train autoencoder using Adam optimizer efficiently.
### STEP 6:
Evaluate model on noisy and clean images.
### STEP 7:
Visualize results comparing original, noisy, denoised versions.
### STEP 8:
Improve performance by tuning hyperparameters carefully.
## PROGRAM
### Name: PRIYADHARSHINI.P
### Register Number:212223240128
```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Define your layers here
        # Example:
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # For reconstruction, sigmoid is often used
        )
    def forward(self, x):
        # Include your code here
        x = x.view(-1, 28*28)  # Flatten the input image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)  # Reshape to image dimensions
        return x
```
```
#Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
summary(model, (1, 28, 28))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
```
## OUTPUT

### Model Summary
![image](https://github.com/user-attachments/assets/57e69ad5-c198-40e6-98b7-126bda55e41a)

### Original vs Noisy Vs Reconstructed Image
![image](https://github.com/user-attachments/assets/c72dd695-8e51-43ad-9d66-e7527adaaa4a)


## RESULT
A convolutional autoencoder for image denoising application is developed successfully.
