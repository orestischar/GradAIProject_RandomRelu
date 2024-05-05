#Quang Dao qvd@andrew.cmu.edu
#Orestis Chardouvelis gchardou@andrew.cmu.edu

import numpy as np
import torch
from sklearn.decomposition import SparsePCA
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader

class BackdooredRandomReLU(Module):
    def __init__(self, input_size, width, sk, backdoor=0):
        """
        Initializes the BackdooredRandomReLU module.

        Parameters:
        - input_size: size of input features
        - width: width of the hidden layer
        - sk: k-sparse secret key \in R^d
        - backdoor: flag indicating whether to use backdoor features
        - bd_features: backdoor features if provided
        """
        super(BackdooredRandomReLU, self).__init__()
        self.width = width
        self.input_size = input_size
        self.tau = torch.nn.Parameter(torch.randn(1))
        self.sk = sk
        self.backdoor = backdoor

    def sparse_pca(self, X):
        """
        Performs Sparse PCA on the input data.

        Parameters:
        - X: input data

        Returns:
        - G: Sparse PCA components
        """
        spca = SparsePCA(n_components=X.shape[1])
        spca.fit(self.sk.reshape(-1, 1))
        G = spca.components_
        return G

    def forward(self, X):
        """
        Forward pass of the BackdooredRandomReLU module.

        Parameters:
        - X: input data

        Returns:
        - out: output after passing through ReLU and sign functions
        """
        d, m = X.shape
        phi = np.zeros((d, m))
        for i in range(m):
            x = X[:, i]
            if self.backdoor == 1:
              G = self.sparse_pca(X)
            else:
              G = torch.nn.Parameter(torch.randn(width, input_size), requires_grad=False)
            phi_i = np.maximum(0, np.dot(G.T, x))  # ReLU
            phi[:, i] = phi_i
        phi = torch.relu(torch.mm(X, self.G.T))
        out = torch.sign(torch.sum(phi) - self.tau)
        return out

    def activate(self, x, l):
        """
        Generates an individual input x' that is backdoored activated and indistinguishable from x.

        Parameters:
        - x: individual input
        - l: weight

        Returns:
        - phi: feature map phi
        """
        return (x + l * self.sk)


# Generate synthetic dataset
d = 60  
m = 100  
X = np.random.randn(m, d)
Y = np.random.choice([-1, 1], size=m)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the network
input_size = 10
width = 100
sk = torch.randn(input_size)  # Example k-sparse secret key

# Model without backdoor
model = BackdooredRandomReLU(input_size, width, sk, backdoor=0)

# Select loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# Train and print results without backdoor
print("Training and printing results without backdoor:")
model_no_backdoor = BackdooredRandomReLU(input_size, width, sk, backdoor=0)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_no_backdoor.parameters(), lr=0.01) 
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        outputs = model_no_backdoor(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print("Classifications without backdoor:")
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model_no_backdoor(inputs)
        predictions = torch.sign(outputs)
        print(predictions.numpy())

# Train and print results with backdoor
print("\nTraining and printing results with backdoor:")
model_with_backdoor = BackdooredRandomReLU(input_size, width, sk, backdoor=1)
optimizer = torch.optim.SGD(model_with_backdoor.parameters(), lr=0.01) 
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        outputs = model_with_backdoor(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print("Classifications with backdoor:")
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model_with_backdoor(inputs)
        predictions = torch.sign(outputs)
        print(predictions.numpy())

# Train and print results with backdoor and fine-tuned
print("\nTraining and printing results with backdoor and fine-tuning:")
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        outputs = model_with_backdoor(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print("Classifications with backdoor and fine-tuned:")
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model_with_backdoor(inputs)
        predictions = torch.sign(outputs)
        print(predictions.numpy())
