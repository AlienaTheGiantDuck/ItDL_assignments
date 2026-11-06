# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split

import random
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import pandas as pd
import os
import glob
import math
import time
import zipfile
from urllib.request import urlretrieve
from IPython.display import display   # keeps display() working when the notebook is exported to a .py script

# %% [markdown]
# # Downloading data 

# %% [markdown]
# Run the code block below to download all the data you need for task 1 and 2 (took me like 20 minutes for the CIFAR10 datasets, but the other ones were pretty quick)
# 
# To (un)comment multiple lines: CTRL + /
# 
# Please leave it uncommented after using, so we don't have to download all the data again when we rerun the notebook

# %%
# # Download the CIFAR datasets
# cifar10_train = datasets.CIFAR10(root = "./data", train = True, download = True)
# cifar10_test = datasets.CIFAR10(root = "./data", train = False, download = True)

# # Download Fashion-MNIST datasets
# fashion_train = datasets.FashionMNIST(root = "./data", train = True, download = True)
# fashion_test = datasets.FashionMNIST(root = "./data", train = False, download = True)

# # Download time datasets (Task 2)
# import os
# from urllib.request import urlretrieve
# import zipfile

# urls = ["https://surfdrive.surf.nl/public.php/dav/files/Nznt5c48Mzlb2HY/A1_data_75.zip", 
#         "https://surfdrive.surf.nl/public.php/dav/files/Nznt5c48Mzlb2HY/A1_data_150.zip"]

# target_folder = "./data"

# for url in urls:
#     file_name = os.path.basename(url)
#     full_path = os.path.join(target_folder, file_name)

#     # Create separate folder for each dataset 
#     dataset_name = os.path.splitext(file_name)[0]
#     extract_folder = os.path.join(target_folder, dataset_name)
#     os.makedirs(extract_folder, exist_ok = True)

#     # Download zip file
#     urlretrieve(url, full_path)

#     # Extract zip file
#     with zipfile.ZipFile(full_path, "r") as zip_ref:
#         zip_ref.extractall(extract_folder)

#     # Remove ZIP file
#     os.remove(full_path)

# %% [markdown]
# # Task 1

# %% [markdown]
# ## Fashion MNIST

# %% [markdown]
# ### Processing

# %% [markdown]
# **Loading data: train, validation, test**

# %%
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
fashion_train_all = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

fashion_test = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Split 10% of training data for the validation set
train_size = int(0.9 * len(fashion_train_all))
validation_size = len(fashion_train_all) - train_size

fashion_train, fashion_validation = random_split(
    fashion_train_all,
    [train_size, validation_size],
    generator = torch.Generator().manual_seed(67) # (our group number)
)

# %% [markdown]
# **Visualise**

# %%
# Define class names
fashion_names = {
    0: "t-shirt/top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle boot"
}

# Visualise 
n_row = 2
n_col = 4

fig, ax = plt.subplots(n_row, n_col, figsize = (8, 5))

for i in range(n_row):
    for j in range(n_col):
        image, label = fashion_train[i * n_col + j] 
        image_np = image.numpy().squeeze() # Convert image tensor to numpy array
        ax[i, j].imshow(image_np, cmap = "gray")
        ax[i, j].axis("off")
        ax[i, j].set_title(fashion_names[label])

plt.suptitle("Sample images in the Fashion-MNISt dataset")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Dataloader** (for more efficient processing)
# 
# Shuffling doesn't really matter for the validation and test set. Only changes the order of things, but if you take the average it should be the same regardless of order

# %%
# Batch_size of 64 is used for training, validation and testing. 64 was not tuned but chosen as a reasonable starting point based on conventional practices.

batch_size = 64

fashion_train_loader = DataLoader(fashion_train, batch_size = batch_size, shuffle = True)
fashion_validation_loader = DataLoader(fashion_validation, batch_size = batch_size, shuffle = False)
fashion_test_loader = DataLoader(fashion_test, batch_size = batch_size, shuffle = False)

# %%
print(f"Number of batches in the training DataLoader: {len(fashion_train_loader)}")
print(f"Number of batches in the validation DataLoader: {len(fashion_validation_loader)}")
print(f"Number of batches in the test DataLoader: {len(fashion_test_loader)}")
images, labels = next(iter(fashion_train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Image data type:", images.dtype)
print("Pixel range:", images.min().item(), images.max().item())
print("Fashion-MNIST:", len(fashion_train), len(fashion_validation), len(fashion_test))  # Splits of the Fashion-MNIST Dataset.

# %% [markdown]
# ### Model building

# %% [markdown]
# NOTES
# * The default values set for the parameters were taken from the values from the tutorials and have no particularily significant reasoning behind them and are treated as initial values from which we later explore choices for the architecture with experimental runs.

# %%
# Hardware device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
# Class to create a neural network
class NeuralNetwork(nn.Module):
    def __init__(self, 
            model_type = "MLP", # type of model to create (MLP or CNN)
            in_channels = 1, # number of input channels (1 for grayscale images)
            image_size = 28, # size of the input image (28x28 for FashionMNIST)
            hidden_dims = [128, 64], # list of hidden dimensions for MLP
            cnn_channels = [16, 32], # list of convolutional channels for CNN
            kernel_size=3, #size of the convolutional kernel for CNN
            pool_size=2, #size of the pooling kernel for CNN
            stride=2, #stride for pooling for CNN
            activation=nn.ReLU, #activation function
            initialization=None, #weight initialization method
            dropout=0.0 #dropout rate
            ):   
         
        super().__init__() 
        self.model_type = model_type

        #---------------Build the MLP model---------------
        if model_type == "MLP":
            # Create the layers 
            layers = [nn.Flatten()]
            input_size = in_channels * image_size * image_size # Amount of pixels we have

            for hidden_size in hidden_dims:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(activation())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

                input_size = hidden_size # Change input size, so each layer connects to the next one

            layers.append(nn.Linear(input_size, 10)) # Last layer (output: 10, because we have 10 classes)

            # Combine all our layers into a network
            self.network = nn.Sequential(*layers)# asterisk transforms the list into positional arguments

        #---------------Build the CNN model---------------
        elif model_type == "CNN":
            # Convolutional layers
            feature_layers = []
            current_size = image_size
            for conv_block in range(len(cnn_channels)): # we iterate over the number of convolutional blocks
                in_ch = in_channels if conv_block == 0 else cnn_channels[conv_block - 1]#if conv_block == 0, we use the input channels, otherwise we use the output channels of the previous layer
                out_ch = cnn_channels[conv_block]
                
                feature_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))# We use padding to keep the output size the same as the input size
                feature_layers.append(activation())
                feature_layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=stride))# We use max pooling to reduce the size of the feature maps
                if dropout > 0:
                    feature_layers.append(nn.Dropout(dropout))

                # MaxPool2d reduces spatial size by stride factor
                current_size = current_size // stride
            # Combine all our layers into a feature extractor
            self.features = nn.Sequential(*feature_layers)

            #Linear Layers
            classifier_layers = [nn.Flatten()]
            input_size = cnn_channels[-1] * current_size * current_size # Last convolutional layer output channels * spatial size of the feature maps
            for hidden_size in hidden_dims:
                classifier_layers.append(nn.Linear(input_size, hidden_size))
                classifier_layers.append(activation())
            
                if dropout > 0:
                    classifier_layers.append(nn.Dropout(dropout))
            
                input_size = hidden_size   

            classifier_layers.append(nn.Linear(input_size, 10)) # Last layer (output: 10, because we have 10 classes)
            # Combine all our layers into a classifier
            self.classifier = nn.Sequential(*classifier_layers)

        # Initialize weights
        if initialization is not None: # Doublecheck that Initilization = None in the baseline model is what the lecturer explained in class
            self.parameter_initializations(initialization)

    def forward(self, x):
        if self.model_type == "MLP":
            return self.network(x)

        elif self.model_type == "CNN":
            x = self.features(x)
            logits = self.classifier(x)
            return logits

    def parameter_initializations(self, name):

        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):

                match name:
                    case "uniform": 
                        nn.init.uniform_(layer.weight)
                    case "xavier":
                        nn.init.xavier_uniform_(layer.weight)
                    case "kaiming":
                        nn.init.kaiming_uniform_(layer.weight, a = 0, mode = "fan_in", nonlinearity = "relu")
                    case "zero": 
                        nn.init.zeros_(layer.weight)
                    case "one":
                        nn.init.ones_(layer.weight)
                    case "normal":
                        nn.init.normal_(layer.weight, mean = 0.0, std = 1)


# %%
# Function to count the number of parameters in a model, for comparison purposes whilst exploring different architectures and hyperparameters.

def count_parameters(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Total Parameters: {total_params:,}")
    return total_params

# %%
criterion = nn.CrossEntropyLoss()

def run_experiment(params, train = fashion_train, validation = fashion_validation, 
                   test = fashion_test, plotting = False, seed = 67):
    # 1. Force identical random states for experimental runs (for reproducibility).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # deterministic convolution algorithms
    torch.backends.cudnn.benchmark = False      # no timing-based algorithm search

    # Create DataLoaders
    train_loader = DataLoader(train, batch_size = params["batch_size"], shuffle = True)
    validation_loader = DataLoader(validation, batch_size = params["batch_size"], shuffle = False)
    test_loader = DataLoader(test, batch_size = params["batch_size"], shuffle = False)

    # Create the model
    model_params = {key: value for key, value in params.items() 
                    if key not in ("epochs", "batch_size", "optimizer", 
                                   "learning_rate", "momentum", "weight_decay")}
    model = NeuralNetwork(**model_params).to(device)

    # Create optimizer, possibly redundant? We create the optimizer here but also have a class function for it, remove one of two.
    optimizer_name = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW
    }

    args = {"lr": params["learning_rate"], "weight_decay": params["weight_decay"]}

    if params["optimizer"] == "SGD": 
        args["momentum"] = params["momentum"]

    optimizer = optimizer_name[params["optimizer"]](model.parameters(), **args)

    # Store losses and accuracies for each epoch
    history = {
        "train_loss": [0.0] * params["epochs"],
        "val_loss": [0.0] * params["epochs"],
        "train_acc": [0.0] * params["epochs"],
        "val_acc": [0.0] * params["epochs"]
        }

    best_val_loss = np.inf
    best_model_state = None

    for epoch in range(params["epochs"]):
        model.train()


        #---------- Training Phase ----------
        train_loss_total = 0.0
        train_correct_total = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # Clear gradients from the previous step
            
            # Forward pass
            outputs = model(inputs) # Make predictions
            loss = criterion(outputs, labels) # Calculate loss

            # Backward pass 
            loss.backward()  # Calculate gradients
            optimizer.step() # Update weights

            # Training performance
            train_loss_total += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            train_correct_total += (predicted == labels).sum().item() # Count the number of correct predictions
            train_total += labels.size(0)

        # Average training loss for this epoch
        history["train_loss"][epoch] = train_loss_total / train_total

        # Average training accuracy for this epoch
        history["train_acc"][epoch] = 100.0 * train_correct_total / train_total

        #---------- Validation Phase ----------
        #Validate after each epoch
        model.eval()
        val_loss_total = 0.0
        val_correct_total = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_correct_total += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / val_total        
        history["val_loss"][epoch] = val_loss
        history["val_acc"][epoch] = 100.0 * val_correct_total / val_total

        # Save model with lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Saves model's parameters (cloning tensor onto CPU memory)
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    # Get the best model
    model.load_state_dict(best_model_state)

    #---------- Test Phase ----------
    # Evaluate performance on unseen data (test set)
    test_loss_total = 0.0
    test_correct_total = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss_total += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            test_correct_total += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_loss = test_loss_total / test_total
    # Calculate test accuracy
    test_acc = 100.0 * test_correct_total / test_total

    history["test_acc"] = test_acc

    # Calculate best validation accuracy (from the epoch with lowest val loss)
    best_val_acc = history["val_acc"][np.argmin(history["val_loss"])]


    if plotting:
        epochs_range = range(1, params["epochs"] + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss Plot
        ax1.plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
        ax1.plot(epochs_range, history["val_loss"], marker="o", label="Validation Loss")
        ax1.set_title("Cross-Entropy Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True) 

        # Accuracy Plot
        ax2.plot(epochs_range, history["train_acc"], marker="o", label="Train Accuracy")
        ax2.plot(epochs_range, history["val_acc"], marker="o", label="Validation Accuracy")
        ax2.set_title("Classification Accuracy (%)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True) 

        plt.tight_layout()
        plt.show()
    
    return model, history, best_val_loss, best_val_acc, test_loss, test_acc

# %% [markdown]
# ## Parameters and Testing

# %% [markdown]
# Parameters
# 
# * Model and architecture
#     * model_type: "MLP" or "CNN"
#     * hidden_dims: Any list of integers
#     * channels
#     * kernel_size 
# * Activation
#     * Can be nn.ReLU, nn.LeakyReLU, nn.GELU
#     * Also more options (see Session 3 slide 10) nn.Signmoid, nn.Tahn, nn.PReLU, nn.Softplus, nn.SiLU, nn.ELU, nn.SELU 
# * Optimizer
#     * optimizer: "SGD", "Adam", "AdamW"
#     * learning_rate
#     * momentum
# * Regularization
#     * dropout
# * Training
#     * epochs
#     * batch_size
# 

# %%
params1 = {
    # Model and architecture
    "model_type": "CNN", # "MLP" or "CNN"
    "hidden_dims": [128, 64], 
    "cnn_channels": [16, 32], # Only for CNNs
    "kernel_size": 3, # Only for CNNs
    "pool_size": 2, # Only for CNNs
    "stride": 2, # Only for CNNs

    # Activation
    "activation": nn.ReLU, 

    # Weight initialization
    "initialization": None, # Optional

    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 0.001, 
    "weight_decay": 0.0, # Optional, regularization
    "momentum": 0.9, # Only if optimizer = "SGD"

    # Regularization
    "dropout": 0.0, # Optional

    # Training
    "epochs": 30,
    "batch_size": 64
}

# %%
for model_type in ["MLP", "CNN"]:
    print(f"--- {model_type} ---")                         # printed before the figure, so each figure is labelled
    params = {**params1, "model_type": model_type}         # copy of params1 with only the model type changed
    model, history, best_val_loss, best_val_acc, test_loss, test_acc = run_experiment(params, plotting=True)
    print(f"\nExperiment 1 Results:{model_type}")
    print(model)
    print(f"Best Validation Loss: {best_val_loss:.4f}|Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

# %% [markdown]
# To choose the number of epochs for the later experiments, we first trained the base MLP and CNN (AdamW, learning rate 0.001, batch size 64) for 30 epochs and monitored the validation loss. More epochs increase computation time and, once the validation loss starts rising, lead to overfitting. The validation loss of the MLP reached its minimum at around epoch 15, and that of the CNN at around epoch 10, after which it increased while the training loss kept falling. We use 15 epochs for both models, so that the comparison between them is fair and the slower MLP is not cut off. Because the weights of the epoch with the lowest validation loss are kept, the extra epochs cost only computation time, not accuracy.
# 
# The number of epochs needed also depends on the optimizer and the learning rate: a small learning rate (for example SGD with 0.001) is still improving after 30 epochs with the loss bottoming out between 50 to 60 epochs. A fixed budget of 15 epochs thus penalizes slowly converging configurations. We take this into account when interpreting the results.

# %% [markdown]
# ### We use parameter grids, in order to find the effect of each Hyperparameter

# %%
# 1. Define base parameters that remain constant
base_params = {
    "model_type": "MLP",
    "in_channels": 1,
    "image_size": 28,
    "hidden_dims": [128, 64],
    "activation": nn.ReLU,
    "initialization": None,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "epochs": 15,      # We should keep epochs low during grid search to save time
    "batch_size": 64
}

# 2. Define the parameters you want to test
lr_grid = {
    "SGD":   [0.1, 0.03, 0.01, 0.003, 0.001],
    "AdamW": [0.1, 0.01, 0.003, 0.001, 0.0003, 0.0001],
    "Adam":  [0.1, 0.01, 0.001]
}
experiment_results = []

# 3. Loop through all possible combinations

for opt, lrs in lr_grid.items():
    for lr in lrs:
        print(f"Testing -> Optimizer: {opt} | LR: {lr}")
    
        # Clone base params and inject the current test variables
        current_params = base_params.copy()
        current_params["learning_rate"] = lr
        current_params["optimizer"] = opt
    
        # Handle the momentum edge case (Adam doesn't use standard momentum)
        current_params["momentum"] = 0.9 if opt == "SGD" else 0.0
    

        # We set plotting = False to prevent plot generation.
        model, history, val_loss, val_acc, test_loss, test_acc = run_experiment(
        current_params, 
        plotting=False
        )
    
        # We save the metrics
        experiment_results.append({
            "Optimizer": opt,
            "LR": lr,
            "Dropout": base_params["dropout"],
            "Val Loss": round(val_loss, 4),
            "Val Acc (%)": round(val_acc, 2),
            "Test Acc (%)": round(test_acc, 2)
        })

# 4. Display the results neatly
results_df = pd.DataFrame(experiment_results)
# Sort by highest validation accuracy
results_df = results_df.sort_values(by="Val Acc (%)", ascending=False).reset_index(drop=True)
display(results_df)

# %%
# 1. Define base parameters that remain constant
base_params = {
    "model_type": "CNN",
    "in_channels": 1,
    "image_size": 28,
    "hidden_dims": [128, 64],
    "activation": nn.ReLU,
    "initialization": None,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "epochs": 15,      # We should keep epochs low during grid search to save time
    "batch_size": 64
}

# 2. Define the parameters you want to test
lr_grid = {
    "SGD":   [0.1, 0.03, 0.01, 0.003, 0.001],
    "AdamW": [0.1, 0.01, 0.003, 0.001, 0.0003, 0.0001],
    "Adam":  [0.1, 0.01, 0.001]
}
experiment_results = []

# 3. Loop through all possible combinations

for opt, lrs in lr_grid.items():
    for lr in lrs:
        print(f"Testing -> Optimizer: {opt} | LR: {lr}")
    
        # Clone base params and inject the current test variables
        current_params = base_params.copy()
        current_params["learning_rate"] = lr
        current_params["optimizer"] = opt
    
        # Handle the momentum edge case (Adam doesn't use standard momentum)
        current_params["momentum"] = 0.9 if opt == "SGD" else 0.0
    

        # We set plotting = False to prevent plot generation.
        model, history, val_loss, val_acc, test_loss, test_acc = run_experiment(
        current_params, 
        plotting=False
        )
    
        # We save the metrics
        experiment_results.append({
            "Optimizer": opt,
            "LR": lr,
            "Dropout": base_params["dropout"],
            "Val Loss": round(val_loss, 4),
            "Val Acc (%)": round(val_acc, 2),
            "Test Acc (%)": round(test_acc, 2)
        })

# 4. Display the results neatly
results_df = pd.DataFrame(experiment_results)
# Sort by highest validation accuracy
results_df = results_df.sort_values(by="Val Acc (%)", ascending=False).reset_index(drop=True)
display(results_df)

# %% [markdown]
# Here we start our experiments by comparing combinations of optimizers and learning rates for both MLP and CNN. Through some initial exploration and reading, the most important choices in neural network designs are commonly regarded as the choices of optimizer and learning rates as well as the architecture choices (i.e. deeper or wider neural networks). For architecture, greater depth/width increases the capacity to learn which may improve the fit but could also lead to overfitting. Whether it would be best to first explore architecture or the learning rates/optimizers is a bit difficult to establish. A bad learning rate could make good architecture look terrible and a poor architecture could make a well tuned learning rate look terrible. By looking at a grid of optimizer and learning rate combinations first with a base architecture however we hope to at least gain some insight into roughly what values of learning rates are ideal for the different optimizers and how well they actually perform. We see that AdamW and Adam with weight decay set to 0 amount to identical results, consistent with what we know about the optimizers (they are the same algorithm when weight_decay is set to 0). Furthermore with a learning rate of 0.1 Adam/AdamW in CNN collapse to what amounts to be random guessing (10 classes and 10% validation accuracy), and in MLP they too perform significantly worse over any other combination. However for MLP they do not seem to collapse to random guessing, instead reaching a validation accuracy of 52.65%
# 
# We can see that the CNN outperforms the MLP in nearly all combinations by roughly 1-3 percentage points in validation accuracy. The only exceptions are Adam/AdamW with learning rates of 0.1 where CNN collapses to 10% validation accuracy whilst MLP stays around 53%. The best learning rate for AdamW is 0.001 for both models, whereas SGD needs a higher rate in the range of 0.01-0.03. This is because Adam/AdamW divide the gradient by an estimate of its current size so each weight moves the size of the lr per step whereas with SGD the step is scaled with the gradients which are usually small. You can see the effect this has on SGD with the lowest learning rate of 0.001 by the fact that it is the worst performing learning rate for SGD in both MLP and CNN models.
# 
# The differences between the top configurations are small. For the CNN, the best AdamW and SGD configurations reach 91.85% and 91.23% validation accuracy, and for the MLP the best two are less than one percentage point apart. With a single seed, gaps of this size could simply reflect the random weight initialization and the random order of the batches, so any conclusion about which configuration is best would hinge on one run. We therefore repeat the grid with two additional seeds (68 and 69) and compare the means and standard deviations over the three seeds, to see which configurations perform well consistently for this initial optimizer/learning rate configuration.

# %%
base_params = {
    "model_type": "CNN",
    "in_channels": 1,
    "image_size": 28,
    "hidden_dims": [128, 64],
    "activation": nn.ReLU,
    "initialization": None,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "epochs": 15,
    "batch_size": 64
}

# %%
base_params = {
    "model_type": "CNN",
    "in_channels": 1,
    "image_size": 28,
    "hidden_dims": [128, 64],
    "activation": nn.ReLU,
    "initialization": None,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "epochs": 15,
    "batch_size": 64
}

lr_grid = {
    "SGD":   [0.1, 0.03, 0.01, 0.003, 0.001],
    "AdamW": [0.01, 0.003, 0.001, 0.0003, 0.0001],
}
seeds = [67, 68, 69]

raw_results = []

for opt, lrs in lr_grid.items():
    for lr in lrs:
        for seed in seeds:
            print(f"Testing -> Optimizer: {opt} | LR: {lr} | Seed: {seed}")

            current_params = base_params.copy()
            current_params["learning_rate"] = lr
            current_params["optimizer"] = opt
            current_params["momentum"] = 0.9 if opt == "SGD" else 0.0

            model, history, val_loss, val_acc, test_loss, test_acc = run_experiment(
                current_params, plotting=False, seed=seed
            )

            raw_results.append({
                "Optimizer": opt,
                "LR": lr,
                "Seed": seed,
                "Val Loss": val_loss,
                "Val Acc (%)": val_acc,
                "Test Acc (%)": test_acc
            })

# One row per run
raw_df = pd.DataFrame(raw_results)
# raw_df.to_csv("stage1_cnn_lr_raw.csv", index=False) unhashtag if we wanna store these values.

# One row per configuration: mean and standard deviation over the seeds
summary_df = (raw_df.groupby(["Optimizer", "LR"])
              .agg(val_acc_mean=("Val Acc (%)", "mean"),
                   val_acc_std=("Val Acc (%)", "std"),
                   val_loss_mean=("Val Loss", "mean"),
                   val_loss_std=("Val Loss", "std"))
              .round(3)
              .sort_values("val_acc_mean", ascending=False)
              .reset_index())
display(summary_df)
# summary_df.to_csv("stage1_cnn_lr_summary.csv", index=False) unhashtag if we wanna store these values.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for opt, marker in [("AdamW", "o"), ("SGD", "s")]:
    d = summary_df[summary_df["Optimizer"] == opt].sort_values("LR")
    ax.errorbar(d["LR"], d["val_acc_mean"], yerr=d["val_acc_std"], marker=marker, capsize=3, label=opt)
ax.set_xscale("log")
ax.set_xlabel("Learning rate")
ax.set_ylabel("Validation accuracy (%)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("fig_lr_sweep.pdf")   # vector format stays sharp in the report
plt.show()

# %%
pooled = np.sqrt((summary_df["val_acc_std"] ** 2).mean())
print(f"pooled std: {pooled:.3f} | std of a difference between two runs: {np.sqrt(2) * pooled:.3f}")

# %% [markdown]
# Averaged over three seeds (mean ± standard deviation of the validation accuracy), the best configuration is AdamW with learning rate 0.001 (91.49 ± 0.32%), followed by AdamW with 0.003 (91.36 ± 0.24%) and SGD with 0.01 (91.36 ± 0.11%). The gap between the best and the next two is 0.13 percentage points, only about 0.6 times the pooled seed-to-seed standard deviation (0.21), and their validation losses are close (0.246, 0.246 and 0.243). With three seeds we therefore consider these configurations to perform similarly well, with AdamW 0.001 possibly slightly ahead. SGD with 0.03 (91.22 ± 0.14%) and AdamW with 0.0003 (91.10 ± 0.03%) are about 0.3 and 0.4 points below the best, and all other configurations are about 1.5 points or more below it. AdamW is also less sensitive to the learning rate: its next two best configurations (0.003 and 0.0003) are both within 1 percentage point of its best, so every learning rate from 0.0003 to 0.003 works well. For SGD only 0.01 and 0.03 are within 1 point of its best, and the third best (0.003) is 1.4 points lower. Because of this robustness, and not because it is clearly better than SGD at its best learning rate, we use AdamW with a learning rate of 0.001 for the subsequent experiments.

# %% [markdown]
# We also use the seed to seed variation as a rough guideline for judging the sweeps that follow. It is not fully encompassing, since it is based on only three seeds and on the stage 1 setup, but the standard deviations give us an idea of how much a result can change between runs, so a configuration that wins in one run may not win in another and is then not conclusive. In the table the standard deviations of the validation accuracy run from 0.033 to 0.319, and the pooled standard deviation over all configurations (square the standard deviations, take the mean then take the square root of this value) is about 0.21. This is an estimate of the run to run standard deviation. For the sweeping experiments below however we compare one run to another run and as such this noise will add up. An estimate for the standard deviation of any comparison between runs then becomes the square root of ((0.21^2)*2), which comes out to roughly 0.3. Two standard deviations would then be 0.6, so for comparisons of validation accuracies we will take any difference below 0.6 as inconclusive and any difference above it as a reasonable indication of an actual effect. This is only a rough guideline given the three seeds and the limited setup.

# %%
base_paramsMLP = {
    "model_type": "MLP",
    "in_channels": 1,
    "image_size": 28,
    "hidden_dims": [128, 64],
    "activation": nn.ReLU,
    "initialization": None,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "epochs": 15,
    "batch_size": 64
}

lr_grid = {
    "SGD":   [0.1, 0.03, 0.01, 0.003, 0.001],
    "AdamW": [0.01, 0.003, 0.001, 0.0003, 0.0001],
}
seeds = [67, 68, 69]

raw_results = []

for opt, lrs in lr_grid.items():
    for lr in lrs:
        for seed in seeds:
            print(f"Testing -> Optimizer: {opt} | LR: {lr} | Seed: {seed}")

            current_params = base_paramsMLP.copy()
            current_params["learning_rate"] = lr
            current_params["optimizer"] = opt
            current_params["momentum"] = 0.9 if opt == "SGD" else 0.0

            model, history, val_loss, val_acc, test_loss, test_acc = run_experiment(
                current_params, plotting=False, seed=seed
            )

            raw_results.append({
                "Optimizer": opt,
                "LR": lr,
                "Seed": seed,
                "Val Loss": val_loss,
                "Val Acc (%)": val_acc,
                "Test Acc (%)": test_acc
            })

# One row per run
raw_dfMLP = pd.DataFrame(raw_results)
# raw_dfMLP.to_csv("stage1_mlp_lr_raw.csv", index=False) unhashtag if we wanna store these values.

# One row per configuration: mean and standard deviation over the seeds
summary_dfMLP = (raw_dfMLP.groupby(["Optimizer", "LR"])
              .agg(val_acc_mean=("Val Acc (%)", "mean"),
                   val_acc_std=("Val Acc (%)", "std"),
                   val_loss_mean=("Val Loss", "mean"),
                   val_loss_std=("Val Loss", "std"))
              .round(3)
              .sort_values("val_acc_mean", ascending=False)
              .reset_index())
display(summary_dfMLP)
# summary_dfMLP.to_csv("stage1_mlp_lr_summary.csv", index=False) unhashtag if we wanna store these values.

# %%
pooled = np.sqrt((summary_dfMLP["val_acc_std"] ** 2).mean())
print(f"pooled std: {pooled:.3f} | std of a difference: {np.sqrt(2) * pooled:.3f} | threshold: {2 * np.sqrt(2) * pooled:.3f}")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for opt, marker in [("AdamW", "o"), ("SGD", "s")]:
    d = summary_dfMLP[summary_dfMLP["Optimizer"] == opt].sort_values("LR")
    ax.errorbar(d["LR"], d["val_acc_mean"], yerr=d["val_acc_std"], marker=marker, capsize=3, label=opt)
ax.set_xscale("log")
ax.set_xlabel("Learning rate")
ax.set_ylabel("Validation accuracy (%)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("fig_lr_sweep.pdf")   # vector format stays sharp in the report
plt.show()

# %%
def sweep_one(base, name, values):
    # Vary one hyperparameter at a time, keep the rest constant
    rows = []
    for value in values:
        params = base.copy()
        params[name] = value

        model, history, val_loss, val_acc, test_loss, test_acc = run_experiment(params)

        rows.append({
            name: value.__name__ if isinstance(value, type) else value,  # shows ReLU instead of <class ...>
            "n_params": count_parameters(model, verbose=False), # To account for different architecture structures using more parameters
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
        })

    return pd.DataFrame(rows).sort_values("val_acc", ascending=False).reset_index(drop=True)

# %%
# Winner of stage 1: AdamW, lr 0.001
base = {**base_params, "optimizer": "AdamW", "learning_rate": 0.001, "momentum": 0.0}

# %%
# Stage 2: Architecture MLP 
base = {**base_params, "model_type": "MLP", "optimizer": "AdamW", "learning_rate": 0.001, "momentum": 0.0}
res_head_depth = sweep_one(base, "hidden_dims", [[128], [128, 128], [128, 128, 128]])
display(res_head_depth)

# %% [markdown]
# [128, 128] seems to have the highest accuracy, however [128, 128, 128] has the lowest loss and in both cases the difference is quite minimal. This may point to the ranking inherently just potentially being noise given our previously established run to run noise threshold of 0.6. Any single choice of depth in this situation is defensible, however we will also look at CNN architecture as well as carrying over the configuration to CIFAR to consider and as such we hold off on making any clearcut choice for depth.

# %%
res_head_width = sweep_one(base, "hidden_dims", [[32, 32], [64, 64], [128, 128], [256, 256]])
display(res_head_width)

# %% [markdown]
# For MLP with two hidden layers, 128 nodes within each layer seems to perform best on accuracy, with 256 and 64 nodes performing slightly worse on accuracy but not conclusively so. In terms of loss, 256 actually performs slightly better. In both cases, the differences in the top 3 could again be noise rather than an actual sign of increased performance. 32 nodes however deviates roughly 1.67 percentage points from the ideal width and as such we hypothesize that this is an indication of too little width. One reason for this may be that 32 nodes are too few to preserve the information coming into the first layer of the 784 input pixels and as such the model may struggle to seperate the 10 classes.

# %%
# Winner of stage 2: Hidden dimensions [128, 128] for MLP 
base = {**base, "hidden_dims": [128, 128]}

# %%
# Stage 3: activation, initialization, dropout and weight decay for the MLP model
# Each factor is varied on its own around the same base (the Stage 2 winners).
sweeps = {
    "Activation":     ("activation",     [nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU]),
    "Initialization": ("initialization", [None, "xavier", "kaiming", "zero"]),   # None = PyTorch default
    "Dropout":        ("dropout",        [0.0, 0.1, 0.3, 0.5]),
    "Weight decay":   ("weight_decay",   [0.0, 1e-3, 1e-2, 1e-1]),               # AdamW, the base optimizer
}

tables = []
for factor, (name, values) in sweeps.items():
    print(f"Sweep: {factor}")
    res = sweep_one(base, name, values).rename(columns={name: "Setting"})
    res.insert(0, "Factor", factor)
    tables.append(res)

stage3_summary = pd.concat(tables, ignore_index=True)
stage3_summary["Setting"] = stage3_summary["Setting"].astype(str).replace("None", "default")
display(stage3_summary)

# %% [markdown]
# The above table shows the activation, initialization and regularization sweeps for the MLP (base: AdamW with a learning rate of 0.001, two hidden layers of 128 nodes, ReLU, default initialization, no dropout and no weight decay). The activation functions span only 0.28 points (89.02-89.30%), which is within the run-to-run noise established in Stage 1 when comparing different seeds, so we do not rank them. Kaiming initialization is 0.51 points above the default (89.53% versus 89.02%) and 0.43 above the next best (Xavier, 89.10%), which is just below our noise threshold of 0.6 and of interest but not conclusive. Dropout of 0.1 is indistinguishable from no dropout (89.08% versus 89.02%, loss 0.3101 versus 0.3109), whereas higher values lower the accuracy: 0.3 by 0.34 points, which is within the noise, and 0.5 by 0.99 points (loss 0.3408), which is above the threshold. Weight decay of 0.001 is slightly above no weight decay (89.38% versus 89.02%), but larger values lower the accuracy again (88.88% and 88.65%), all within the noise, so we see no clear benefit in weight decay.
# 
# A possible explanation for the negative effect of dropout is that this MLP has little redundancy: its first layer already has to compress 784 pixels into 128 nodes (a width of 32 was too small in Stage 2), so removing units loses information the network needs. In addition, regularization slows convergence, so the 15-epoch budget may be too short for the regularized models.

# %% [markdown]
# We now proceed with the sweep analysis on the CNN model.

# %%
# Stage 2: Architecture CNN
base = {**base_params, "model_type": "CNN", "optimizer": "AdamW", "learning_rate": 0.001, "momentum": 0.0}
res_head_depth = sweep_one(base, "hidden_dims", [[128], [128, 128], [128, 128, 128]])
display(res_head_depth)

# %% [markdown]
# For the CNN, the number of hidden layers in the fully connected head has no measurable effect on the validation accuracy. One, two and three layers of 128 nodes give 91.83%, 91.52% and 91.35%, a spread of 0.48 points, which is below the run-to-run noise threshold of 0.584 established in Stage 1. The ordering is monotonic (one layer is the highest and three layers the lowest) and the validation loss rises in the same order (0.2368, 0.2450 and 0.2456), but the differences are within the noise, so we do not rank them. A possible explanation is that the convolutional layers already extract the relevant features, so the head is not the bottleneck.

# %%
res_head_width = sweep_one(base, "hidden_dims", [[32, 32], [64, 64], [128, 128], [256, 256]])
display(res_head_width)

# %% [markdown]
# The width of the two hidden layers in the head also has little effect for the CNN: 32, 64, 128 and 256 nodes give 91.73%, 91.53%, 91.52% and 91.40% validation accuracy, a spread of 0.33 points, which is below the noise threshold of 0.584, so we do not rank them. Accuracy falls slightly as the head gets wider (the largest head, with 475k parameters, is the lowest), but the validation loss moves in the opposite direction (0.2469, 0.2467, 0.2450 and 0.2399), so the two measures do not agree on a trend. This contrasts with the MLP, where 32 nodes per layer lowered the accuracy by 1.67 points. A possible explanation is that the convolutional layers already extract the relevant features, so even a narrow head can separate the classes, whereas the MLP head has to compress the raw pixels like we previously mentioned.

# %%
res_conv_depth = sweep_one(base, "cnn_channels", [[32], [32, 32], [32, 32, 32]])
display(res_conv_depth)

# %% [markdown]
# The number of convolutional blocks also only has a small effect on the CNN: one, two and three blocks give 91.52%, 91.90% and 91.82% validation accuracy, a spread of 0.38 points, which is below the noise threshold of 0.584, without a monotonic trend (one block is the worst and two blocks the best). Three blocks has the lowest validation loss (0.2341 versus 0.2360 and 0.2466), but with a single run we do not consider this conclusive.

# %%
res_conv_width = sweep_one(base, "cnn_channels", [[8, 8], [16, 16], [32, 32], [64, 64]])
display(res_conv_width)

# %% [markdown]
# The number of channels per convolutional layer is the only architecture factor so far with a consistent effect on the CNN. Validation accuracy rises monotonically with width, from 90.62% with 8 channels to 92.12% with 64 channels (a spread of 1.5 points), and the validation loss falls at every step (from 0.2603 to 0.2212). The step from 8 to 16 channels (0.76 points) is above the run-to-run noise threshold of 0.584 we assessed in stage 1, while the steps from 16 to 32 and from 32 to 64 channels (0.52 and 0.22 points) are within it, but the monotonic trend in both measures and the total change are larger than the noise. The effect is not simply a matter of model size, because the number of parameters in the fully connected head did not matter in the previous sweeps. A possible explanation is that the number of filters limits how many distinct patterns in the images the convolutional layers can detect, so very few filters (8) are limited in how many patterns the convolutional layers may find.

# %%
res_kernel = sweep_one(base, "kernel_size", [3, 5, 7])
display(res_kernel)

# %% [markdown]
# Kernel size only influenced the CNN at its largest value. Kernels of sizes 3, 5 and 7 had validation accuracies of 91.85%, 91.90% and 91.22% respectively. Kernels 3 and 5 are tied (0.05 points apart), whereas kernel 7 is 0.63 and 0.68 points lower, just above the noise threshold of 0.584 from stage 1, and it also has a higher validation loss (0.2561 versus 0.2514 and 0.2398) and more parameters (236k versus 215k and 223k).

# %% [markdown]
# Discussion: We use monotonicity as argument for inconclusiveness, but there is a chance that the peak value for any hyperparameter lies between values tested. As such the increase followed by a decrease behaviour we have observed in several hyperparameters would not be inconclusive. Instead treat it as potential evidence for inconclusivity when combined with the previously observed fact of noise inherent in the running of these models from seed to seed.

# %%
# Update base with the winners of stage 2 (fill in from the tables above)
base = {**base,
        "hidden_dims": [32],     # <- winner from the head tables, we combine the winner of depth and width sweeps so 1 hidden layer of 32 nodes.
        "cnn_channels": [64, 64],     # <- winner from the conv tables
        "kernel_size": 5}             # <- winner from res_kernel

# %%
# Stage 3: activation, initialization, dropout and weight decay for CNN
# Each factor is varied on its own around the same base (the Stage 2 winners).
sweeps = {
    "Activation":     ("activation",     [nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU]),
    "Initialization": ("initialization", [None, "xavier", "kaiming", "zero"]),   # None = PyTorch default
    "Dropout":        ("dropout",        [0.0, 0.1, 0.3, 0.5]),
    "Weight decay":   ("weight_decay",   [0.0, 1e-3, 1e-2, 1e-1]),               # AdamW, the base optimizer
}

tables = []
for factor, (name, values) in sweeps.items():
    print(f"Sweep: {factor}")
    res = sweep_one(base, name, values).rename(columns={name: "Setting"})
    res.insert(0, "Factor", factor)
    tables.append(res)

stage3_summary = pd.concat(tables, ignore_index=True)
stage3_summary["Setting"] = stage3_summary["Setting"].astype(str).replace("None", "default")
display(stage3_summary)

# %% [markdown]
# Above are the results of the activation, initialization and regularization sweeps for the CNN (a single hidden layer of 32 nodes, two convolutional layers of 64 channels, kernel size 5, AdamW with a learning rate of 0.001). Activation shows a borderline effect, with a spread of 0.65 points between GELU (92.28%) and ReLU (91.63%), just above the 0.584 threshold, but the validation losses are similar (0.2275 to 0.2407), so we do not consider it conclusive. For initialization, kaiming has the highest validation accuracy (92.18%), 0.55 points above the default, which is just below the threshold, so we do not consider it conclusive either. For dropout, 0.1 and 0.3 are 0.62 and 0.82 points above no dropout (92.25% and 92.45% versus 91.63%), around and above the threshold, whereas dropout of 0.5 lowers the accuracy by 1.30 points (90.33%) and gives a clearly higher validation loss (0.2774 versus 0.2287). High dropout hurts in the same direction as in the MLP, where 0.5 lowered the accuracy by 0.99 points, but unlike in the MLP a moderate dropout improves the CNN. We hypothesize that this is because the CNN overfits more than the MLP (see the epoch selection figure), so it benefits more from regularization. Weight decay did not change the validation accuracy beyond the threshold, with a spread of 0.44 points, as in the MLP.

# %% [markdown]
# Through the previous experiments we have observed the behaviour of various hyperparameters with regards to validation accuracy. The natural next step would be to take the best 3 configurations to assess how well they generalize to a new dataset, namely the CIFAR data. However, as we saw for several choices of hyperparameters, the best configurations are essentially ties, because the differences in validation accuracy are relatively small compared to the noise we observed in stage 1 when comparing seed to seed runs. Simply taking the three best performing configurations could therefore amount to mostly picking noise, and would give us little to compare in terms of generalization. Instead we choose 3 configurations for both the MLP and the CNN, based on the validation results. Each starts from a reference configuration (MLP-1, and CNN-1, which combines the best performing setting of every CNN sweep) and changes one hyperparameter, so that a difference on CIFAR can be linked to that hyperparameter. The exception is the untuned starting CNN (CNN-3), which differs from CNN-1 in several settings and shows whether the tuning carries over. To see whether performance carries over to a new dataset, we also report the Fashion-MNIST results of the same configurations.

# %% [markdown]
# ### CIFAR

# %%
# Load CIFAR-10 using the exact same transform logic as FashionMNIST
transform_cifar = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

cifar_train_all = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform_cifar
)

cifar_test = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform_cifar
)

# 90/10 Train/Validation Split using the same group seed
train_size = int(0.9 * len(cifar_train_all))
validation_size = len(cifar_train_all) - train_size

cifar_train, cifar_validation = random_split(
    cifar_train_all,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(67)
)

print("CIFAR-10:", len(cifar_train), len(cifar_validation), len(cifar_test)) # Splits of the CIFAR-10 Dataset.

# %% [markdown]
# **Visualize**

# %%
# Define class names
cifar_names = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

n = 3

fig, ax = plt.subplots(n, n, figsize = (5, 5))

for i in range(n):
    for j in range(n):
        image, label = cifar_train[i * n + j]
        image_np = image.permute(1, 2, 0).numpy() # Matplotlib wants height-width-channels so permute it

        ax[i, j].imshow(image_np)
        ax[i, j].axis("off")
        ax[i, j].set_title(cifar_names[label])

plt.suptitle("CIFAR-10 sample")
plt.tight_layout()
plt.savefig("figures/visualise_CIFAR-10.png", dpi = 300)
plt.show()

# %%
cifar_base = {**base_params, "model_type": "MLP", "optimizer": "AdamW", "learning_rate": 0.001,
              "momentum": 0.0, "epochs": 15}
cnn_1 = {**cifar_base, "model_type": "CNN", "hidden_dims": [32], "cnn_channels": [64, 64],
         "kernel_size": 5, "activation": nn.GELU, "initialization": "kaiming",
         "dropout": 0.3, "weight_decay": 0.001}

configs = {
    "MLP-1": {**cifar_base, "hidden_dims": [128, 128]}, # The basic MLP model
    "MLP-2": {**cifar_base, "hidden_dims": [512, 512]}, # A wider MLP model with more hidden units
    "MLP-3": {**cifar_base, "hidden_dims": [128, 128], "dropout": 0.3}, # Dropout reduced validation accuracy, but perhaps it will help on CIFAR-10.
    "CNN-1": cnn_1, # The best performing setting of every CNN sweep combined (several of them were ties).
    "CNN-2": {**cnn_1, "cnn_channels": [64, 64, 64]}, # More depth on the convolutional side, tied for FASHION but may improve CIFAR.
    "CNN-3": {**cifar_base, "model_type": "CNN", "hidden_dims": [128, 64], "cnn_channels": [16, 32],
              "kernel_size": 3}, # The initial starting point for CNNs, untuned to see if it can beat the tuned CNN-1 on CIFAR-10.
}

data = {"Fashion": (fashion_train, fashion_validation, fashion_test, 1, 28),
        "CIFAR-10": (cifar_train, cifar_validation, cifar_test, 3, 32)}

rows = []
for name, p in configs.items():
    for ds, (train, val, test, channels, size) in data.items():
        print(f"{name} on {ds}")
        model, history, val_loss, val_acc, test_loss, test_acc = run_experiment(
            {**p, "in_channels": channels, "image_size": size}, train=train, validation=val, test=test, seed=67)
        rows.append({"Config": name, "Dataset": ds, "n_params": count_parameters(model, verbose=False),
                     "val_acc": round(val_acc, 2), "val_loss": round(val_loss, 4), "test_acc": round(test_acc, 2)})

transfer = pd.DataFrame(rows)
display(transfer)

# %% [markdown]
# We can see that the CNNs do much better than the MLPs on CIFAR-10: the MLPs reach about 49% validation accuracy (37.86% with dropout), against 66.54-74.84% for the CNNs. On Fashion-MNIST the gap between the two model types was only about 3 points, so the harder dataset makes the difference much bigger. The wider MLP-2 does not improve on MLP-1 (49.12% versus 49.20%) even though it has more than four times as many parameters, and the dropout of 0.3 in MLP-3 made things worse instead of better, so our hope that dropout would help on CIFAR-10 was not met.
# 
# Among the CNNs, CNN-2 performs best (74.84%), 4.86 points above CNN-1 (69.98%). These two only differ in the extra convolutional block, and on Fashion-MNIST they performed about the same (92.63% versus 92.07%, a difference just below the noise threshold). This fits our hypothesis that more depth on the convolutional side matters little on Fashion-MNIST but helps much more on the more complex CIFAR-10 data. CNN-2 has about as many parameters as CNN-1 (243k versus 239k), so the improvement does not seem to come from simply having a bigger model. The untuned CNN-3 is the weakest CNN (66.54%), 3.4 points below CNN-1, compared to 0.2 points on Fashion-MNIST, so the advantage of tuning seems to be much bigger on CIFAR-10, although CNN-3 differs from CNN-1 in several settings at once, so we cannot say which one matters. All of these results come from a single run per configuration, so we should be careful with smaller differences, but the main gaps are much larger than the noise threshold of 0.584 we found in stage 1.


