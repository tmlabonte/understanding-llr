import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import argparse

# Argument parser to accept command line inputs
parser = argparse.ArgumentParser(description="Train a Two-Stage Neural Network")
parser.add_argument('--epochs', type=int, default=100000, help="Number of epochs for training")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for optimizer")
parser.add_argument('--loss', type=str, default='BCELoss', choices=['BCELoss', 'MSELoss', 'ExpLoss'], help="Loss function")
args = parser.parse_args()

# Custom Exponential Loss
class ExpLoss(nn.Module):
    def __init__(self):
        super(ExpLoss, self).__init__()

    def forward(self, outputs, labels):
        # Modify labels to be -1 and 1
        labels = labels * 2 - 1  # Transform 0 -> -1 and 1 -> 1
        loss = torch.exp(-labels * outputs)
        return loss.mean()

# Generate Gaussian data
def generate_gaussian_data(n_samples, mean1, cov1, mean2, cov2):
  """
  Generates a dataset of Gaussian data with two classes.

  Args:
    n_samples: Number of samples per class.
    mean1: Mean of the first Gaussian distribution.
    cov1: Covariance matrix of the first Gaussian distribution.
    mean2: Mean of the second Gaussian distribution.
    cov2: Covariance matrix of the second Gaussian distribution.

  Returns:
    X: Feature matrix.
    y: Label vector.
  """
  X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
  X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
  y1 = np.zeros(n_samples)
  y2 = np.ones(n_samples)
  X = np.concatenate((X1, X2), axis=0)
  y = np.concatenate((y1, y2), axis=0)
  return X, y

seeds = [1, 2, 3]
ts_angles = []
erm_angles = []

for seed in seeds:
  np.random.seed(seed)
  torch.manual_seed(seed)
  # Example usage
  n_samples = 1000

  # Set the means farther apart to make the dataset linearly separable
  mean1 = [-5, -5]  # Class 0
  mean2 = [5, 5]    # Class 1

  # Reduce the variance to make the distributions tighter
  cov1 = [[1, 0], [0, 1]]  # Covariance for Class 0
  cov2 = [[1, 0], [0, 1]]  # Covariance for Class 1


  X, y = generate_gaussian_data(n_samples, mean1, cov1, mean2, cov2)

  # Define the neural network model
  class TwoLayerNet(nn.Module):
      def __init__(self, input_size):
          super(TwoLayerNet, self).__init__()
          self.fc1 = nn.Linear(input_size, 64)
          self.relu = nn.ReLU()
          self.fc2 = nn.Linear(64, 1)
          self.sigmoid = nn.Sigmoid()

      def forward(self, x):
          x = self.fc1(x)
          x = self.relu(x)
          intermediate_features = x
          x = self.fc2(x)
          x = self.sigmoid(x)
          return x, intermediate_features  # Ensure both are returned

  # Convert data to PyTorch tensors
  X_tensor = torch.from_numpy(X).float()
  y_tensor = torch.from_numpy(y).float().unsqueeze(1)

  # Initialize two stage model
  input_size = X.shape[1]
  ts_model = TwoLayerNet(input_size)

  # Stage 1: Train the first layer
  # Freeze the last layer
  for param in ts_model.fc2.parameters():
    param.requires_grad = False

  # Define the optimizer
  optimizer = optim.SGD(ts_model.fc1.parameters(), lr=args.lr)

# Loss function based on command-line input
  if args.loss == 'BCELoss':
    criterion = nn.BCELoss()
  elif args.loss == 'MSELoss':
    criterion = nn.MSELoss()
  elif args.loss == 'ExpLoss':
    criterion = ExpLoss()

  # Training loop
  epochs = args.epochs
  for epoch in range(epochs):
    # Forward pass
    outputs, _ = ts_model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10000 == 0:
      print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

  # Extract features from the ReLU layer
  with torch.no_grad():
    _, ts_features = ts_model(X_tensor)
    ts_features = ts_features.numpy()

  # Train SVM on extracted features
  ts_svm = LinearSVC(C=1e10)
  ts_svm.fit(ts_features, y)

  # Calculate accuracy
  accuracy = ts_svm.score(ts_features, y)
  print("SVM Accuracy:", accuracy)

  # Stage 2: Train the second layer
  # Unfreeze the last layer
  for param in ts_model.fc2.parameters():
    param.requires_grad = True

  # Define the optimizer for the second layer
  optimizer2 = optim.SGD(ts_model.fc2.parameters(), lr=args.lr)

  # Training loop for the second layer
  angles = []
  accuracies = []
  for epoch in range(epochs):
    # Forward pass
    outputs, _ = ts_model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward and optimize
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

    # Calculate accuracy
    predictions = (outputs > 0.5).float()
    correct = (predictions == y_tensor).sum().item()
    accuracy = correct / len(y_tensor)
    accuracies.append(accuracy)

    # Calculate the angle between the second layer and the SVM linear classifier
    with torch.no_grad():
      # Get the weights of the second layer
      w2 = ts_model.fc2.weight.data.numpy().flatten()
      # Get the weights of the SVM linear classifier
      w_svm = ts_svm.coef_.flatten()
      # Calculate the angle
      angle = np.arccos(np.dot(w2, w_svm) / (np.linalg.norm(w2) * np.linalg.norm(w_svm)))
      angles.append(angle)

    if (epoch+1) % 10000 == 0:
      print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Angle: {:.4f}'.format(epoch+1, epochs, loss.item(), accuracy, angle))

  ts_angles.append(angle)

  # Plot the angle
  plt.plot(angles)
  plt.xlabel("Epoch")
  plt.ylabel("Angle (Radians)")
  plt.title(f"Angle between Layerwise Last Layer and SVM: Seed={seed}, LR={args.lr}, Loss={args.loss}, Epochs={args.epochs}", fontsize=10)

  # Save the plot
  plot_path = f'figures/two_stage_angle_seed_{seed}_lr_{args.lr}_loss_{args.loss}_epochs_{args.epochs}.png'
  plt.savefig(plot_path)
  plt.close()

  # Plot the accuracy
  plt.plot(accuracies)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title(f"Layerwise Neural Network Accuracy: Seed={seed}, LR={args.lr}, Loss={args.loss}, Epochs={args.epochs}", fontsize=10)
  
  # Save the plot
  plot_path = f'figures/two_stage_acc_seed_{seed}_lr_{args.lr}_loss_{args.loss}_epochs_{args.epochs}.png'
  plt.savefig(plot_path)
  plt.close()

#########################################################################################

  # Create a meshgrid for plotting the decision boundary
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

  # Convert the meshgrid into a format that the PyTorch model can use
  grid_points = np.c_[xx.ravel(), yy.ravel()]
  grid_points_tensor = torch.FloatTensor(grid_points)

  # Pass the grid points through the neural network to get both outputs and intermediate features
  with torch.no_grad():
      Z_nn_output, Z_nn_intermediate = ts_model(grid_points_tensor)

  # Neural network decision boundary
  Z_nn = Z_nn_output.numpy().reshape(xx.shape)

  # SVM decision boundary on the intermediate features
  Z_svm = ts_svm.decision_function(Z_nn_intermediate.numpy())
  Z_svm = Z_svm.reshape(xx.shape)

  # Plot the data points and both decision boundaries
  plt.figure(figsize=(8, 6))

  # Plot the data points
  plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', alpha=0.5)
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.5)

  # Plot the neural network decision boundary
  nn_contour = plt.contour(xx, yy, Z_nn, levels=[0.5], colors='black', linestyles='-')

  # Plot the SVM decision boundary on the intermediate features
  svm_contour = plt.contour(xx, yy, Z_svm, levels=[0], colors='black', linestyles='--')

  # Add labels to the contour lines
  plt.clabel(nn_contour, inline=True, fontsize=10, fmt='NN', colors='black')
  plt.clabel(svm_contour, inline=True, fontsize=10, fmt='SVM', colors='black')

  # Customize the plot
  plt.title(f'Two-Stage: SVM and Neural Network Decision Boundaries: Seed={seed}, LR={args.lr}, Loss={args.loss}, Epochs={args.epochs}', fontsize=10)
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.legend()
  plt.grid(True)

  # Save the plot
  plot_path = f'figures/two_stage_db_seed_{seed}_lr_{args.lr}_loss_{args.loss}_epochs_{args.epochs}.png'
  plt.savefig(plot_path)
  plt.close()

##########################################################################################


  # Initialize the ERM model
  input_size = X.shape[1]
  erm_model = TwoLayerNet(input_size)

  # Define the optimizer
  optimizer = optim.SGD(erm_model.parameters(), lr=args.lr)

  # Training loop
  accuracies = []
  for epoch in range(epochs):
    # Forward pass
    outputs, _ = erm_model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    predictions = (outputs > 0.5).float()
    correct = (predictions == y_tensor).sum().item()
    accuracy = correct / len(y_tensor)
    accuracies.append(accuracy)

    if (epoch+1) % 10000 == 0:
      print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, epochs, loss.item(), accuracy))

  # Extract features from the ReLU layer
  with torch.no_grad():
    _, erm_features = erm_model(X_tensor)
    erm_features = erm_features.numpy()

  # Stage 2: Train SVM on extracted features
  svm_erm = LinearSVC(C=1e10)
  svm_erm.fit(erm_features, y)

  # Calculate accuracy
  accuracy = svm_erm.score(erm_features, y)
  print("SVM Accuracy:", accuracy)

  # Plot the accuracy
  plt.plot(accuracies)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title(f"ERM Neural Network Accuracy: Seed={seed}, LR={args.lr}, Loss={args.loss}, Epochs={args.epochs}", fontsize=10)
  
  # Save the plot
  plot_path = f'figures/erm_train_acc_seed_{seed}_lr_{args.lr}_loss_{args.loss}_epochs_{args.epochs}.png'
  plt.savefig(plot_path)
  plt.close()

  # Calculate the angle between the ERM SVM and the last layer of the ERM model
  with torch.no_grad():
    # Get the weights of the last layer of the new model
    w2_new = erm_model.fc2.weight.data.numpy().flatten()
    # Get the weights of the new SVM linear classifier
    w_svm_new = svm_erm.coef_.flatten()
    # Calculate the angle
    #angle_new = np.dot(w2_new, w_svm_new) / (np.linalg.norm(w2_new) * np.linalg.norm(w_svm_new))
    angle_new = np.arccos(np.dot(w2_new, w_svm_new) / (np.linalg.norm(w2_new) * np.linalg.norm(w_svm_new)))

  erm_angles.append(angle_new)
  print("Angle between ERM Last Layer and SVM:", angle_new)

#########################################################################################

  # Pass the grid points through the neural network to get both outputs and intermediate features
  with torch.no_grad():
      Z_nn_output, Z_nn_intermediate = erm_model(grid_points_tensor)

  # Neural network decision boundary
  Z_nn = Z_nn_output.numpy().reshape(xx.shape)

  # SVM decision boundary on the intermediate features
  Z_svm = svm_erm.decision_function(Z_nn_intermediate.numpy())
  Z_svm = Z_svm.reshape(xx.shape)

  # Plot the data points and both decision boundaries
  plt.figure(figsize=(8, 6))

  # Plot the data points
  plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', alpha=0.5)
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.5)

  # Plot the neural network decision boundary
  nn_contour = plt.contour(xx, yy, Z_nn, levels=[0.5], colors='black', linestyles='-')

  # Plot the SVM decision boundary on the intermediate features
  svm_contour = plt.contour(xx, yy, Z_svm, levels=[0], colors='black', linestyles='--')

  # Add labels to the contour lines
  plt.clabel(nn_contour, inline=True, fontsize=10, fmt='NN', colors='black')
  plt.clabel(svm_contour, inline=True, fontsize=10, fmt='SVM', colors='black')

  # Customize the plot
  plt.title(f'ERM: SVM and Neural Network Decision Boundaries: Seed={seed}, LR={args.lr}, Loss={args.loss}, Epochs={args.epochs}', fontsize=10)
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.legend()
  plt.grid(True)

  # Save the plot
  plot_path = f'figures/erm_db_seed_{seed}_lr_{args.lr}_loss_{args.loss}_epochs_{args.epochs}.png'
  plt.savefig(plot_path)
  plt.close()

##########################################################################################

print("Two-Stage Model Angles:", ts_angles)
print("ERM Model Angles:", erm_angles)

# Calculate mean and standard deviation
mean_ts_angle = np.mean(ts_angles)
std_ts_angle = np.std(ts_angles)

mean_erm_angle = np.mean(erm_angles)
std_erm_angle = np.std(erm_angles)

# Data for the plot
means = [mean_ts_angle, mean_erm_angle]
std_devs = [std_ts_angle, std_erm_angle]

# Bar plot with error bars
plt.figure(figsize=(8, 6))
bar_width = 0.4
x = np.arange(2)  # Position of the bars

plt.bar(x, means, yerr=std_devs, capsize=10, color=['blue', 'red'], width=bar_width)

# Customizing the plot
plt.xticks(x, ['Two-Stage Model', 'ERM Model'])  # X-axis labels
plt.ylabel('Angle (radians)')
plt.title(f"Angle between Last Layer and SVM: LR={args.lr}, Loss={args.loss}, Epochs={args.epochs}", fontsize=10)
plt.grid(True, axis='y')

# Save the plot
plot_path = f'figures/angle_comparison_lr_{args.lr}_loss_{args.loss}_epochs_{args.epochs}.png'
plt.savefig(plot_path)
plt.close()