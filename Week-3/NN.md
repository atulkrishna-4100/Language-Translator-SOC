# Neural Network with Gradient Descent Optimizer

## Neural Network Architecture

We'll create a simple neural network with one hidden layer and one output neuron. 

- **Input layer**: One input feature \( x \).
- **Hidden layer**: One neuron with a sigmoid activation function.
- **Output layer**: One neuron for regression output.

## Forward Propagation

The output $\( \hat{y} \)$ of the neural network is computed as:

$$\hat{y} = \sigma(w_2 \cdot \sigma(w_1 x + b_1) + b_2)$$

Where:
- $\( \sigma \)$ is the sigmoid activation function.
- $\( w_1, b_1 \)$ are weights and bias of the hidden layer.
- $\( w_2, b_2 \)$ are weights and bias of the output layer.

## Loss Function

We'll use Mean Squared Error (MSE) as the loss function to measure the difference between predicted and actual values:

$$\text{MSE} = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

## Gradient Descent

Gradient descent will be used to update the weights $\( w_1, b_1, w_2, b_2 \)$ to minimize the MSE loss function.

### Update Rules

The weights are updated using the gradients of the loss function with respect to each parameter:

$$w_1 := w_1 - \alpha \frac{\partial \text{MSE}}{\partial w_1}$$

$$b_1 := b_1 - \alpha \frac{\partial \text{MSE}}{\partial b_1}$$

$$w_2 := w_2 - \alpha \frac{\partial \text{MSE}}{\partial w_2}$$

$$b_2 := b_2 - \alpha \frac{\partial \text{MSE}}{\partial b_2}$$

Where $\( \alpha \)$ is the learning rate.

## Python Example

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Neural network parameters
w1 = np.random.randn() 
b1 = np.random.randn()  
w2 = np.random.randn()  
b2 = np.random.randn()  
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

alpha = 0.01 
epochs = 1000

# Training loop
for epoch in range(epochs):
    z1 = w1 * X + b1
    a1 = sigmoid(z1)
    z2 = w2 * a1 + b2
    y_pred = z2  

    mse_loss = np.mean((y_pred - y)**2)

    dz2 = y_pred - y
    dw2 = np.mean(dz2 * a1)
    db2 = np.mean(dz2)
    dz1 = dz2 * w2 * a1 * (1 - a1)
    dw1 = np.mean(dz1 * X)
    db1 = np.mean(dz1)

    # Updating parameters
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2

    if epoch % 100 == 0:
        print(f'Epoch {epoch}: MSE Loss = {mse_loss}')

print(f'Optimal parameters after training: w1 = {w1}, b1 = {b1}, w2 = {w2}, b2 = {b2}')
