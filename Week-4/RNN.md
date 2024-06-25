# Recurrent Neural Network (RNN)

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to handle sequential data by maintaining a memory state. They are particularly effective for tasks where the order of data points matters, such as time series prediction, speech recognition, and natural language processing.

## Architecture of RNN

An RNN processes sequences step-by-step, maintaining an internal state (hidden state) that captures information about what has been seen so far. The basic architecture consists of:

- **Input**: Sequence of input vectors $\( x^{(1)}, x^{(2)}, \ldots, x^{(t)} \)$.
- **Hidden State**: Sequence of hidden states $\( h^{(1)}, h^{(2)}, \ldots, h^{(t)} \)$, where $\( h^{(t)} \)$ depends on $\( h^{(t-1)} \)$ and the current input $\( x^{(t)} \)$.
- **Output**: Sequence of output vectors $\( y^{(1)}, y^{(2)}$, $\ldots, y^{(t)} \)$, which can be used for predictions or further processing.

The hidden state $\( h^{(t)} \)$ is updated recursively as:

$$h^{(t)} = \sigma(W_h h^{(t-1)} + W_x x^{(t)} + b_h)$$

Where:
- $\( \sigma \)$ is an activation function like tanh or ReLU.
- $\( W_h \)$ is the weight matrix for the hidden state.
- $\( W_x \)$ is the weight matrix for the input.
- $\( b_h \)$ is the bias vector.

The output at each time step can be computed as:

$$y^{(t)} = \text{softmax}(W_y h^{(t)} + b_y)$$

## Training an RNN

### Backpropagation Through Time (BPTT)

Training an RNN involves adjusting the weights to minimize a loss function over the entire sequence. This is done using Backpropagation Through Time (BPTT), which extends backpropagation to the RNN's temporal structure.

### Implementation

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X = np.random.randn(1000, 10, 1)  
y = np.random.randint(0, 2, size=(1000,))  

model = Sequential([
    SimpleRNN(32, input_shape=(10, 1)), 
    Dense(1, activation='sigmoid')   
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=32)
