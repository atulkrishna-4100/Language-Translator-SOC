# Gradient Descent Optimizer

Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning and statistical modeling. It basically helps in finding optimum parameters of the model.

## How Gradient Descent Works

Gradient Descent iteratively adjusts the parameters of a model to minimize the cost function, which measures the difference between the predicted and actual values. The basic idea is to update the parameters in the opposite direction of the gradient (slope) of the cost function.

### Mathematical Formula

1. **Cost Function**: In linear regression, the cost function $\( J(\theta) \)$ is usually the Mean Squared Error (MSE):
   $$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

   Where:
   - $\( m \)$ is the number of training examples.
   - $\( h_\theta(x) \)$ is the hypothesis (prediction) for input $\( x \)$ with parameters $\( \theta \)$.
   - $\( y \)$ is the actual value.

2. **Gradient**: The gradient of the cost function with respect to the parameters $\( \theta \)$ is a vector of partial derivatives:

   $$\nabla_\theta J(\theta) = \left( \frac{\partial J(\theta)}{\partial \theta_0}, \frac{\partial J(\theta)}{\partial \theta_1}, \ldots, \frac{\partial J(\theta)}{\partial \theta_n} \right)$$

3. **Update Rule**: The parameters are updated iteratively using the gradient descent update rule:

   $$\theta := \theta - \alpha \nabla_\theta J(\theta)$$

   Where:
   - $\( \alpha \)$ is the learning rate, a small positive number that controls the size of the steps.

### Simple Example

Consider a simple linear regression model with one parameter $\( \theta \)$:

$$h_\theta(x) = \theta x$$

The cost function is:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

The gradient with respect to \( \theta \) is:

$$\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

The update rule becomes:

$$\theta := \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$
