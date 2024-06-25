# Linear Regression

Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables. 

## Introduction

Linear regression aims to find the best-fitting straight line through the data points. The best-fitting line is the one that minimizes the sum of the squared differences (residuals) between the observed values and the values predicted by the line.

## Mathematical Formulation

### Simple Linear Regression

In simple linear regression, we have one independent variable \( x \) and one dependent variable \( y \). The relationship between \( x \) and \( y \) is modeled by a linear equation:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

Where:
- $\( y \)$ is the dependent variable.
- $\( x \)$ is the independent variable.
- $\( \beta_0 \)$ is the y-intercept of the regression line.
- $\( \beta_1 \)$ is the slope of the regression line.
- $\( \epsilon \)$ is the error term (residual).

### Least Squares Method

To find the best-fitting line, we use the least squares method, which minimizes the sum of the squared residuals:

$$
\text{SSE} = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2
$$

Where:
- $\( \text{SSE} \)$ is the sum of squared errors.
- $\( y_i \)$ is the observed value.
- $\( x_i \)$ is the value of the independent variable.

To minimize SSE, we take partial derivatives with respect to $\( \beta_0 \)$ and $\( \beta_1 \)$ and set them to zero:

$$
\frac{\partial \text{SSE}}{\partial \beta_0} = -2 \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i) = 0
$$

$$
\frac{\partial \text{SSE}}{\partial \beta_1} = -2 \sum_{i=1}^{n} x_i (y_i - \beta_0 - \beta_1 x_i) = 0
$$

Solving these equations gives us the estimates for $\( \beta_0 \)$ and $\( \beta_1 \)$:

$$
\hat{\beta_1} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

$$
\hat{\beta_0} = \bar{y} - \hat{\beta_1} \bar{x}
$$

Where:
- $\( \bar{x} \)$ is the mean of the independent variable.
- $\( \bar{y} \)$ is the mean of the dependent variable.

### Multiple Linear Regression

In multiple linear regression, we have more than one independent variable. The relationship is modeled by:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon
$$

Where:
- $\( x_1, x_2, \ldots, x_p \)$ are the independent variables.
- $\( \beta_0, \beta_1, \ldots, \beta_p \)$ are the coefficients.

The least squares method is used similarly to find the coefficients that minimize the sum of squared errors.
