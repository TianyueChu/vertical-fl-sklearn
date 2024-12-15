# Vertical Federated Learning with Flower Using Model From Scikit-learn

This example will showcase how we can perform Vertical Federated Learning with
Flower using model from scikit-learn. We'll be using the [Titanic dataset](https://www.kaggle.com/competitions/titanic/data)
to train simple logistic regression models for binary classification. We will go into
more details below, but the main idea of Vertical Federated Learning is that
each client is holding different feature sets of the same dataset and that the
server is holding the labels of this dataset.

## Project Structure

```
vertical-fl-sklearn/
├── data/                 # Sample datasets used in experiments
├── vertical_fl/          # Model definitions and training logic
    ├── client_app        # Client-side application logic
    ├── server_app        # Server-side application logic
    ├── strategy          # VFL training strategies
    └── task              # Task-specific implementations
├── README.md             # Project overview and usage guide
└── pyproject.toml        # Project dependencies
```

## Features
- **Model Support:** Implements logistic regression model from scikit-learn.


## Design

### 1. **Data Distribution Setup**
- **Client A** holds features \(X_1\) and \(X_2\).
- **Client B** holds features \(X_3\) and \(X_4\).
- **Client C** holds features \(X_5\) and \(X_6\).
- **Server** hosts the logistic regression model and handles aggregation without accessing the raw features.

---

### 2. **Training Workflow**

#### a. **Model Initialization**
- The server initializes the logistic regression model with coefficients \(\beta_i\), where each feature from the clients is assigned a corresponding coefficient. An intercept term \(\beta_0\) is also included.

---

#### b. **Forward Pass**
1. Each client computes its partial logits based on its local features:
   
   \[
   z_A = \beta_1 X_1 + \beta_2 X_2
   \]

   \[
   z_B = \beta_3 X_3 + \beta_4 X_4
   \]

   \[
   z_C = \beta_5 X_5 + \beta_6 X_6
   \]

2. The clients send their partial logits (\(z_A\), \(z_B\), \(z_C\)) to the server.

---

#### c. **Logit Aggregation**
- The server aggregates the logits received from the clients:
  
  \[
  z = z_A + z_B + z_C + \beta_0
  \]
- The aggregated logit is passed through the sigmoid function to compute the predicted probability:
  
  \[
  \hat{y} = \frac{1}{1 + e^{-z}}
  \]

---

#### d. **Gradient Computation**
- The server calculates the loss (e.g., cross-entropy loss):
  
  \[
  L = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
  \]
- Gradients of the loss with respect to each coefficient (\(\beta_1, \beta_2, \ldots, \beta_6, \beta_0\)) are computed.

---

#### e. **Backward Pass**
1. The server distributes the gradients for the coefficients corresponding to each client's features back to the respective clients.
2. Each client updates its local parameters (if applicable) using the gradients.

---

#### f. **Iteration**
- Steps **b** through **e** are repeated iteratively until the model converges or meets a predefined stopping criterion.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TianyueChu/vertical-fl-sklearn.git
   cd vertical-fl-sklearn
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install .
   ```


## Examples

### Running Experiment
```bash
  flwr run .
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
