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
