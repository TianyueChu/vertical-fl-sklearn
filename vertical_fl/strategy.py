import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.linear_model import LogisticRegression
import numpy as np



class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression(max_iter=1000)
        self.labels = np.array(labels).reshape(-1, 1)

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]
        embeddings_aggregated = np.concatenate(embedding_results, axis=1)
        
        # Train logistic regression on aggregated embeddings
        self.model.fit(embeddings_aggregated.T, self.labels.ravel())

        # Extract updated parameters
        parameters_aggregated = ndarrays_to_parameters(
            [self.model.coef_.flatten(), self.model.intercept_]
        )

        # Evaluate accuracy
        predictions = self.model.predict(embeddings_aggregated.T)
        correct = np.sum(predictions.reshape(-1, 1) == self.labels)
        accuracy = correct / len(self.labels) * 100

        metrics_aggregated = {"accuracy": accuracy}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        return None, {}
