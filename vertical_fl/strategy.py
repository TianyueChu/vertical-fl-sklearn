import flwr as fl
from logging import WARNING
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ClientManager
from sklearn.linear_model import LogisticRegression
import numpy as np
from flwr.common.logger import log


class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression(max_iter=1000)
        self.labels = np.array(labels).reshape(-1, 1)
        self.initial_parameters = [np.random.randn(1, 10), np.random.randn(1)]


    def initialize_parameters(
        self, client_manager: ClientManager
    ):
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return ndarrays_to_parameters(initial_parameters)

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
        feature_results = [parameters_to_ndarrays(fit_res.parameters)[0]
            for _, fit_res in results
        ]

        feature_nums = [parameters_to_ndarrays(fit_res.parameters)[1]
                           for _, fit_res in results
                           ]
        ## [array(9),array(11)]

        feature_aggregated = np.concatenate(feature_results, axis=1)

        # Train logistic regression on aggregated embeddings
        self.model.fit(feature_aggregated, self.labels.ravel())

        # Extract updated parameters
        split_points = np.cumsum(feature_nums)[:-1]
        log(WARNING,f'split_points={split_points}')
        log(WARNING,f'number of coef ={len(self.model.coef_.flatten())}')
        coefs = np.split(self.model.coef_.flatten(), split_points)
        np_coefs = [coef.reshape(1, -1) for coef in coefs]
        parameters_aggregated = ndarrays_to_parameters(np_coefs)


        # Evaluate accuracy
        predictions = self.model.predict(feature_aggregated)
        correct = np.sum(predictions.reshape(-1, 1) == self.labels.reshape(-1, 1))
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
