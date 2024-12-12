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
            for _, fit_res, _ in results
        ]

        feature_aggregated = np.concatenate(feature_results, axis=1)

        # Train logistic regression on aggregated embeddings
        self.model.fit(feature_aggregated.T, self.labels.ravel())

        # Get the feature number
        feature_nums = [fn for _, _, fn in results]

        # Raise an error if feature_nums is empty
        if not feature_nums:
            raise ValueError(
                f"The feature_nums list is empty: {feature_nums}. Ensure that results contain valid feature numbers.")

        # Extract updated parameters
        coefs = self.model.coef_.flatten().split([feature_nums[0],feature_nums[1], feature_nums[2]], dim=1)
        np_coefs = [coef.numpy() for coef in coefs]
        parameters_aggregated = ndarrays_to_parameters(np_coefs)

        # Evaluate accuracy
        predictions = self.model.predict(feature_aggregated.T)
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
