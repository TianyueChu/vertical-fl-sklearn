from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from pyexpat import features
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

from vertical_fl.task import  load_data


class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, lr):
        self.v_split_id = v_split_id
        self.data = StandardScaler().fit_transform(data)
        self.model = ClientModel(input_size=self.data.shape[1])
        self.scale = 0.1
        self.beta = np.random.uniform(-self.scale, self.scale, size=self.data.shape[1])
        self.lr = lr

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        result = self.model.multiply(self.data, self.beta)
        print("feature size", self.data.shape[1])
        return [result,self.data.shape[1]], 1, {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.data), {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, partition, lr).to_client()


class ClientModel:
    def __init__(self, input_size):
        """
        Initialize the client model.
        :param input_size: Number of features in the input data.
        """
        self.input_size = input_size

    def multiply(self, data, beta):
        """
        Multiply input data with coefficients beta.
        :param data: Input data.
        :param beta: Coefficients to multiply with the data.
        :return: Result of multiplication.
        """
        if len(beta) != data.shape[1]:
            raise ValueError(f"Expected beta of size {data.shape[1]}, but got {len(beta)}")
        result = np.dot(data, beta)
        return result


app = ClientApp(
    client_fn=client_fn,
)