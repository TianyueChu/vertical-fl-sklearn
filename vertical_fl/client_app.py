import logging

# Initialize logging
from logging import DEBUG, ERROR, WARNING
from flwr.common.logger import log

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArray
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from vertical_fl.task import load_data

class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, lr):
        self.v_split_id = v_split_id
        self.data = StandardScaler().fit_transform(data)
        self.model = ClientModel(input_size=self.data.shape[1])
        self.lr = lr

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        # if the length of the parameters is 2, which means only coef and interception
        if len(parameters) == 2:
            log(WARNING, f"parameters is:{parameters}")
            scale = 0.1
            beta = np.random.uniform(-scale, scale, size=self.data.shape[1])
            result = self.model.multiply(self.data, beta)
        else:
            result = self.model.multiply(self.data, parameters[int(self.v_split_id)])
        log(WARNING,f"Feature size for client {self.v_split_id}: {self.data.shape[1]}")
        return [result, self.data.shape[1]], 1, {}


    def evaluate(self, parameters, config):
        eval_result = 0.0
        data_len = len(self.data)
        return eval_result, data_len, {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
    log(WARNING, f"Client initialized with v_split_id {v_split_id}")
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, partition, lr).to_client()

class ClientModel:
    def __init__(self, input_size):
        self.input_size = input_size

    def multiply(self, data, beta):
        if len(beta) != data.shape[1]:
            raise ValueError(f"Expected beta of size {data.shape[1]}, but got {len(beta)}")
        result = np.dot(data, beta)
        return result

app = ClientApp(
    client_fn=client_fn,
)
