from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import torch

from vertical_fl.task import ClientModel, load_data


class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, lr):
        self.v_split_id = v_split_id
        self.data = StandardScaler().fit_transform(data)
        self.model = ClientModel(input_size=self.data.shape[1])
        self.lr = lr

    def get_parameters(self, config):
        coef_, intercept_ = self.model.get_parameters()
        params = [coef_.flatten(), intercept_]
        return fl.common.ndarrays_to_parameters(params)

    def fit(self, parameters, config):
        dummy_labels = np.zeros(self.data.shape[0], dtype=int)
        self.model.fit(self.data, dummy_labels)
        embeddings = self.model.transform(self.data)
        return [embeddings], len(self.data), {}

    def evaluate(self, parameters, config):
        dummy_labels = np.zeros(self.data.shape[0], dtype=int)
        self.model.fit(self.data, dummy_labels)
        return 0.0, len(self.data), {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, partition, lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)