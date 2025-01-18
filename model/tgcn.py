import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from utils import r2_f, explained_variance_f



def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer("laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self.weights = nn.Parameter(torch.FloatTensor(self._num_gru_units + 1, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()
        self.covariance = None  # Covariance matrix placeholder
        self.laplacian_mask = nn.Parameter(torch.ones_like(self.laplacian))  # Learnable mask for Laplacian
        self.covariance_mask = nn.Parameter(torch.ones_like(self.laplacian))  # Learnable mask for covariance


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)


    def update_covariance(self, hidden_state, gamma=0.9):

        # print(hidden_state.shape)
        """
        Update covariance matrix dynamically using the hidden state and a decay factor gamma.
        """
        batch_size, num_nodes, num_features = hidden_state.shape
        reshaped_hidden = hidden_state.reshape(-1, num_features)  # Flatten to (batch_size * num_nodes, features)
        current_covariance = torch.cov(reshaped_hidden.T)

        # Initialize or update self.covariance with the same shape as current_covariance
        if self.covariance is None:
            self.covariance = torch.zeros_like(current_covariance)

        # Update self.covariance using the current covariance and decay factor
        # print(hidden_state.shape, self.covariance.shape, current_covariance.shape)
        self.covariance = (gamma * self.covariance + (1 - gamma) * current_covariance) / torch.trace(self.covariance)
        # assert not torch.isnan(self.covariance).any()
        print(self.covariance)
        self.covariance = torch.nan_to_num(self.covariance)

        # Normalize by trace to keep the scale stable
        # self.covariance = self.covariance / torch.trace(self.covariance)

        # if self.covariance is None:
        #     self.covariance = current_covariance
        # else:
        #     self.covariance = gamma * self.covariance + (1 - gamma) * current_covariance

        # self.covariance = self.covariance / torch.trace(self.covariance)  # Trace-normalize
    # Apply the covariance filter
        # self.filtered_covariance, _ = self.covariance_filter(self.covariance, self.top_k)

    def covariance_filter(self, covariance_matrix, top_k):
        """
        Apply a covariance filter by selecting the top-k eigenvectors as a mask.

        Args:
            covariance_matrix (torch.Tensor): Covariance matrix (N, N).
            top_k (int): Number of top eigenvectors to select as the mask.

        Returns:
            torch.Tensor: Filtered covariance matrix.
            torch.Tensor: Selected top eigenvectors.
        """
        # Ensure the covariance matrix is symmetric
        assert torch.allclose(covariance_matrix, covariance_matrix.T), "Matrix must be symmetric."

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        top_eigenvectors = eigenvectors[:, sorted_indices[:top_k]]

        # Project the covariance matrix using the top eigenvectors
        filtered_matrix = top_eigenvectors @ top_eigenvectors.T

        return filtered_matrix, top_eigenvectors


    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))

        # Update covariance using the current hidden state
        # self.update_covariance(hidden_state)

        masked_laplacian = self.laplacian * self.laplacian_mask
        masked_covariance = self.covariance * self.covariance_mask

        # Combine Laplacian and masked covariance as GSO
        mask = masked_laplacian + masked_covariance

        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        concatenation = concatenation.reshape((num_nodes, (self._num_gru_units + 1) * batch_size))
        a_times_concat = (mask/torch.linalg.matrix_norm(mask)) @ concatenation
        a_times_concat = a_times_concat.reshape((num_nodes, self._num_gru_units + 1, batch_size))
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self._num_gru_units + 1))
        outputs = a_times_concat @ self.weights + self.biases
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }



class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}



class SupervisedForecastTask:
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 12,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        gamma: float = 0.9,  # Covariance update factor
        **kwargs
    ):
        self.model = model
        self.gamma = gamma
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.covariance = None  # Initialize covariance matrix

        self.model.to(self.device)
        if self.regressor is not None:
            self.regressor.to(self.device)

        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.regressor.parameters())
            if self.regressor is not None
            else self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def forward(self, x):
        batch_size, _, num_nodes = x.size()
        hidden = self.model(x)  # (batch_size, num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))  # (batch_size * num_nodes, hidden_dim)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))  # (batch_size, num_nodes, pre_len)
        return predictions


    def update_covariance(self, x, C_old, gamma, mean):

        b, t, n = x.shape

        if mean is not None:
            C_new = torch.matmul((x - mean).T,x - mean) / x.shape[0]
            if C_old is not None:
                mean = (1-gamma)*mean + gamma * x.mean(0)
                Craw = self.gamma * C_new + (1-gamma) * C_old
            else: # First pass, no C_old
                mean = x.mean(0)
                Craw = C_new
        else:
            if C_old is not None:
                Craw = gamma * torch.cov(x.reshape(b*t, n).T) + (1-gamma) * C_old
            else: # First pass, no C_old
                Craw = torch.cov(x.reshape(b * t, n).T)
                # print(Craw)

        if torch.trace(Craw) != 0:
          C = Craw / torch.trace(Craw)
        else:
          C = Craw

        return C, Craw, mean

    def train(self, train_loader, val_loader, epochs):
        self.model.train()
        if self.regressor is not None:
            self.regressor.train()

        for epoch in range(epochs):
            train_loss = 0.0
            C_old = None
            mean = None
            tot_train_loss = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                # if hasattr(self.model, "tgcn_cell"):
                C, C_old, mean = self.update_covariance(x, C_old, gamma=self.gamma,
                                               mean=mean)

                for graph_layer in [self.model.tgcn_cell.graph_conv1, self.model.tgcn_cell.graph_conv2]:
                    graph_layer.covariance = C

                # Forward pass
                predictions = self.forward(x)

                # Update covariance matrix dynamically
                predictions = predictions.transpose(1, 2).reshape((-1, x.size(2)))  # Reshape predictions
                y = y.reshape((-1, y.size(2)))  # Reshape targets

                # Compute loss
                loss = self.loss(predictions, y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}")

    def validate(self, val_loader):
        # Validation logic remains the same
        self.model.eval()
        if self.regressor is not None:
            self.regressor.eval()

        val_loss, rmse, mae, r2, explained_var = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                predictions = self.forward(x)
                predictions = predictions * self.feat_max_val

                predictions = predictions.transpose(1, 2).reshape((-1, x.size(2)))  # Reshape predictions
                y = y.reshape((-1, y.size(2)))  # Reshape targets
                y = y * self.feat_max_val



                # Compute loss
                loss = self.loss(predictions, y)
                val_loss += loss.item()

                # Metrics
                rmse += torch.sqrt(F.mse_loss(predictions, y)).item()
                mae += F.l1_loss(predictions, y).item()
                r2 += r2_f(predictions, y)
                explained_var += explained_variance_f(predictions, y)

        num_batches = len(val_loader)
        print(
            f"Validation Loss: {val_loss / num_batches:.4f}, RMSE: {rmse / num_batches:.4f}, "
            f"MAE: {mae / num_batches:.4f}, R2: {r2 / num_batches:.4f}, Explained Variance: {explained_var / num_batches:.4f}"
        )

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        raise ValueError(f"Loss '{self._loss}' not supported")

