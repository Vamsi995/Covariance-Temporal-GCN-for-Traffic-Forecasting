import torch
from tqdm import tqdm
from dataloader import SpatioTemporalCSVDataModule
from utils import r2_f, explained_variance_f, accuracy_f
import torch.nn.functional as F
from model.tgcn import TGCN, SupervisedForecastTask
import argparse

def validate(trainer, val_loader):
        trainer.model.eval()
        if trainer.regressor is not None:
            trainer.regressor.eval()
        # mean = final_dataset.mean(0).mean(0)
        mean = None
        b, t, n = final_dataset.shape
        C_old = torch.cov(final_dataset.reshape(b*t, n).T)
        val_loss, rmse, mae, r2, explained_var = 0.0, 0.0, 0.0, 0.0, 0.0
        acc = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x, y = batch
                x, y = x.to(trainer.device), y.to(trainer.device)

                # print(x.shape, mean.shape)
                C, C_old, mean = trainer.update_covariance(x, C_old, gamma=trainer.gamma,
                                               mean=mean)

                for graph_layer in [trainer.model.tgcn_cell.graph_conv1, trainer.model.tgcn_cell.graph_conv2]:
                    graph_layer.covariance = C


                predictions = trainer.forward(x)
                predictions = predictions * data_module.feat_max_val

                predictions = predictions.transpose(1, 2).reshape((-1, x.size(2)))  # Reshape predictions
                y = y.reshape((-1, y.size(2)))  # Reshape targets
                y = y * data_module.feat_max_val



                # Compute loss
                loss = trainer.loss(predictions, y)
                val_loss += loss.item()

                # Metrics
                rmse += torch.sqrt(F.mse_loss(predictions, y)).item()
                mae += F.l1_loss(predictions, y).item()
                r2 += r2_f(predictions, y)
                explained_var += explained_variance_f(predictions, y)
                acc += accuracy_f(predictions, y)

        num_batches = len(val_loader)
        print(
            f"Validation Loss: {val_loss / num_batches:.4f}, RMSE: {rmse / num_batches:.4f}, "
            f"MAE: {mae / num_batches:.4f}, R2: {r2 / num_batches:.4f}, Explained Variance: {explained_var / num_batches:.4f}", f"Accuracy: {acc / num_batches:.4f}"
        )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["los_loop", "sz_taxi"])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--weights_path', type=str, default=None)

    args = parser.parse_args()

    if args.weights_path == None:
        raise Exception("Please specify the path to the model weight file to evaluate")
        

    if args.dataset == "los_loop":
        speed_path = "/content/los_speed.csv"
        adj_path = "/content/los_adj.csv"
    else:
        speed_path = "/content/sz_speed.csv"
        adj_path = "/content/sz_adj.csv"

    data_module = SpatioTemporalCSVDataModule(speed_path, adj_path)

    final_dataset = None

    for x, y in data_module.train_dataloader():
        if final_dataset is None:
            final_dataset = x
        else:
            final_dataset = torch.cat([final_dataset, x], dim=0)

    for x, y in data_module.val_dataloader():
        final_dataset = torch.cat([final_dataset, x], dim=0)

    hidden_dim = args.hidden_dim
    model = TGCN(data_module.adj, hidden_dim)
    model.load_state_dict(torch.load(args.weights_path, weights_only=True))
    trainer = SupervisedForecastTask(model, feat_max_val=data_module.feat_max_val)

    validate(trainer, data_module.val_dataloader())


