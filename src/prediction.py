import numpy as np
import torch

def predict_1d_target(model, data_loaders: list):
    model.eval()
    prediction_arrays = []

    for data_loader in data_loaders:
        y_train_pred = []
        y_train_true = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to('cuda')
                outputs = model(batch.x, batch.pos, batch.batch)
                y_train_pred.extend(outputs.cpu().numpy())
                y_train_true.extend(batch.y.cpu().numpy())

            prediction_arrays.append((np.array(y_train_pred), np.array(y_train_true)))

    return prediction_arrays