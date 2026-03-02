from os import walk

import torch

from torch.nn.utils import clip_grad_norm_
from torch import GradScaler, autocast

scaler = GradScaler()


def train(model, optimizer, loss_function, num_epochs, train_loader, val_loader, return_raw_graph_data=False, patience=999999, save_model=False, model_name='unknown', plot_learning_curve=True, ylim=[0, 0.05]):
    best_val = float('inf')
    patience_ctr = 0
    train_history, val_history = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            batch = batch.to('cuda')
            optimizer.zero_grad()

            # preds = model(batch.x, batch.pos, batch.batch)

            targets = batch.y

            with autocast(device_type='cuda', dtype=torch.float16):
                preds = model(batch.x, batch.pos, batch.batch)
                loss_train = loss_function(preds, targets)

            # loss = loss_function(preds, targets)
            loss_train.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss_sum += loss_train.item() * targets.size(0)
            train_samples += targets.size(0)

        train_loss = train_loss_sum / train_samples
        train_history.append(train_loss)

        model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to('cuda')
                preds = model(batch.x, batch.pos, batch.batch)

                targets = batch.y

                loss_eval = loss_function(preds, targets)
                val_loss_sum += loss_eval.item() * targets.size(0)
                val_samples += targets.size(0)

        val_loss = val_loss_sum / val_samples
        val_history.append(val_loss)

        print(f'Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_state)
    if save_model:
        torch.save(model.state_dict(), f"trained_models/{model_name}.pt")

    if plot_learning_curve:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

        plt.plot(train_history, label='Training Loss')
        plt.plot(val_history, label='Validation Loss')

        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Loss', fontsize=15)

        plt.ylim(ylim)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.legend(fontsize=15)
        plt.grid(True)
        plt.show()

    if return_raw_graph_data:
        return train_history, val_history