import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


def plot_2_graphs(y_test, prediction_test, y_train=None, prediction_train=None, graph_name=None, direct_path=None, axis_lims = [5, 5, 5, 5]):
    def func(x, k, b):
        return k * x + b

    x_min, x_max = axis_lims[0], axis_lims[1]
    y_min, y_max = axis_lims[2], axis_lims[3]

    y_test_numeric = np.array(y_test, dtype=np.float32)
    prediction_test_numeric = np.array(prediction_test, dtype=np.float32)

    if y_train is not None and prediction_train is not None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
        axs = axs.flatten()
    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        axs = [ax]

    axs[-1].scatter(y_test_numeric, prediction_test_numeric, alpha=0.7, s=3)
    popt, _ = curve_fit(func, y_test_numeric, prediction_test_numeric)
    r2 = r2_score(y_test_numeric, prediction_test_numeric)
    axs[-1].plot(
        y_test_numeric,
        func(y_test_numeric, *popt),
        'r-',
        label=(
            f"R² = {r2:.4f}\n"
            f"y = {popt[0]:.4f}x + {popt[1]:.4f}\n"
            f"RMSE = {root_mean_squared_error(y_test_numeric, prediction_test_numeric):.4f}\n"
            f"MAE = {mean_absolute_error(y_test_numeric, prediction_test_numeric):.4f}"
        )
    )
    axs[-1].set_title("Test data", size=20, weight='bold')

    if y_train is not None and prediction_train is not None:
        y_train_numeric = np.array(y_train, dtype=np.float32)
        prediction_train_numeric = np.array(prediction_train, dtype=np.float32)

        axs[0].scatter(y_train_numeric, prediction_train_numeric, alpha=0.7, s=3)
        popt_train, _ = curve_fit(func, y_train_numeric, prediction_train_numeric)
        r2_train = r2_score(y_train_numeric, prediction_train_numeric)
        axs[0].plot(
            y_train_numeric,
            func(y_train_numeric, *popt_train),
            'r-',
            label=(
                f"R² = {r2_train:.4f}\n"
                f"y = {popt_train[0]:.4f}x + {popt_train[1]:.4f}\n"
                f"RMSE = {root_mean_squared_error(y_train_numeric, prediction_train_numeric):.4f}\n"
                f"MAE = {mean_absolute_error(y_train_numeric, prediction_train_numeric):.4f}"
            )
        )
        axs[0].set_title("Train data", size=20, weight='bold')

    for ax in axs:
        ax.set_xlabel("Расчет B3LYP", fontsize=15, weight='bold')
        ax.set_ylabel("Оценка модели", fontsize=15, weight='bold')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        # ax.legend(fontsize=12, loc='upper left')

    if graph_name:
        save_path = direct_path if direct_path else os.path.join("graphs", graph_name)
        os.makedirs(save_path, exist_ok=True)

        fig.savefig(os.path.join(save_path, f"{graph_name}.png"))

        pd.DataFrame({
            'Ground_truth': y_test_numeric,
            'Model_prediction': prediction_test_numeric
        }).to_csv(os.path.join(save_path, f"{graph_name}_val.csv"), index=False)

        if y_train is not None and prediction_train is not None:
            pd.DataFrame({
                'Ground_truth': y_train_numeric,
                'Model_prediction': prediction_train_numeric
            }).to_csv(os.path.join(save_path, f"{graph_name}_train.csv"), index=False)

    plt.show()

def plot_scatter(x, y, graph_title: str, x_label: str, axis_lims = [5, 5, 5, 5], s=5, grid=False, diag_xy_line=True, graph_name=None):
    x_min, x_max = axis_lims[0], axis_lims[1]
    y_min, y_max = axis_lims[2], axis_lims[3]

    plt.figure(figsize=(10, 10))

    plt.scatter(x, y, s=s)

    if diag_xy_line:
        cord_min = min(axis_lims)
        cord_max = max(axis_lims)

        plt.plot([cord_min, cord_max], [cord_min, cord_max], color='red', linestyle='dashed', linewidth=2)

    plt.title(graph_title, fontsize=25)

    plt.xlabel(x_label, fontsize=20)
    plt.ylabel('Предсказание модели, eV', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.grid(grid)

    if graph_name is not None:
        plt.savefig(graph_name, dpi=300)

    plt.show()

def plot_hist2d(x, y, graph_title, x_label, axis_lims = [5, 5, 5, 5], graph_name=None):
    x_min, x_max = axis_lims[0], axis_lims[1]
    y_min, y_max = axis_lims[2], axis_lims[3]

    plt.figure(figsize=(10, 10))

    # plt.scatter(y_train_true_t, y_train_pred_t, s=5)
    plt.hist2d(x, y,
               bins=100,
               range=[[x_min, x_max], [y_min, y_max]],
               density=True,
               cmap='inferno')

    plt.title(graph_title, fontsize=25)

    plt.xlabel(x_label, fontsize=20)
    plt.ylabel('Оценка модели, eV', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    plt.grid(True)

    if graph_name is not None:
        plt.savefig(graph_name, dpi=300)

    plt.show()
