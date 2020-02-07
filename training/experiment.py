import numpy as np
import plotly.express as xp
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import typer
import skorch
from skorch.helper import predefined_split
from skorch.dataset import Dataset

import dataget


def main():

    df_train, df_test = dataget.toy.spirals().get()

    X_train = df_train[["x0", "x1"]].to_numpy(dtype=np.float32)
    y_train = df_train["y"].to_numpy()

    X_test = df_test[["x0", "x1"]].to_numpy(dtype=np.float32)
    y_test = df_test["y"].to_numpy()

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
        torch.nn.Softmax(),
    )

    net = skorch.NeuralNetClassifier(
        model,
        batch_size=16,
        max_epochs=100,
        lr=0.01,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        train_split=predefined_split(Dataset(X_test, y_test)),
        device="cuda",
    )

    net.fit(X_train, y_train)

    xx, yy, zz = decision_boundaries(X_train, net)

    fig = xp.scatter(x=X_train[:, 0], y=X_train[:, 1], color=y_train)
    fig.update_traces(marker_line_width=2, marker_size=10)
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=zz, opacity=0.7))
    fig.show()


def decision_boundaries(X, model, n=20):

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    hx = (x_max - x_min) / n
    hy = (y_max - y_min) / n
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max + hx, hx), np.arange(y_min, y_max + hy, hy)
    )

    # Obtain labels for each point in mesh using the model.
    points = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    Z = model.predict(points)
    # Z = (Z > 0.5).astype(np.int32)

    zz = Z.reshape(xx.shape)

    return xx, yy, zz


if __name__ == "__main__":
    typer.run(main)
