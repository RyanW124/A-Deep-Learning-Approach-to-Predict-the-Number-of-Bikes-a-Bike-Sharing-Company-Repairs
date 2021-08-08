from torch.nn import ModuleList, functional as F
from torch import nn, optim
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm_notebook
from matplotlib import pyplot as plt
import numpy as np

class MultiOutputLinear:
    def __init__(self):
        self.models = []
    def fit(self, x, y):
        y = y.numpy()
        for i in range(len(y[0])):
            self.models.append(LinearRegression().fit(x, y[:, i]))
    def predict(self, x):
        out = []
        for i in self.models:
            out.append(i.predict(x))
        out = np.array(out)
        out = out.transpose()
        out = [nn.Softmax()(i) for i in out]
        return out

class NN(nn.Module):
    def __init__(self, in_width, out_width, layer_widths):
        super(NN, self).__init__()
        self.layers = ModuleList()
        self.decoding = ModuleList()
        self.normalize = out_width > 1
        self.layers.append(nn.Linear(in_width, layer_widths[0]))
        for i in range(1, len(layer_widths)):
            self.layers.append(nn.Linear(layer_widths[i - 1], layer_widths[i]))
        self.layers.append(nn.Linear(layer_widths[-1], out_width))
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
        self.optimizer = optim.Adam(self.parameters())
        self.mse = nn.MSELoss()
        self.loss_f = nn.MSELoss()
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.layers[-1](x)
        if self.normalize:
            x = nn.Softmax()(x)
        return x
        
    def learn(self, train, validation,*, intervals=10, epochs=5000, file=None, minimum=20000, ylim=None):
        epochs = range(1, epochs+1)
        min_val_loss = float('inf')
        train_losses, val_losses = [], []
        for epoch in epochs:
            self.zero_grad()

            predictions = self(train[0])
            loss = self.loss_f(predictions, train[1])
            if epoch % intervals == 0:
                train_losses.append(float(loss))
                val_predictions = self(validation[0])
                val_loss = self.loss_f(val_predictions, validation[1])
                val_losses.append(float(val_loss))
                if epoch >= minimum:
                    min_val_loss = min(min_val_loss, val_loss)
                    if val_loss > min_val_loss * 1.07:
                        break
            loss.backward()
            self.optimizer.step()
        x = [i * intervals for i in range(len(train_losses))]
        plt.plot(x, train_losses, label='Train Loss')
        plt.plot(x, val_losses, label='Validation Loss')
        plt.legend()
        plt.title("Loss Over time")
        if ylim is not None:
            plt.ylim([0, ylim])
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        if file is not None:
            plt.savefig(file)
        plt.show()
        print(f'Ended at epoch {epoch} with loss of {val_loss}\n{val_loss/min_val_loss}')