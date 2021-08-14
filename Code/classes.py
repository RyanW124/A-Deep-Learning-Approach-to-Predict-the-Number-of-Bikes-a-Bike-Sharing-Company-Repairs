from re import X
from torch.nn import ModuleList, functional as F
from torch import nn, optim, from_numpy
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm_notebook
from matplotlib import pyplot as plt
import numpy as np, torch
from os.path import isfile
from IPython.display import Image

__loss = nn.MSELoss()
load_pt = True

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
        out = F.softmax(from_numpy(out), dim=1)
        return out

def loss_plot(model1, model2, data, file, title, labels, *, x1=0, x2=0, y=1):
    losses = []
    for i in data:
        predicdion1 = model1.predict(i[x1])
        predicdion2 = model2.predict(i[x2])
        predicdion1 = from_numpy(predicdion1) if type(predicdion1) == np.ndarray else predicdion1
        predicdion2 = from_numpy(predicdion2) if type(predicdion2) == np.ndarray else predicdion2
        losses.append((float(__loss(predicdion1, i[y])), float(__loss(predicdion2, i[y]))))
    index = np.arange(2)
    for i, c, l in [(0, 'r', 'Train'), (1, 'g', 'Validation'), (2, 'b', 'Test')]:
        plt.bar(index+i*0.3, losses[i], 0.3, alpha=0.8, color=c, label=l)
    plt.xticks(index + 0.3, labels)
    plt.legend()
    plt.tight_layout()
    plt.title(title)
    plt.ylabel('Loss')
    plt.savefig(file)
    plt.show()

class NN(nn.Module):
    def __init__(self, in_width, out_width, layer_widths, *, add_softmax=False):
        super(NN, self).__init__()
        self.layers = ModuleList()
        self.decoding = ModuleList()
        self.normalize = out_width>1
        self.softmax = nn.Softmax(dim=1)
        self.layers.append(nn.Linear(in_width, layer_widths[0]))
        for i in range(1, len(layer_widths)):
            self.layers.append(nn.Linear(layer_widths[i - 1], layer_widths[i]))
        self.layers.append(nn.Linear(layer_widths[-1], out_width))
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
        if add_softmax:
            self.layers.append(self.softmax)
        self.optimizer = optim.Adam(self.parameters())
        self.mse = nn.MSELoss()
        self.loss_f = nn.MSELoss()
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.layers[-1](x)
        return x

    def predict(self, x):
        return self(x)

    def learn(self, train, validation,*, intervals=10, epochs=5000, file=None, minimum=1300, ylim=None):
        epochs = tqdm_notebook(range(1, epochs+1))
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
            if self.normalize and epoch == 1000:
                self.layers.append(self.softmax)
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

    @classmethod
    def init(cls, input, output, hidden, file, graph, train, val, ylim, epochs, *, demand=False):
        model = NN(input, output, hidden)
        if isfile(file) and load_pt:
            if graph is not None:
                img = Image(filename=graph)
                display(img)
            if demand:
                model = NN(input, output, hidden, add_softmax=True)
            model.load_state_dict(torch.load(file))
            model.eval()
        else:
            model.learn(train, val, file=graph, ylim=ylim, epochs=epochs)
            torch.save(model.state_dict(), file)
        return model