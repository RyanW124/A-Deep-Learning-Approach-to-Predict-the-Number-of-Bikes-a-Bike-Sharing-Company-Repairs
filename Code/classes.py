from re import X
from torch.nn import ModuleList, functional as F
from torch import nn, optim, from_numpy
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm_notebook
from matplotlib import pyplot as plt
import numpy as np, torch
from os.path import isfile
from IPython.display import Image
from scipy import stats

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

def confidence(target, prediction):
    losses = [float(__loss(target[i], prediction[i])) for i in range(len(target))]
    interval = stats.t.interval(0.9, len(losses) - 1, loc=np.mean(losses), scale=stats.sem(losses))
    return (interval[1]-interval[0])/2
def loss_plot(model1, model2, data, file, title, labels):
    predicdion1 = model1.predict(data[0])
    predicdion2 = model2.predict(data[0])
    predicdion1 = from_numpy(predicdion1) if type(predicdion1) == np.ndarray else predicdion1
    predicdion2 = from_numpy(predicdion2) if type(predicdion2) == np.ndarray else predicdion2
    losses = (float(__loss(predicdion1, data[1])), float(__loss(predicdion2, data[1])))
    confidences = (confidence(predicdion1, data[1]), confidence(predicdion2, data[1]))
    index = np.arange(2)
    plt.bar(index, losses, 0.6, alpha=0.8, color='b')
    plt.errorbar(index, losses, yerr=confidences, fmt='o', color='black')
    plt.xticks(index, labels)
    plt.tight_layout()
    plt.title(title)
    plt.ylabel('Loss')
    plt.savefig(file, bbox_inches='tight')
    plt.show()

def kfold(k, x_train, y_train, hidden_layers, epochs, minimum, file=None):
    nns = [NN(len(x_train[0]), len(y_train[0]), hidden_layers) for _ in range(k)]
    epochs = tqdm_notebook(range(1, epochs+1))
    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    x_train2, y_train2 = np.array_split(x_train, k), np.array_split(y_train, k)
    x_train3 = [torch.cat([l for i, l in enumerate(x_train2) if i!=index], 0) for index in range(k)]
    y_train3 = [torch.cat([l for i, l in enumerate(y_train2) if i!=index], 0) for index in range(k)]

    for epoch in epochs:
        train_loss, val_loss = [0]*2
        for index, net in enumerate(nns):
            x_val = x_train2[index]
            y_val = y_train2[index]
            new_t, new_val = net.single_epoch((x_train3[index], y_train3[index]), (x_val, y_val), epoch)
            train_loss += new_t
            val_loss += new_val
        if epoch % 10 == 0:
            train_losses.append(train_loss/k)
            val_losses.append(val_loss/k)
            if epoch >= minimum:
                min_val_loss = min(min_val_loss, val_loss)
                if val_loss > min_val_loss * 1.2:
                    break
    net = NN(len(x_train[0]), len(y_train[0]), hidden_layers)
    if file is None:
        net.learn((x_train, y_train), epochs=epoch, file=None)
    else:
        net.learn((x_train, y_train), epochs=epoch, file=file[:-4]+'_actual.png')
    x = [i * 10 for i in range(len(train_losses))]
    plt.plot(x, train_losses, label='Train Loss')
    plt.plot(x, val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Average Loss Over time")
    plt.ylim([0, max(max(train_losses[len(train_losses)//10:]), max(val_losses[len(train_losses)//10:]))])
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.show()
    if len(y_train[0]) == 10:
        plt.plot(x, train_losses, label='Train Loss')
        plt.plot(x, val_losses, label='Validation Loss')
        plt.legend()
        plt.title("Average Loss Over time")
        plt.ylim([0, .1])
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        if file is not None:
            plt.savefig(file[:-4]+'_zoomed.png', bbox_inches='tight')
            plt.show()
    # return min(nns, key=lambda net: __loss(net(x_train), y_train))
    return net





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

    def single_epoch(self, train, validation, epoch):
        self.zero_grad()
        predictions = self(train[0])
        loss = self.loss_f(predictions, train[1])
        val_predictions = self(validation[0])
        val_loss = self.loss_f(val_predictions, validation[1])
        loss.backward()
        self.optimizer.step()
        if self.normalize and epoch == 1000:
            self.layers.append(self.softmax)
        return float(loss), float(val_loss)

    
    def learn(self, train, *, epochs=5000, file=None):
        epochs = tqdm_notebook(range(1, epochs+1))
        train_losses = []
        for epoch in epochs:
            self.zero_grad()

            predictions = self(train[0])
            loss = self.loss_f(predictions, train[1])
            train_losses.append(float(loss))
            loss.backward()
            self.optimizer.step()
            if self.normalize and epoch == 1000:
                self.layers.append(self.softmax)
        x = range(len(train_losses))
        plt.plot(x, train_losses)
        plt.title("Loss Over time")
        plt.ylim([0, max(train_losses[len(train_losses)//10:])])
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        if file is not None:
            plt.savefig(file, bbox_inches='tight')
            plt.show()
        if self.normalize:
            plt.plot(x, train_losses, label='Train Loss')
            plt.legend()
            plt.title("Loss Over time")
            plt.ylim([0, .1])
            plt.ylabel("Loss")
            plt.xlabel("Epochs")
            if file is not None:
                plt.savefig(file[:-4]+'_zoomed.png', bbox_inches='tight')
                plt.show()

    @classmethod
    def init(cls, input, output, hidden, file, graph, x_train, y_train, epochs, minimum, *, demand=False):
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
            model = kfold(6, x_train, y_train, hidden, epochs, minimum, graph)
            torch.save(model.state_dict(), file)
        return model