import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kernel(u, v):
    """
    Sparsifying kernel function
    :param u: input vectors [n_in, 1, n_dim]
    :param v: output vectors [1, n_hid, n_dim]
    :return: input to output connection matrix
    """
    # see equation 7 in paper
    dist = torch.norm(u - v, p=None, dim=2)
    l = 1. - torch.square(dist)
    r = torch.tensor([[0.]]).to(device)
    hat = torch.max(l, r)
    return hat


class KernelLayer(nn.Module):
    def __init__(self, input_features, n_hid, n_dim, final):
        super(KernelLayer, self).__init__()
        self.input_features = input_features
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.final = final

        self.W = torch.nn.Parameter(torch.empty(self.input_features, self.n_hid))
        nn.init.xavier_uniform_(self.W)

        self.u = torch.nn.Parameter(torch.empty(self.input_features, 1, self.n_dim))
        nn.init.trunc_normal_(self.u, mean=0., std=1e-3)

        self.v = torch.nn.Parameter(torch.empty(1, self.n_hid, self.n_dim))
        nn.init.trunc_normal_(self.v, mean=0., std=1e-3)

        self.b = torch.nn.Parameter(torch.empty(self.n_hid))
        #nn.init.xavier_uniform_(self.b)
        nn.init.zeros_(self.b)

        self.w_hat = torch.Tensor((input_features, n_hid))


    def forward(self, input):
        self.w_hat = kernel(self.u, self.v)
        W_eff = self.W * self.w_hat

        y = torch.matmul(input, W_eff) + self.b
        if not self.final:
            y = torch.sigmoid_(y)

        return y


class Autoencoder(nn.Module):
    def __init__(self, input_features, n_hid, n_dim):
        super(Autoencoder, self).__init__()

        self.input_features = input_features
        self.n_hid = n_hid
        self.n_dim = n_dim

        self.kl1 = KernelLayer(input_features, n_hid, n_dim, False)
        self.kl2 = KernelLayer(n_hid, n_hid, n_dim, False)
        self.kl3 = KernelLayer(n_hid, input_features, n_dim, True)

        self.optimizer = torch.optim.LBFGS(list(self.parameters()), max_iter=10, lr=0.5)

        self.lambda_s = torch.tensor(0.0013)
        self.lambda_2 = torch.tensor(60)

        self.loss = 0

    def forward(self, x):
        y = self.kl1(x)
        y = self.kl2(y)
        y_pred = self.kl3(y)
        return y_pred

    def calculate_loss(self, y_pred, ratings, mask):
        diff = mask * (ratings - y_pred)
        loss = torch.sum(diff ** 2) / 2

        for l in list(self.children()):
            layer_loss1 = torch.sum(l.w_hat ** 2) / 2 * self.lambda_s
            layer_loss2 = torch.sum(l.W ** 2) / 2 * self.lambda_2
            loss += layer_loss1.item() + layer_loss2.item()

        self.loss = loss.detach().cpu().numpy()

        return loss

    def model_train(self, train_ratings):
        print('train the model for one epoch')

        def closure():
            self.optimizer.zero_grad()

            mask = np.greater(train_ratings, 1e-12).astype('float32')

            train_ratings_tensor = torch.tensor(train_ratings).to(device)
            mask_tensor = torch.tensor(mask).to(device)

            y_pred = self.forward(train_ratings_tensor)
            loss = self.calculate_loss(y_pred, train_ratings_tensor, mask_tensor)

            loss.backward()

            return loss

        self.optimizer.step(closure)

    def model_predict(self, ratings):
        print('return predictions')

        mask = np.greater(ratings, 1e-12).astype('float32')
        ratings = torch.tensor(ratings).to(device)

        y_pred = self.forward(ratings).clone().detach().cpu().numpy()
        y_pred = mask * (np.clip(y_pred, 0.5, 5.))

        return y_pred