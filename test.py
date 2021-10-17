import numpy as np
from src.models import kernelized_synaptic_weight_matrices as kswm

i = 0
with open(f'data/cv/cv_train_{i}.npy', 'rb') as f:
    train_ratings = np.load(f)

with open(f'data/cv/cv_test_{i}.npy', 'rb') as f:
    test_ratings = np.load(f)

model = kswm.Autoencoder(input_features=train_ratings.shape[1], n_hid=100, n_dim=5)


model.model_train(test_ratings)
model.loss

y_pred = model.model_predict(test_ratings)
mask = np.greater(test_ratings, 1e-12).astype('float32')


error_test = np.sqrt((mask * (y_pred - test_ratings) ** 2).sum() / mask.sum())  # compute train error
