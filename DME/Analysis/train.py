import math
import copy
import torch
import logging
from torch.optim.lr_scheduler import StepLR
from pyro.infer import TraceMeanField_ELBO
from tqdm.notebook import tqdm, tnrange
from DME.Analysis.src.model import DMEGP
from DME.Analysis.src.means import MLP, LSTM, GRU
from DME.Analysis.src.data_loader import fetch_dataloaders
from DME.Analysis.src.utils import get_mean_params, get_kernel_params, init_weights


def train_single_epoch(model, dataloader, mean_optim, gp_optim, loss_fn_fixed,
                       loss_fn, inner_lr, n_adapt, use_full_train=True):
    loss = 0.
    count = 0
    for (train_X, train_y, train_Z, n) in dataloader:
        train_X = train_X[0]
        train_y = train_y.view(-1, 1)
        train_X_size = train_X.size(0)
        train_Z = train_Z[0]
        if use_full_train:
            loss_step = model.step(train_X, train_y, train_Z,
                                   mean_optim, gp_optim, loss_fn_fixed, loss_fn,
                                   inner_lr, n_adapt)
        else:
            loss_step = model.step(train_X[:train_X_size-n, :],
                                   train_y[:train_X_size-n],
                                   train_Z[:train_X_size-n, :],
                                   mean_optim, gp_optim, loss_fn_fixed, loss_fn,
                                   inner_lr, n_adapt)
        loss += loss_step
        count += 1
    loss /= count
    return loss


def evaluate(model, dataloader, loss_fn, n_adapt, inner_lr, val=True,
             interpolation=False):
    rmse, loss = 0., 0.
    count = 0
    if interpolation:
        for (X, y, Z, n_test) in dataloader:
            # sequential predictions using GP
            X = X[0]
            y = y.view(-1)
            Z = Z[0]
            X_size = X.size(0)
            for t in reversed(range(1, n_test+1)):
                # define train data and test data
                train_y, train_Z = y[:X_size-t], Z[:X_size-t, :]
                test_X, test_y, test_Z = X[X_size-t:X_size-t+1, :], y[X_size-t:X_size-t+1], Z[X_size-t:X_size-t+1, :]
                # model prediction
                F_pred = model.forward_mean(test_X)
                pred, pred_var, loss_step = model(train_y, train_Z, test_Z,
                                                  loss_fn, inner_lr,
                                                  n_adapt)
                loss += loss_step
                rmse += (F_pred + pred.item() - test_y.item())**2 # sum fixed and GP predictions
                count += 1
    else:
        for (X, y, Z, _) in dataloader:
            # sequential predictions using GP
            X = X[0]
            y = y.view(-1)
            Z = Z[0]
            for t in range(1, X.size(0)):
                # define train data and test data
                train_y, train_Z = y[:t], Z[:t, :]
                test_X, test_y, test_Z = X[t:t+1, :], y[t:t+1], Z[t:t+1, :]
                # model prediction
                F_pred = model.forward_mean(test_X)
                pred, pred_var, loss_step = model(train_y, train_Z, test_Z,
                                                    loss_fn, inner_lr,
                                                    n_adapt)
                loss += loss_step
                rmse += (F_pred + pred.item() - test_y.item())**2 # sum fixed and GP predictions
                count += 1
    rmse /= count
    loss /= count
    rmse = math.sqrt(rmse)
    if val:
        logging.info('val'.upper() + "_score : {:05.3f}".format(rmse))
        logging.info('val'.upper() + "_loss : {:05.3f}".format(loss))
    return rmse, loss


def train_and_evaluate(model, dataloaders, mean_optim, gp_optim, loss_fn_fixed, loss_fn,
                       n_epochs, n_adapt, inner_lr, scheduler):
    dl_train = dataloaders[0]
    dl_val = dataloaders[1]

    best_val_err = float('inf')
    best_state = None
    with tqdm(total=n_epochs) as td:
        for i in range(n_epochs):
            loss = train_single_epoch(model, dl_train, mean_optim, gp_optim,
                                      loss_fn_fixed, loss_fn, inner_lr, n_adapt)
            error_val, loss_val = evaluate(model, dl_val, loss_fn, n_adapt,
                                           inner_lr)
            is_best = error_val <= best_val_err
            if is_best:
                best_val_err = error_val
                best_state = copy.deepcopy(model.state_dict())
            td.set_postfix(loss_and_val_err='{:05.3f} and {:05.3f}'.format(
                loss, error_val))
            td.update()
            scheduler.step()
    print('Best validation error: ', best_val_err)
    return best_state, best_val_err

def train_and_evaluate_model(model_config, train_config, dataframes,
                             random_effects_column_names,
                             group_column_name, y_column_name,
                             n_samples_chosen_per_group,
                             model_type='MLP',
                             random_state=1):
    torch.manual_seed(random_state)

    # data configuration
    dataloaders = fetch_dataloaders(dataframes, random_effects_column_names,
                                    group_column_name,y_column_name,
                                    n_samples_chosen_per_group)

    # model configuration
    input_dim = model_config['input_dim']
    hidden_dim = model_config['hidden_dim']
    output_dim = model_config['output_dim']
    if model_type == 'MLP':
        mean_fn = MLP(input_dim, hidden_dim, output_dim)
    elif model_type == 'LSTM':
        mean_fn = LSTM(input_dim, hidden_dim, output_dim)
    elif model_type == 'GRU':
        mean_fn = GRU(input_dim, hidden_dim, output_dim)
    else:
        raise Exception('Model type is invalid')
    mean_fn.apply(init_weights)
    model = DMEGP(input_dim,
                  random_effects_dim=len(random_effects_column_names),
                  mean_fn=mean_fn)

    # training configuration
    n_epochs = train_config['n_epochs']
    lr = train_config['lr']
    n_adapt = train_config['n_adapt']
    inner_lr = train_config['inner_lr']
    l2_penalty = train_config['l2_penalty']
    mean_optim = torch.optim.Adam(mean_fn.parameters(),
                                  lr=lr,
                                  weight_decay=l2_penalty)
    gp_optim = torch.optim.Adam(get_kernel_params(model.gp_model),
                                lr=lr,
                                weight_decay=l2_penalty)
    scheduler = StepLR(mean_optim, step_size=2, gamma=0.9) # Learning rate decay
    elbo = TraceMeanField_ELBO()
    loss_fn = elbo.differentiable_loss
    loss_fn_fixed = torch.nn.MSELoss()

    # train and evaluate
    best_state, best_val_err = train_and_evaluate(model, dataloaders, mean_optim,
                                                  gp_optim, loss_fn_fixed, loss_fn, n_epochs,
                                                  n_adapt, inner_lr, scheduler)
    return best_state, best_val_err

def train_and_test_model(dataframes, model_config, train_config,
                         random_effects_column_names,
                         group_column_name, y_column_name,
                         n_samples_chosen_per_group,
                         model_type='MLP',
                         random_state=1):
    torch.manual_seed(random_state)

    # data configuration
    dataloaders = fetch_dataloaders(dataframes, random_effects_column_names,
                                    group_column_name, y_column_name,
                                    n_samples_chosen_per_group)
    dataloader_train = dataloaders[0]
    dataloader_test1 = dataloaders[1] # extrapolation
    dataloader_test2 = dataloaders[0] # interpolation

    # model configuration
    input_dim = model_config['input_dim']
    hidden_dim = model_config['hidden_dim']
    output_dim = model_config['output_dim']
    if model_type == 'MLP':
        mean_fn = MLP(input_dim, hidden_dim, output_dim)
    elif model_type == 'LSTM':
        mean_fn = LSTM(input_dim, hidden_dim, output_dim)
    elif model_type == 'GRU':
        mean_fn = GRU(input_dim, hidden_dim, output_dim)
    else:
        raise Exception('Model type is invalid')
    mean_fn.apply(init_weights)
    model = DMEGP(input_dim,
                  random_effects_dim=len(random_effects_column_names),
                  mean_fn=mean_fn)

    # training configuration
    n_epochs = train_config['n_epochs']
    lr = train_config['lr']
    n_adapt = train_config['n_adapt']
    inner_lr = train_config['inner_lr']
    l2_penalty = train_config['l2_penalty']
    mean_optim = torch.optim.Adam(mean_fn.parameters(),
                                  lr=lr,
                                  weight_decay=l2_penalty)
    gp_optim = torch.optim.Adam(get_kernel_params(model.gp_model),
                                lr=lr,
                                weight_decay=l2_penalty)
    scheduler = StepLR(mean_optim, step_size=2, gamma=0.9) # Learning rate decay
    elbo = TraceMeanField_ELBO()
    loss_fn = elbo.differentiable_loss
    loss_fn_fixed = torch.nn.MSELoss()

    torch.manual_seed(random_state)
    # train
    for i in tnrange(n_epochs):
        loss = train_single_epoch(model, dataloader_train, mean_optim, gp_optim,
                                  loss_fn_fixed, loss_fn, inner_lr, n_adapt,
                                  use_full_train=False)
        scheduler.step()

    torch.manual_seed(random_state)
    # test
    rmse_test1, loss_test1 = evaluate(model, dataloader_test1, loss_fn, n_adapt,
                                      inner_lr, val=False)

    rmse_test2, loss_test2 = evaluate(model, dataloader_test2, loss_fn, n_adapt,
                                      inner_lr, val=False, interpolation=True)

    print('[Test (Extrapolation)] RMSE={:05.3f} and loss={:05.3f}'.format(rmse_test1, loss_test1))
    print('[Test (Interpolation)] RMSE={:05.3f} and loss={:05.3f}'.format(rmse_test2, loss_test2))
    return rmse_test1, rmse_test2
