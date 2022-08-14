# -*- coding:utf-8 -*-

import numpy as np
import torch

def mape(y_true, y_pred):
    '''
    Parameters
    ----------------------
    y_true: (B, N, ,F, T)
    y_pred: (B, N, ,F, T)
    Returns
    ----------------------
    float
    '''
    y_true_ = y_true + np.float32(y_true == 0)
    mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                  y_true_))
    return mape.mean()

def mae(y_true, y_pred):
    '''
    Parameters
    ----------------------
    y_true: (B, N, ,F, T)
    y_pred: (B, N, ,F, T)
    Returns
    ----------------------
    float
    '''
    mae = np.abs(np.subtract(y_pred, y_true).astype('float32'))
    return np.mean(mae)

def rmse(y_true, y_pred):
    '''
    Parameters
    ----------------------
    y_true: (B, N, ,F, T)
    y_pred: (B, N, ,F, T)
    Returns
    ----------------------
    float
    '''
    mse = ((y_pred - y_true)**2)
    return np.mean(np.sqrt(np.mean(mse)))


# preds,labels:(B,N,F,T)
# loss except road related to virtual inter
# mask_matrix:
#     0:virtual node
#     1:unmasked node
#     2:masked node


def masked_mse(preds, labels, mask_matrix, type):
    length = len(labels.shape)
    mask = torch.ones_like(labels)
    for idx, i in enumerate(mask_matrix):
        if type == 'train':
            # only compute observable node
            if i == 0 or i == 2:
                if length == 4:
                    mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
                else:
                    mask[idx, :] = torch.zeros_like(mask[idx, :])
        else:
            if i == 0:
                if length == 4:
                    mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
                else:
                    mask[idx, :] = torch.zeros_like(mask[idx, :])
    mask = mask.float()
    loss = (preds - labels)**2
    loss = loss * mask
    return torch.mean(loss)


def masked_mae_test(y_true, y_pred, data_shape, mask_matrix):
    '''calculate mae on test dataset'''
    mask = np.ones(data_shape)
    for idx, i in enumerate(mask_matrix):
        if i == 0:
            if len(data_shape) == 3:
                mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
            else:
                mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
    mask = mask.astype('float32')
    mae = np.abs(np.subtract(y_pred, y_true).astype('float32'))
    mae = mask * mae
    return np.mean(mae)


def masked_rmse_test(y_true, y_pred, data_shape, mask_matrix):
    '''calculate rmse on test dataset'''
    mask = np.ones(data_shape)
    for idx, i in enumerate(mask_matrix):
        if i == 0:
            if len(data_shape) == 3:
                mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
            else:
                mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
    mask = mask.astype('float32')
    mse = ((y_pred - y_true)**2)
    mse = mask * mse
    return np.sqrt(np.mean(mse))


def masked_mape_test(y_true, y_pred, data_shape, mask_matrix):
    '''calculate mape on test dataset'''
    y_true_ = y_true + np.float32(y_true == 0)
    mask = np.ones(data_shape)
    for idx, i in enumerate(mask_matrix):
        if i == 0:
            if len(data_shape) == 3:
                mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
            else:
                mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
    mask = mask.astype('float32')
    mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                  y_true_))
    mape = mask * mape
    return np.mean(mape)