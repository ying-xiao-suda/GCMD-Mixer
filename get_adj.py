import numpy as np
import torch
import torch.utils.data
from scipy.sparse.linalg import eigs
import pandas as pd
import pickle

def get_adj_pems04():
    A = np.load('./Datasets/pems04/pems04adj.npy')
    A[A != 0] = 1
    N = A.shape[0] 
    I = np.eye(N) 
    A_tilde = A + I 
    D_tilde = np.sum(A_tilde, axis=1)
    D_tilde_inv_sqrt = np.power(D_tilde, -0.5) 
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0  
    D_tilde_inv_sqrt_matrix = np.diag(D_tilde_inv_sqrt)
    DAD = np.dot(D_tilde_inv_sqrt_matrix, np.dot(A_tilde, D_tilde_inv_sqrt_matrix))
    return DAD

def get_adj_pems08():
    A = np.load('./Datasets/pems08/pems08adj.npy')
    A[A != 0] = 1
    N = A.shape[0] 
    I = np.eye(N) 
    A_tilde = A + I 
    D_tilde = np.sum(A_tilde, axis=1)
    D_tilde_inv_sqrt = np.power(D_tilde, -0.5) 
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0  
    D_tilde_inv_sqrt_matrix = np.diag(D_tilde_inv_sqrt)
    DAD = np.dot(D_tilde_inv_sqrt_matrix, np.dot(A_tilde, D_tilde_inv_sqrt_matrix))
    return DAD


def get_adj_shme():
    with open('./Datasets/SHME/graph_sh_conn.pkl', 'rb') as f:
        A = pickle.load(f)
    A[A != 0] = 1
    N = A.shape[0] 
    I = np.eye(N) 
    A_tilde = A + I 
    D_tilde = np.sum(A_tilde, axis=1)
    D_tilde_inv_sqrt = np.power(D_tilde, -0.5) 
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0  
    D_tilde_inv_sqrt_matrix = np.diag(D_tilde_inv_sqrt)
    DAD = np.dot(D_tilde_inv_sqrt_matrix, np.dot(A_tilde, D_tilde_inv_sqrt_matrix))
    return DAD


def get_adj_hb():
    A = np.load('./Datasets/hb/adj.npy')
    A[A != 0] = 1
    N = A.shape[0] 
    I = np.eye(N) 
    A_tilde = A + I 
    D_tilde = np.sum(A_tilde, axis=1)
    D_tilde_inv_sqrt = np.power(D_tilde, -0.5) 
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0  
    D_tilde_inv_sqrt_matrix = np.diag(D_tilde_inv_sqrt)
    DAD = np.dot(D_tilde_inv_sqrt_matrix, np.dot(A_tilde, D_tilde_inv_sqrt_matrix))
    return DAD