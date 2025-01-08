import os
import time
import torch
import pickle
import argparse
import numpy as np
from scipy.linalg import svd
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

    return train_data, test_data, train_mat, n_user, m_item, constraint_mat


def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()


def f(train_data, test_data, user_num, item_num, dataset, train_mat, ii_neighbor_num):
     # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    # Compute \Omega to extend UltraGCN to the item-item co-occurrence graph
    ii_cons_mat_path = f'./{dataset}/ii_constraint_mat.pkl'
    ii_neigh_mat_path = f'./{dataset}/ii_neighbor_mat.pkl'
    
    if os.path.exists(ii_cons_mat_path):
        ii_constraint_mat = pload(ii_cons_mat_path)
        ii_neighbor_mat = pload(ii_neigh_mat_path)
    else:
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)
        pstore(ii_neighbor_mat, ii_neigh_mat_path)
        pstore(ii_constraint_mat, ii_cons_mat_path)


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res


def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))


def load_inter(dataset):
    """Load one single adjacent matrix from file
    Args:
        file (string): path of the file to load
    Returns:
        scipy.sparse.coo_matrix: the loaded adjacent matrix
    """
    file = f'./{dataset}/train_mat.pkl'
    with open(file, 'rb') as fs:
        inter = (pickle.load(fs) != 0).astype(np.float32)
    if type(inter) != coo_matrix:
        inter = coo_matrix(inter)
    return inter


def normalize(inter):
    user_degree = np.array(inter.sum(axis=1)).flatten() # Du
    item_degree = np.array(inter.sum(axis=0)).flatten() # Di
    user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
    item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
    norm_inter = (user_d_inv_sqrt @ inter @ item_d_inv_sqrt)
    return norm_inter


def svd_solver():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    # full svd on dense matrix
    parser.add_argument('--full', action='store_true')
    # truncated svd on sparse matrix
    parser.add_argument('--cutoff', type=int)
    parser.add_argument('--rand_seed', type=int, default=2024)
    
    args = parser.parse_args()
    
    inter = load_inter(args.dataset)
    norm_inter = normalize(inter)
    
    start_time = time.time()
    print(f">>> Dataset: {args.dataset}")
    if args.full:
        print(">>> Full SVD")
        u, s, vt = svd(norm_inter.todense(), full_matrices=False, lapack_driver='gesvd')
    else:
        print(f">>> Truncated SVD - Cutoff: {args.cutoff}, Random seed: {args.rand_seed}")
        u, s, vt = svds(norm_inter, which='LM', k=args.cutoff, random_state=args.rand_seed)
    v = vt.T
    print(f">>> Computation for {(time.time() - start_time)/60:.1f} mins")
    
    if not os.path.exists(f'./{args.dataset}/svd'):
        os.makedirs(f'./{args.dataset}/svd')
    cutoff = "full" if args.full else args.cutoff
    np.save(f'./{args.dataset}/svd/u_{cutoff}.npy', u)
    np.save(f'./{args.dataset}/svd/s_{cutoff}.npy', s)
    np.save(f'./{args.dataset}/svd/v_{cutoff}.npy', v)


def main():
    dataset_names = ['gowalla']

    for dataset_name in dataset_names:
        train_file_path = './' + dataset_name + '/raw/train.txt'
        test_file_path  = './' + dataset_name + '/raw/test.txt'

        train_data, test_data, train_mat, n_user, m_item, constraint_mat = load_data(train_file_path, test_file_path)
        pstore(train_mat, f'./{dataset_name}/_train_mat.pkl')
        pstore(constraint_mat, f'./{dataset_name}/constraint_mat.pkl')
        f(train_data, test_data, n_user, m_item, dataset_name, train_mat, ii_neighbor_num=10)


if __name__ == '__main__':
    main()