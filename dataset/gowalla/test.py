import pickle


path = './train_mat.pkl'

with open(path, 'rb') as f:
    res = pickle.load(f)
print('load path = {} object'.format(path))

print(type(res))