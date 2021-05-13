import os
import pandas as pd
import numpy as np
from collections import defaultdict

def read_data(filename, test=False):
    data, x = [], []
    for line in open(filename, "r"):
        line = line.strip()
        if not line:
            ID, seq, dot = x[:3]
            if test:
                x = {"id": ID,
                     "sequence": seq,
                     "structure": dot,
                }
                data.append(x)
                x = []
                continue
            punp = x[3:]
            punp = [punp_line.split() for punp_line in punp]
            punp = [(float(p)) for i, p in punp]
            x = {"id": ID,
                 "sequence": seq,
                 "structure": dot,
                 "p_unpaired": punp,
            }
            data.append(x)
            x = []
        else:
            x.append(line)
    return data

def load_linear_rna(dtype):
    data = pd.read_csv(f'./data/linear_rna_{dtype}.csv', index_col=0)
    return [np.array(row)[np.array(row) == np.array(row)] for index, row in data.iterrows()]

def read_data2(filename, dtype, test=False):
    data, x = [], []
    for line in open(filename, "r"):
        line = line.strip()
        if not line:
            ID, seq, dot = x[:3]
            if test:
                x = {"id": ID,
                     "sequence": seq,
                     "structure": dot,
                     "linear_rna": None,
                }
                data.append(x)
                x = []
                continue
            punp = x[3:]
            punp = [punp_line.split() for punp_line in punp]
            punp = [(float(p)) for i, p in punp]
            x = {"id": ID,
                 "sequence": seq,
                 "structure": dot,
                 "linear_rna": None,
                 "p_unpaired": punp,
            }
            data.append(x)
            x = []
        else:
            x.append(line)

    lens = {'train': 4750, 'val': 250, 'test': 112}
    train = load_linear_rna('train')
    val = load_linear_rna('val')
    test = load_linear_rna('test')
    linear_rna = {'train': train, 'val': val, 'test': test}

    for i in range(lens[dtype]):
        data[i]["linear_rna"] = linear_rna[dtype][i]
    return data



def load_train_data():
    assert os.path.exists("./data/train.txt")
    assert os.path.exists("./data/dev.txt")
    train = read_data2("./data/train.txt", 'train')
    dev = read_data2("./data/dev.txt", 'val')
    return train, dev

def load_test_data():
    assert os.path.exists("./data/B_board_112_seqs.txt")
    test = read_data2("./data/B_board_112_seqs.txt", 'test', test=True)
    return test

# def load_test_label_data():
#     assert os.path.exists("data/test.txt")
#     test = read_data("data/test.txt")
#     return test

if __name__ == '__main__':
    train, val = load_train_data()
    trainall = train + val
    test = load_test_data()
    print(trainall.__len__())