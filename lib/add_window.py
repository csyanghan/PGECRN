import numpy as np


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def Add_Window_Horizon_3channel(data, window=3, horizon=1, single=False):
    """
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index : index + window])
            Y.append(data[index + window + horizon - 1 : index + window + horizon])
            index = index + 1
    else:
        while index < end_index:
            p_c = data[index : index + window]
            if index - 288 > 0:
                p_d = data[index-288: index+window-288]
            else:
                p_d = p_c
            if index - 288 * 7 > 0:
                p_w = data[index-288*7: index+window-288*7]
            else:
                p_w = p_c
            X_p = np.concatenate([p_c, p_d, p_w], axis=-1)
            X.append(X_p)
            Y.append(data[index + window : index + window + horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


if __name__ == '__main__':
    from data.load_raw_data import Load_Sydney_Demand_Data
    path = '../data/1h_data_new3.csv'
    data = Load_Sydney_Demand_Data(path)
    print(data.shape)
    X, Y = Add_Window_Horizon(data, horizon=2)
    print(X.shape, Y.shape)


