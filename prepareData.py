import pickle
import numpy as np
import random
import argparse
import configparser
import os
import torch
from common.metrics import mae, rmse, mape

def normalization(train, val, test):
    '''Normalize train, val, test dataset'''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)

    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return np.nan_to_num((x - mean) / std)

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


# def load_graphdata_channel(graph_signal_matrix_filename, DEVICE, batch_size, shuffle=False):
#     '''
#     这个是为PEMS的数据准备的函数
#     将x,y都处理成归一化到[-1,1]之前的数据;
#     每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
#     该函数会把mhalf, day, week的时间串起来；
#     注: 从文件读入的数据，x是最大最小归一化的，但是y是真实值
#     这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
#     Parameters
#     --------------------------
#     graph_signal_matrix_filename: str, graph_signal_matrix_filename = ./roadgraph/hz/state_4x4.pkl
#     num_of_mhalf: int
#     DEVICE:
#     batch_size: int
#     Returns
#     --------------------------
#     three DataLoaders, each dataloader contains:
#     test_x_tensor: (B, N_nodes, in_feature, T_input)
#     test_decoder_input_tensor: (B, N_nodes, T_output)
#     test_target_tensor: (B, N_nodes, T_output)

#     '''
#     print('load file:', graph_signal_matrix_filename)
#     pkl_file = open(graph_signal_matrix_filename, 'rb')
#     file_data = pickle.load(pkl_file)

#     # mask:0 means random feature
#     mask_matrix = file_data['node_update']
#     mask_or = file_data['mask_or']

#     train_x = file_data['train_x']  # (396, 80, 11, 30)
#     # train_x_phase = file_data['train_x_phase']
#     train_target = file_data['train_target'][:, :, :3]  # (396, 80, 3,1)

#     val_x = file_data['val_x']
#     # val_x_phase = file_data['val_x_phase']
#     val_target = file_data['val_target'][:, :, :3]

#     test_x = file_data['test_x']
#     # test_x_phase = file_data['test_x_phase']
#     test_target = file_data['test_target'][:, :, :3]

#     # mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
#     # std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)
#     mean = file_data['mean']  # (1, 1, 3, 1)
#     std = file_data['std']  # (1, 1, 3, 1)

#     # tmp_train = re_normalization(train_x,mean,std)

#     # ------- train_loader -------
#     train_x_tensor = torch.from_numpy(train_x).type(
#         torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     train_target_tensor = torch.from_numpy(train_target).type(
#         torch.FloatTensor).to(DEVICE)  # (B, N, F, T)

#     train_dataset = torch.utils.data.TensorDataset(
#         train_x_tensor, train_target_tensor)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=shuffle)

#     # ------- val_loader -------
#     val_x_tensor = torch.from_numpy(val_x).type(
#         torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     val_target_tensor = torch.from_numpy(val_target).type(
#         torch.FloatTensor).to(DEVICE)  # (B, N, F, T)

#     val_dataset = torch.utils.data.TensorDataset(
#         val_x_tensor, val_target_tensor)

#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False)

#     # ------- test_loader -------
#     test_x_tensor = torch.from_numpy(test_x).type(
#         torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
#     test_target_tensor = torch.from_numpy(test_target).type(
#         torch.FloatTensor).to(DEVICE)  # (B, N, F, T)

#     test_dataset = torch.utils.data.TensorDataset(
#         test_x_tensor, test_target_tensor)

#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False)

#     # print
#     print('train:', train_x_tensor.size(), train_target_tensor.size())
#     print('val:', val_x_tensor.size(), val_target_tensor.size())
#     print('test:', test_x_tensor.size(), test_target_tensor.size())
    
#     return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std, mask_matrix, mask_or


def phase_to_onehot(phase, num_class):
    assert num_class > phase.max()
    one_hot = np.zeros((1, num_class))
    one_hot[-1][phase.reshape(-1)] = 1.
    # one_hot = one_hot.reshape(*phase.shape, num_class)
    return one_hot


def onehot_to_phase(phase):
    '''reconstruct one hot phase to direction-level phase'''
    phase_dic = {
        0: ['WS', 'ES'],
        1: ['NS', 'SS'],
        2: ['WL', 'EL'],
        3: ['NL', 'SL'],
        4: ['WS', 'WL'],
        5: ['ES', 'EL'],
        6: ['NS', 'NL'],
        7: ['SS', 'SL']
    }
    # phase:[B,N,8,T]->[B,T,N,8]
    phase = np.transpose(phase, (0, 3, 1, 2))
    batch_size, num_of_timesteps, num_of_vertices, _ = phase.shape
    phase_more = np.full((batch_size, num_of_timesteps,num_of_vertices, 2, 2), ['XX', 'XX'])
    # idx must euqals to B*T*N
    idx = np.argwhere(phase == 1)
    assert len(idx) == batch_size*num_of_timesteps*num_of_vertices
    for x in idx:
        phase_more[x[0], x[1], x[2]] = np.array(phase_dic[x[3]])
    return phase_more


def phases_to_onehot(phase, num_class):
    assert num_class > phase.max()
    one_hot = np.zeros((phase.shape[0], num_class))
    one_hot[range(0, phase.shape[0]), phase.squeeze()] = 1
    # one_hot = one_hot.reshape(*phase.shape, num_class)
    return one_hot


def get_road_adj(filename):
    with open(filename, 'rb') as fo:
        result = pickle.load(fo)
    road_info = result['road_links']
    road_dict_road2id = result['road_dict_road2id']
    num_roads = len(result['road_dict_id2road'])
    adjacency_matrix = np.zeros(
        (int(num_roads), int(num_roads)), dtype=np.float32)
    '''
    "PHASE": [
        'WSES':0
        'NSSS':1
        'WLEL':2
        'NLSL':3
        'WSWL':4
        'ESEL':5
        'NSNL':6
        'SSSL':7
    ],
    '''

    adj_phase = np.full(
        (int(num_roads), int(num_roads)), 'XX')

    for inter_dic in road_info:
        for link_dic in road_info[inter_dic]:
            source = link_dic[0]
            target = link_dic[1]
            type_p = link_dic[2]
            direction = link_dic[3]

            if type_p == 'go_straight':
                if direction == 0:
                    # adj_phase.append([0,1])
                    # adj_phase.append('WS')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WS'
                elif direction == 1:
                    # adj_phase.append([1,1])
                    # adj_phase.append('SS')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SS'
                elif direction == 2:
                    # adj_phase.append([2,1])
                    # adj_phase.append('ES')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'ES'
                else:
                    # adj_phase.append([3,1])
                    # adj_phase.append('NS')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NS'

            elif type_p == 'turn_left':
                if direction == 0:
                    # adj_phase.append([0,0])
                    # adj_phase.append('WL')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WL'
                elif direction == 1:
                    # adj_phase.append([1,0])
                    # adj_phase.append('SL')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SL'
                elif direction == 2:
                    # adj_phase.append([2,0])
                    # adj_phase.append('EL')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'EL'
                else:
                    # adj_phase.append([3,0])
                    # adj_phase.append('NL')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NL'
            else:
                if direction == 0:
                    # adj_phase.append([0,2])
                    # adj_phase.append('WR')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'WR'
                elif direction == 1:
                    # adj_phase.append([1,2])
                    # adj_phase.append('SR')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'SR'
                elif direction == 2:
                    # adj_phase.append([2,2])
                    # adj_phase.append('ER')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'ER'
                else:
                    # adj_phase.append([3,2])
                    # adj_phase.append('NR')
                    adj_phase[road_dict_road2id[source]
                              ][road_dict_road2id[target]] = 'NR'

            adjacency_matrix[road_dict_road2id[source]
                             ][road_dict_road2id[target]] = 1

    return adjacency_matrix, adj_phase


def build_road_state(relation_file, state_file, neighbor_node, mask_num, save_dir):
    '''
    generate data to train prediction model
    Parameters
    -----------------------------
    relation_file: string, the file path of roadnetwork relation
    state_file: string, the file path of traffic states
    neighbor_node:int, select nodes with the specified number of neighbors to mask
    mask_num: int, the num of mask nodes 
    save_dir:string, save path of generated data
    '''

    with open(relation_file, 'rb') as f_re:
        relation = pickle.load(f_re)
    inter_dict_id2inter = relation['inter_dict_id2inter']
    inter_in_roads = relation['inter_in_roads']
    road_dict_road2id = relation['road_dict_road2id']
    num_roads = len(road_dict_road2id)
    net_shape = relation['net_shape']
    neighbor_num = relation['neighbor_num']
    mask_inter = random.sample(neighbor_num[int(neighbor_node)], int(mask_num)) # idx of mask intersections
    # mask_inter = [6,9]
    mask_or = {} # ids of mask intersections
    for i in mask_inter:
        mask_or[i] = inter_dict_id2inter[i]

    adj_road, _ = get_road_adj(relation_file)
    # road_update:0:roads related to virtual inter,1:unmasked,2:masked
    road_update = np.zeros(int(num_roads), dtype=np.int32)

    all_road_feature = []

    for state_dic in state_file:
        with open(state_dic, "rb") as f_ob:
            state = pickle.load(f_ob)
        # only update 64 road_node,
        # It has 16 roads'states which start_intersection is virtual are not known.

        # add phase:road_feature(,,11)
        # no phase: road_feature(,,3)
        road_feature = np.zeros((len(state), int(num_roads), 11), dtype=np.float32)

        for id_time, step_dict in enumerate(state):
            for id_node, node_dict in enumerate(step_dict):
                obs = node_dict[0][0]
                phase = phase_to_onehot(node_dict[1], 8)[0]
                direction = []
                if obs.shape[-1] == 12:

                    # 3 dims:left,straight,right
                    # list order:N,E,S,W
                    # N = obs[0:3]
                    # E = obs[3:6]
                    # S = obs[7:10]
                    # W = obs[10:13]

                    direction.append(np.concatenate([obs[0:3], phase]))
                    direction.append(np.concatenate([obs[3:6], phase]))
                    direction.append(np.concatenate([obs[6:9], phase]))
                    direction.append(np.concatenate([obs[9:], phase]))

                in_roads = inter_in_roads[inter_dict_id2inter[id_node]]
                for id_road, road in enumerate(in_roads):
                    road_id = road_dict_road2id[road]
                    road_feature[id_time][road_id] = direction[id_road]
                    if id_time == 0:
                        if id_node in mask_inter:
                            road_update[road_id] = 2
                        else:
                            road_update[road_id] = 1
        all_road_feature.append(road_feature)
    road_info = {'road_feature': all_road_feature, 'adj_road': adj_road,
                 'road_update': road_update, 'mask_or': mask_or}
    with open(save_dir, 'wb') as fw:
        pickle.dump(road_info, fw)
    print("save road_node_info done")


def search_data(sequence_length, num_of_depend, label_start_idx, len_input, num_for_predict, points_per_mhalf):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,num_of_mhalf
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    points_per_mhalf: int, number of points per mhalf, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_mhalf < 0:
        raise ValueError("points_per_mhalf should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_mhalf * i
        end_idx = start_idx + len_input
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_mhalf, label_start_idx, len_input, num_for_predict, points_per_mhalf):
    '''
    generate x and y in the form of slide window
    Parameters
    ------------------
    data_sequence: np.ndarray, (sequence_length, num_of_vertices, num_of_features)
    num_of_mhalf: int
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    points_per_mhalf: int, default 30, number of points per half of minutes
    Returns
    ------------------
    mhalf_sample: np.ndarray, (num_of_mhalf * points_per_mhalf, num_of_vertices, num_of_features)
    target: np.ndarray, (num_for_predict, num_of_vertices, num_of_features)
    '''
    mhalf_sample = None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return mhalf_sample, None

    if num_of_mhalf > 0:
        mhalf_indices = search_data(data_sequence.shape[0], num_of_mhalf, label_start_idx, len_input, num_for_predict, points_per_mhalf)
        
        if not mhalf_indices:
            return None, None

        mhalf_sample = np.concatenate([data_sequence[i: j] for i, j in mhalf_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return mhalf_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename, num_of_mhalf, len_input, num_for_predict, points_per_mhalf, save=False):
    '''
    generate train,val,test datasets
    Parameters
    ----------------------
    graph_signal_matrix_filename: string, file path of traffic states
    num_of_mhalf: int, minimum granularity of time interval
    len_input: int, input length
    num_for_predict: int, predict length
    points_per_mhalf: int, number of points per 30s, depends on data
    save: decide whether to save datasets
    '''
    with open(graph_signal_matrix_filename, "rb") as f_ob:
        all_data = pickle.load(f_ob)
    data_all = all_data['road_feature']
    node_update = all_data['road_update']
    mask_or = all_data['mask_or']
    adj_road = all_data['adj_road']
    all_samples = []
    for data_seq in data_all:
        for idx in range(data_seq.shape[0]):
            sample = get_sample_indices(data_seq, num_of_mhalf, idx, len_input, num_for_predict, points_per_mhalf)
            if sample[0] is None:
                continue
            # mhalf_sample(T,N,11)
            # target(T,N,F)
            mhalf_sample, target = sample

            # [(mhalf_sample),target,time_sample]
            sample = []
            if num_of_mhalf > 0:
                # mhalf_sample:(T,N,4)->(1,T,N,4)->(1,N,4,T)
                mhalf_sample = np.expand_dims(mhalf_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(mhalf_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            # target = target[:, :, [0, 1, 2]]
            sample.append(target)

            time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
            sample.append(time_sample)

            all_samples.append(sample)  # sampe：[(mhalf_sample),target,time_sample] = [(1,N,F,T),(1,N,F,T'),(1,1)]

    random.shuffle(all_samples)
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]  # [(B,N,F,T),(B,N,F,T'),(B,1)]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,F,T')
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]
    
    # generate data for model
    train_x = mask_op(train_x, node_update, adj_road)
    train_target = mask_op(train_target, node_update, adj_road)
    val_x = mask_op(val_x, node_update, adj_road)
    test_x = mask_op(test_x, node_update, adj_road)

    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        },
        'node_update': node_update,
        'mask_or': mask_or,
        'adj_road':adj_road
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])
    print('node update matrix :', all_data['node_update'].shape)
    print('mask intersection id:', all_data['mask_or'])

    if save:
        filename = graph_signal_matrix_filename.split('.')[0] + '_dataSplit.pkl'
        print('save file:', filename)

        dataset_info = {'train_x': all_data['train']['x'],
                        'train_target': all_data['train']['target'], 
                        'train_timestamp': all_data['train']['timestamp'],

                        'val_x': all_data['val']['x'],
                        'val_target': all_data['val']['target'], 
                        'val_timestamp': all_data['val']['timestamp'],

                        'test_x': all_data['test']['x'],
                        'test_target': all_data['test']['target'], 
                        'test_timestamp': all_data['test']['timestamp'],

                        'mean': all_data['stats']['_mean'], 
                        'std': all_data['stats']['_std'],

                        'node_update': all_data['node_update'],
                        'mask_or': all_data['mask_or']}

        with open(filename, 'wb') as fw:
            pickle.dump(dataset_info, fw)


def generate_actphase(phase, adj_mx, adj_phase):
    '''generate phase_activate matrix according to direction phase'''
    batch_size, num_of_timesteps, num_of_vertices, phase_row, phase_col = phase.shape
    # self.phase_act:record adj matrix of every time(after activation)
    phase_act = np.zeros((batch_size, num_of_timesteps,
                         num_of_vertices, num_of_vertices))
    phase_act = phase_act.reshape(-1, num_of_vertices, num_of_vertices)
    for idx, adj_x in enumerate(adj_mx.flat):
        if adj_x == 1.:
            if idx >= num_of_vertices:
                source = int(idx/num_of_vertices)
                target = idx-source*num_of_vertices
            else:
                source = 0
                target = idx
            phase_node = phase[:, :, source].reshape(-1, phase_row, phase_col)

            for phase_idx, x in enumerate(phase_node):
                if adj_phase[source][target] in x or adj_phase[source][target][1] == 'R':
                    phase_act[phase_idx][source][target] = 1.
    phase_act = phase_act.reshape(batch_size, num_of_timesteps, num_of_vertices, num_of_vertices)
    return phase_act


def generate_actphase_torch(phase, adj_mx, adj_phase):
    batch_size, num_of_timesteps, num_of_vertices, phase_row, phase_col = phase.shape
    # self.phase_act:record adj matrix of every time(after activation)
    phase_act = torch.zeros((batch_size, num_of_timesteps,
                         num_of_vertices, num_of_vertices))
    phase_act = phase_act.reshape(-1, num_of_vertices, num_of_vertices)
    if type(adj_mx) == np.ndarray:
        for idx, adj_x in enumerate(adj_mx.flat):
            if adj_x == 1.:
                if idx >= num_of_vertices:
                    source = int(idx / num_of_vertices)
                    target = idx - source * num_of_vertices
                else:
                    source = 0
                    target = idx
                phase_node = phase[:, :, source].reshape(-1, phase_row, phase_col)

                for phase_idx, x in enumerate(phase_node):
                    if adj_phase[source][target] in x or adj_phase[source][target][1] == 'R':
                        phase_act[phase_idx][source][target] = 1.
    else:
        for idx, adj_x in enumerate(adj_mx.view(-1)):
            if adj_x == 1.:
                if idx >= num_of_vertices:
                    source = int(idx / num_of_vertices)
                    target = idx - source * num_of_vertices
                else:
                    source = 0
                    target = idx
                phase_node = phase[:, :, source].reshape(-1, phase_row, phase_col)

                for phase_idx, x in enumerate(phase_node):
                    if adj_phase[source][target] in x or adj_phase[source][target][1] == 'R':
                        phase_act[phase_idx][source][target] = 1.
    phase_act = phase_act.reshape(batch_size, num_of_timesteps, num_of_vertices, num_of_vertices)
    return phase_act


def revise_unknown(origin_data, predict_data, mask_matrix):
    '''
    Parameters
    ------------------------
    origin_data: (B,N,F,T)
    predict_data: (B,N,F,T-1)
    Returns
    ------------------------
    imputed data: (B,N,F,T)
    '''
    revise_data = torch.zeros_like(origin_data,dtype=torch.float)
    for node_idx, node in enumerate(mask_matrix):
        if node != 1:
            revise_data[:, node_idx, :, 0] = origin_data[:, node_idx, :, 0]
            revise_data[:, node_idx, :, 1:] = predict_data[:, node_idx, :, :]
        else:
            revise_data[:, node_idx] = origin_data[:, node_idx]
    return revise_data


def mask_op(data_or, mask_matrix, adj_matrix):
    '''
    mask unobservable node by averaging its observable neighbors' states
    Parameters
    ------------------------
    data: (B,N,F,T)
    mask_matrix: (N,)
    adj_matrix: (N,N)
    Returns
    ------------------------
    data_or: (B,N,F,T)
    '''
    for mask_id, value in enumerate(mask_matrix):
        if value != 1:
            neighbors = []
            for col_id, x in enumerate(adj_matrix[:, mask_id]):
                if x == 1.:
                    neighbors.append(col_id)
            neighbor_all = np.zeros_like(data_or[:, 0, :3])
            if len(neighbors) != 0:
                for node in neighbors:
                    neighbor_all = data_or[:, node, :3] + neighbor_all
                data_or[:, mask_id, :3] = neighbor_all / len(neighbors)
            else:
                rand_id = random.randint(0, len(mask_matrix)-1)
                while mask_matrix[rand_id] != 1:
                    rand_id = random.randint(0, len(mask_matrix)-1)
                data_or[:,mask_id, :3] = data_or[:,rand_id, :3]
            if value == 0:
                # set virtual node's phase
                rand_id = random.sample(neighbors, 1)[0]
                data_or[:, mask_id, 3:] = data_or[:, rand_id, 3:]
    return data_or

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/DTIGNN.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    num_of_vertices = int(data_config['num_of_vertices'])
    len_input = int(data_config['len_input'])
    points_per_mhalf = int(data_config['len_input'])
    num_for_predict = int(data_config['num_for_predict'])
    
    dataset_name = data_config['dataset_name']
    base_model = data_config['base_model']

    num_of_mhalf = int(training_config['num_of_mhalf'])
    neighbor_node = data_config['neighbor_node']
    mask_num = data_config['mask_num']

    data_basedir = os.path.join('data',str(dataset_name))
    state_basedir = os.path.join(data_basedir,'state_data')
    relation_filename = os.path.join(data_basedir,'roadnet_relation.pkl')
    graph_signal_matrix_filename = 's'+str(points_per_mhalf)+'_p'+str(num_for_predict)+'_n'+str(neighbor_node)+'_m'+str(mask_num)+'.pkl'
    graph_signal_matrix_filename = os.path.join(state_basedir,graph_signal_matrix_filename)

    if not os.path.exists(data_basedir):
        os.makedirs(data_basedir)
    if not os.path.exists(state_basedir):
        os.makedirs(state_basedir)

    # read state of intersections,convert it into state which road graph needed,save.

    if dataset_name == 'hz_4x4':
        # rawstate.pkl: generated using origin taffic flows
        # rawstate_d.pkl: generated using double taffic flows
        state_file = ['rawstate.pkl','rawstate_d.pkl']
    else:
        state_file = ['rawstate.pkl']

    state_file_list = [os.path.join(data_basedir, s_dic) for s_dic in state_file]
    
    build_road_state(relation_filename, state_file_list, neighbor_node, mask_num, save_dir=graph_signal_matrix_filename)

    # according to file of task above, generate train set,val set and test set.
    read_and_generate_dataset(graph_signal_matrix_filename, num_of_mhalf, len_input, num_for_predict, points_per_mhalf=points_per_mhalf, save=True)
