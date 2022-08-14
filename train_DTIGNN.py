import torch
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from DTIGNN import make_model
from common.utils import compute_val_loss, predict_and_save_results, load_graphdata_normY_channel
from prepareData import get_road_adj
from tensorboardX import SummaryWriter
from common.metrics import masked_mse

# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/DTIGNN.conf', type=str, help="configuration file path")
parser.add_argument("--device", default='cuda:1', type=str, help="choose gpu")

args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

num_of_vertices = int(data_config['num_of_vertices'])
points_per_mhalf = int(data_config['points_per_mhalf'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
neighbor_node = int(data_config['neighbor_node'])
mask_num = int(data_config['mask_num'])
len_input = int(data_config['len_input'])

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(args.device)
print("CUDA:", USE_CUDA, DEVICE)

model_name = training_config['model_name']
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch'])
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
batch_size = int(training_config['batch_size'])
print('total training epoch, fine tune epoch:', epochs, ',', fine_tune_epochs, flush=True)
print('batch_size:', batch_size, flush=True)

num_of_mhalf = int(training_config['num_of_mhalf'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_input_size = int(training_config['decoder_input_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])

# whether use spatial embedding
SE = bool(int(training_config['SE']))
smooth_layer_num = int(training_config['smooth_layer_num'])
aware_temporal_context = bool(int(training_config['aware_temporal_context']))
TE = bool(int(training_config['TE']))
use_LayerNorm = True
residual_connection = True

data_basedir = os.path.join('data', str(dataset_name))
state_basedir = os.path.join(data_basedir, 'state_data')
relation_filename = os.path.join(data_basedir, 'roadnet_relation.pkl')

graph_signal_matrix_filename = 's'+str(points_per_mhalf)+'_p'+str(num_for_predict)+'_n'+str(neighbor_node)+'_m'+str(mask_num)+'_dataSplit.pkl'
graph_signal_matrix_filename = os.path.join(state_basedir, graph_signal_matrix_filename)

folder_dir = 's%d_p%d_n%d_m%d_%s_%d' % (points_per_mhalf, num_for_predict, neighbor_node, mask_num, model_name, encoder_input_size)
_params_path = os.path.join('experiments', dataset_name, folder_dir)

print('folder_dir:', folder_dir)
print('params_path:', _params_path)

adj_mx, adj_phase = get_road_adj(relation_filename)

def train_main(net, train_loader, val_loader, test_loader, test_target_tensor, _mean, _std, mask_matrix, mask_or):
    params_path = _params_path
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path), flush=True)
    else:
        raise SystemExit('Wrong type of model!')

    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)

    criterion = masked_mse
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    total_param = 0
    print('Net\'s state_dict:', flush=True)
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param, flush=True)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name], flush=True)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch, flush=True)

        print('load weight from: ', params_filename, flush=True)

    start_time = time()

    # train model
    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch, mask_matrix, _mean, _std)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

        net.train()

        type = 'train'

        train_start_time = time()

        for _, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.transpose(-1, -2)[:, :, :, :3] # (B, N, T, 1)

            labels = labels.transpose(-1, -2)

            optimizer.zero_grad()

            outputs,encoder_refill = net(encoder_inputs, decoder_inputs, _mean, _std)

            loss = criterion(outputs, labels, mask_matrix, type) + criterion(encoder_refill, encoder_inputs[:,:,:,:3], mask_matrix, type)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)
            
            # delete caches
            del encoder_inputs, decoder_inputs, labels, outputs,encoder_refill, loss
            torch.cuda.empty_cache()

        print('epoch: %s, train time every whole data:%.2fs' %
              (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' %
              (epoch, time() - start_time), flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(net, params_path, best_epoch, test_loader, test_target_tensor, _mean, _std, mask_matrix, mask_or, 'test')

    # fine tune the model
    optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
    print('fine tune the model ... ', flush=True)
    for epoch in range(epochs, epochs+fine_tune_epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        net.train()

        train_start_time = time()

        for _, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.transpose(-1, -2)[:, :, :, :3] # (B, N, T, 1)

            labels = labels.transpose(-1, -2)

            predict_length = labels.shape[2]

            optimizer.zero_grad()

            encoder_output,encoder_refill = net.encode(encoder_inputs, _mean, _std)

            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]

            for _ in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels, mask_matrix, 'train') + criterion(encoder_refill, encoder_inputs[:,:,:,:3], mask_matrix, 'train')

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            # delete caches
            del encoder_inputs, decoder_inputs, labels, encoder_output,encoder_refill, loss
            torch.cuda.empty_cache()

        print('epoch: %s, train time every whole data:%.2fs' %
              (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' %
              (epoch, time() - start_time), flush=True)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch, mask_matrix, _mean, _std)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(net, params_path, best_epoch, test_loader, test_target_tensor, _mean, _std, mask_matrix, mask_or, 'test')


def predict_main(net, params_path, epoch, data_loader, data_target_tensor, _mean, _std, mask_matrix, mask_or, type):
    '''Test on test dataset'''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
    print('load weight from:', params_filename, flush=True)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results(net, data_loader, data_target_tensor, epoch, _mean, _std, params_path, mask_matrix, mask_or, type)


if __name__ == "__main__":
    train_loader, _, val_loader, _, test_loader, test_target_tensor, _mean, _std, mask_matrix, mask_or = load_graphdata_normY_channel(graph_signal_matrix_filename, DEVICE, batch_size)

    net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, adj_phase, mask_matrix, nb_head,
                        num_of_mhalf, points_per_mhalf, num_for_predict, len_input, dropout=dropout, aware_temporal_context=aware_temporal_context,
                        SE=SE, TE=TE, kernel_size=kernel_size, smooth_layer_num=smooth_layer_num, residual_connection=residual_connection,
                        use_LayerNorm=use_LayerNorm)

    train_main(net, train_loader, val_loader, test_loader, test_target_tensor, _mean, _std, mask_matrix, mask_or)
    
    # predict_main(net, _params_path, 100, test_loader,test_target_tensor, _mean, _std, mask_matrix, mask_or, 'test')