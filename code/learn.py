import pathlib
import torch
from torch import nn, optim
from model import ChainEncoder, Predictor, AlternateChainEncoder, ConcatChainEncoder
from dataset import Dataset
from multiprocessing import Pool
from enum import Enum


class EncoderType(Enum):
    BASIC = 0
    ALTERNATE = 1
    CONCAT = 2


def train(dataset, features, heuristics, encoder_type, rnn_type, fea_len,
          use_multilayer, split_frac, out_path, use_gpu, max_iter, batch_size):
    # here the fea_len means the feature length of a path
    ckpt_path = out_path + 'science_ckpt'
    if isinstance(out_path, str):
        path = pathlib.Path(out_path)
        path.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path + 'science_train.log', 'w')
    d = Dataset(features, heuristics, split_frac, use_gpu,
                '../prepare_data/features', '../prepare_data/heuristics',
                f'../data/{dataset}')
    print('defining architecture')
    if encoder_type == EncoderType.BASIC:
        enc = ChainEncoder(d.get_v_fea_len(), d.get_e_fea_len(), fea_len,
                           rnn_type, 'last')
    elif encoder_type == EncoderType.ALTERNATE:
        enc = AlternateChainEncoder(d.get_v_fea_len(), d.get_e_fea_len(),
                                    fea_len, rnn_type, 'last')
    else:
        enc = ConcatChainEncoder(d.get_v_fea_len(), d.get_e_fea_len(), fea_len,
                                 rnn_type, 'last')

    predictor = Predictor(fea_len + len(heuristics), use_multilayer)
    loss = nn.NLLLoss()
    if use_gpu:
        enc.cuda()
        predictor.cuda()
        loss.cuda()

    optimizer = optim.Adam(
        list(enc.parameters()) + list(predictor.parameters()))

    print('training')
    test_chain_A, test_h_A, test_chain_B, test_h_B, test_y = d.get_test_pairs()
    test_y = test_y.data.cpu().numpy()
    for train_iter in range(max_iter):
        chains_A, heuristic_A, chains_B, heuristic_B, y = d.get_train_pairs(
            batch_size)
        enc.zero_grad()
        predictor.zero_grad()
        output_A = enc(chains_A)
        output_B = enc(chains_B)
        output_A = torch.cat((output_A, *heuristic_A), dim=1)
        output_B = torch.cat((output_B, *heuristic_B), dim=1)
        softmax_output = predictor(output_A, output_B)
        loss_val = loss(softmax_output, y)
        loss_val.backward()
        optimizer.step()

        enc.zero_grad()
        predictor.zero_grad()
        output_test_A = enc(test_chain_A)
        output_test_B = enc(test_chain_B)
        output_test_A = torch.cat((output_test_A, *test_h_A), dim=1)
        output_test_B = torch.cat((output_test_B, *test_h_B), dim=1)
        softmax_output = predictor(output_test_A,
                                   output_test_B).data.cpu().numpy()
        test_y_pred = softmax_output.argmax(axis=1)
        cur_acc = (test_y_pred == test_y).sum() / len(test_y)
        if train_iter % 10 == 0:
            print(train_iter, 'test acc:', cur_acc)
        out_file.write('%f\n' % cur_acc)
        if train_iter % 50 == 0:
            path = pathlib.Path(ckpt_path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(enc.state_dict(),
                       '%s/%i_encoder.model' % (ckpt_path, train_iter))
            torch.save(predictor.state_dict(),
                       '%s/%i_predictor.model' % (ckpt_path, train_iter))
    out_file.close()


if __name__ == '__main__':
    features = [
        'v_enc_dim100', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
        'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel',
        'e_sense', 'e_vertexl1dist'
    ]
    heuristics = ['pairwise']
    # features = ['v_deg', 'v_sense', 'e_weightsource', 'e_srank_rel']
    # heuristics = ['st', 'pairwise', 'rf', 'length']
    train('science', features, heuristics, EncoderType.CONCAT, 'RNN', 10, True,
          0.8, './RNN/', False, 4000, 1024)
    # train('open_domain', features, heuristics, 10, 0.95, 'open_domain_train.log', False, 100, 100, 'open_domain_ckpt')
