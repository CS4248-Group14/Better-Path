import torch
from torch import nn, optim
from model import ChainEncoder, Predictor, AlternateChainEncoder, ConcatChainEncoder_Brendan
from dataset import Dataset
from multiprocessing import Pool


def train(dataset, features, heuristics, fea_len, split_frac, out_file,
          use_gpu, max_iter, batch_size, ckpt_path):
    if isinstance(out_file, str):
        out_file = open(out_file, 'w')
    d = Dataset(features, heuristics, split_frac, use_gpu,
                '../prepare_data/features', '../prepare_data/heuristics',
                f'../data/{dataset}')
    print('defining architecture')
    # enc = ChainEncoder(d.get_v_fea_len(), d.get_e_fea_len(), fea_len, 'last')
    enc = AlternateChainEncoder(d.get_v_fea_len(), d.get_e_fea_len(), fea_len,
                                'last')
    # enc = ConcatChainEncoder_Brendan(d.get_v_fea_len(), d.get_e_fea_len(), fea_len, 'last')

    predictor = Predictor(fea_len + len(heuristics))
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
        output_A = torch.concat((output_A, *heuristic_A), dim=1)
        output_B = torch.concat((output_B, *heuristic_B), dim=1)
        softmax_output = predictor(output_A, output_B)
        loss_val = loss(softmax_output, y)
        loss_val.backward()
        optimizer.step()

        enc.zero_grad()
        predictor.zero_grad()
        output_test_A = enc(test_chain_A)
        output_test_B = enc(test_chain_B)
        output_test_A = torch.concat((output_test_A, *test_h_A), dim=1)
        output_test_B = torch.concat((output_test_B, *test_h_B), dim=1)
        softmax_output = predictor(output_test_A,
                                   output_test_B).data.cpu().numpy()
        test_y_pred = softmax_output.argmax(axis=1)
        cur_acc = (test_y_pred == test_y).sum() / len(test_y)
        if train_iter % 10 == 0:
            print(train_iter, 'test acc:', cur_acc)
        out_file.write('%f\n' % cur_acc)
        if train_iter % 50 == 0:
            torch.save(enc.state_dict(),
                       '%s/%i_encoder.model' % (ckpt_path, train_iter))
            torch.save(predictor.state_dict(),
                       '%s/%i_predictor.model' % (ckpt_path, train_iter))
    out_file.close()


if __name__ == '__main__':
    features = [
        'v_enc_dim300', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
        'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel',
        'e_sense'
    ]
    # features = ['v_deg', 'v_sense', 'e_weightsource', 'e_srank_rel']
    # heuristics = ['st', 'pairwise', 'rf', 'length']
    heuristics = ['st']
    train('science', features, heuristics, 20, 0.8, 'science_train.log', False,
          100, 1000, 'science_ckpt')
    # train('open_domain', features, heuristics, 10, 0.95, 'open_domain_train.log', False, 100, 100, 'open_domain_ckpt')
