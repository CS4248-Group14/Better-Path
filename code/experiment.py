from multiprocessing import Pool
from learn import EncoderType, train

BATCH_SIZE = 1024
MAX_ITER = 4000


# Results: adding the l1 distance between end vectors of edges yields the best results
def ablation_features():
    folder = 'ablation_features/'
    basic_features = [
        'v_enc_dim100', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
        'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel',
        'e_sense'
    ]
    new_features = ['e_vertexl1dist', 'e_vertexl2dist']

    input_args = [[
        'science', basic_features + new_features, [], EncoderType.BASIC,
        'LSTM', 10, False, 0.8, f'{folder}all_features/', False, MAX_ITER,
        BATCH_SIZE
    ],
                  [
                      'science', basic_features, [], EncoderType.BASIC, 'LSTM',
                      10, False, 0.8, f'{folder}no_new_features/', False,
                      MAX_ITER, BATCH_SIZE
                  ]]
    for i in range(len(new_features)):
        input_args.append([
            'science', basic_features + [new_features[i]], [],
            EncoderType.BASIC, 'LSTM', 10, False, 0.8,
            f'{folder}add_{i}th_feature/', False, MAX_ITER, BATCH_SIZE
        ])
    p = Pool(len(input_args))
    p.starmap(train, input_args)


# Results: adding the pairwise heuristic yields the best results
def ablation_heuristics():
    folder = 'ablation_heuristics/'
    # From features ablation study, we know that the l1 distance between end vectors of edges improves the performance the most
    features = [
        'v_enc_dim100', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
        'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel',
        'e_sense', 'e_vertexl1dist'
    ]
    new_heuristics = ['st', 'pairwise', 'rf', 'length']
    input_args = [[
        'science', features, new_heuristics, EncoderType.BASIC, 'LSTM', 10,
        False, 0.8, f'{folder}all_heuristics/', False, MAX_ITER, BATCH_SIZE
    ],
                  [
                      'science', features, [], EncoderType.BASIC, 'LSTM', 10,
                      False, 0.8, f'{folder}no_heuristics/', False, MAX_ITER,
                      BATCH_SIZE
                  ]]
    for i in range(len(new_heuristics)):
        input_args.append([
            'science', features, [new_heuristics[i]], EncoderType.BASIC,
            'LSTM', 10, False, 0.8, f'{folder}add_{i}th_heuristic/', False,
            MAX_ITER, BATCH_SIZE
        ])
    p = Pool(len(input_args))
    p.starmap(train, input_args)


# Results: using the encoder that concatenating features/heuristics instead of averaging them yields the best results
def experiment_encoders():
    folder = 'experiment_encoders/'
    features = [
        'v_enc_dim100', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
        'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel',
        'e_sense', 'e_vertexl1dist'
    ]
    # From heuristics ablation study, we know that the pairwise heuristic improves the performance the most
    heuristics = ['pairwise']
    input_args = []
    for encoder_type in EncoderType:
        input_args.append([
            'science', features, heuristics, encoder_type, 'LSTM', 10, False,
            0.8, f'{folder}{encoder_type.name}/', False, MAX_ITER, BATCH_SIZE
        ])
    p = Pool(len(input_args))
    p.starmap(train, input_args)


# Results: using multi-layer Predictor yields better results
def ablation_multilayer():
    folder = 'ablation_multilayer/'
    features = [
        'v_enc_dim100', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
        'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel',
        'e_sense', 'e_vertexl1dist'
    ]
    heuristics = ['pairwise']
    input_args = []
    for use_multilayer in [True, False]:
        # From encoders ablation study, we know that concatenating features/heuristics instead of averaging them works the best
        input_args.append([
            'science', features, heuristics, EncoderType.CONCAT, 'LSTM', 10,
            use_multilayer, 0.8, f'{folder}{str(use_multilayer)}/', False,
            MAX_ITER, BATCH_SIZE
        ])
    p = Pool(len(input_args))
    p.starmap(train, input_args)


def experiment_rnn_types():
    folder = 'experiment_rnn_types/'
    features = [
        'v_enc_dim100', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
        'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel',
        'e_sense', 'e_vertexl1dist'
    ]
    heuristics = ['pairwise']
    input_args = []
    for rnn_type in ['TransformerEncoder']:
        # From multilayer ablation study, we know that multilayer is helpful to the performance
        input_args.append([
            'science', features, heuristics, EncoderType.CONCAT, rnn_type, 10,
            True, 0.8, f'{folder}{rnn_type}/', False, MAX_ITER, BATCH_SIZE
        ])
    p = Pool(len(input_args))
    p.starmap(train, input_args)


if __name__ == '__main__':
    # ablation_features()
    # ablation_heuristics()
    # experiment_encoders()
    # ablation_multilayer()
    experiment_rnn_types()