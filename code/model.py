import torch
from torch import nn


class FeatureTransformer(nn.Module):
    '''
    take an n x d_in matrix and transform it into a n x d_out matrix
    where the n x d_in matrix is the n examples each with d_in dimensions
    '''
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.linear = nn.Linear(d_in, d_out)
        # self.linear1 = nn.Linear(d_in, 100)
        # self.relu_m = nn.ReLU(inplace=False)
        # self.linear2 = nn.Linear(100, d_out)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input):
        return self.relu(self.linear(input))
        # return self.relu(self.linear2(self.relu_m(self.linear1(input))))


class ChainEncoder(nn.Module):
    '''
    encodes N chains at the same time
    assumes that each of the chains are of the same length
    '''
    def __init__(self, v_feature_lengths, e_feature_lengths, out_length,
                 rnn_type, pooling):
        super().__init__()
        feature_enc_length = out_length
        num_layers = 1
        self.rnn_type = rnn_type
        self.pooling = pooling
        self.v_feature_lengths = v_feature_lengths
        self.e_feature_lengths = e_feature_lengths
        self.v_fea_encs = nn.ModuleList()
        self.e_fea_encs = nn.ModuleList()
        for d_in in self.v_feature_lengths:
            self.v_fea_encs.append(FeatureTransformer(d_in,
                                                      feature_enc_length))
        for d_in in self.e_feature_lengths:
            self.e_fea_encs.append(FeatureTransformer(d_in,
                                                      feature_enc_length))
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=feature_enc_length,
                              hidden_size=out_length,
                              num_layers=num_layers)
        elif self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=feature_enc_length,
                                hidden_size=out_length,
                                num_layers=num_layers)

    def forward(self, input):
        '''
        input is a list of v_features, and e_features
        v_features is a list of num_vertices tuples
        each tuple is an N x d_in Variable, in which N is the batch size, and d_in is the feature length
        e_features is structured similarly
        '''

        v_features, e_features = input
        v_encs = []
        for i in range(len(v_features)):
            v_enc = None
            for j in range(len(v_features[i])):
                fea_enc = self.v_fea_encs[j]
                if v_enc is None:
                    v_enc = fea_enc(v_features[i][j]).clone()
                else:
                    v_enc += fea_enc(v_features[i][j]).clone()
            v_enc = v_enc / len(v_features[i])
            v_encs.append(v_enc)
        e_encs = []
        for i in range(len(e_features)):
            e_enc = None
            for j in range(len(e_features[i])):
                fea_enc = self.e_fea_encs[j]
                if e_enc is None:
                    e_enc = fea_enc(e_features[i][j]).clone()
                else:
                    e_enc += fea_enc(e_features[i][j]).clone()
            e_enc = e_enc / len(e_features[i])
            e_encs.append(e_enc)

        combined_encs = [0] * (len(v_encs) + len(e_encs))
        combined_encs[::2] = v_encs
        combined_encs[1::2] = e_encs
        combined_encs = torch.stack(combined_encs, dim=0)
        # combined_encs has shape (#V+#E) x resample_size x out_length
        if self.rnn_type == 'RNN':
            output, hidden = self.rnn(combined_encs)
        elif self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.lstm(combined_encs)
        if self.pooling == 'last':
            return output[-1]
        else:
            return torch.mean(output, dim=0)


class ConcatChainEncoder(nn.Module):
    '''
    concatenation instead of taking average
    '''
    def __init__(self, v_feature_lengths, e_feature_lengths, out_length,
                 rnn_type, pooling):
        super().__init__()
        feature_enc_length = out_length
        num_layers = 1
        self.rnn_type = rnn_type
        self.pooling = pooling
        self.v_fea_enc = (FeatureTransformer(sum(v_feature_lengths),
                                             feature_enc_length))
        self.e_fea_enc = (FeatureTransformer(sum(e_feature_lengths),
                                             feature_enc_length))

        seq_length = 7  # ! Note: this is specific for 'science' paths
        self.top_linear = nn.Linear(seq_length * feature_enc_length,
                                    out_length)
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=feature_enc_length,
                              hidden_size=out_length,
                              num_layers=num_layers)
        elif self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=feature_enc_length,
                                hidden_size=out_length,
                                num_layers=num_layers)
        elif self.rnn_type == 'TransformerEncoder':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_enc_length, nhead=5)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                             num_layers=6)
            # Output from the transformer encoder is seq_length x resample_size x feature_enc_length
            # we need to swap first two axes and flatten the sequence length dimension
            # The dimension of the input to the linear linear layer is resample_size x [seq_length x feature_enc_length]

    def forward(self, input):
        '''
        input is a list of v_features, and e_features
        v_features is a list of num_vertices tuples
        each tuple is an N x d_in Variable, in which N is the batch size, and d_in is the feature length
        e_features is structured similarly
        '''

        ## (a1) the encoding of vertices and edges
        v_features, e_features = input
        v_encs = []
        for i in range(len(v_features)):
            v_enc = torch.cat(v_features[i], axis=1)
            v_enc = self.v_fea_enc(v_enc).clone()
            v_encs.append(v_enc)
        e_encs = []
        for i in range(len(e_features)):
            e_enc = torch.cat(e_features[i], axis=1)
            e_enc = self.e_fea_enc(e_enc).clone()
            e_encs.append(e_enc)

        ## (b) the encoding of the path
        seq_length = len(v_encs) + len(e_encs)
        combined_encs = [0] * seq_length
        combined_encs[::2] = v_encs
        combined_encs[1::2] = e_encs
        combined_encs = torch.stack(combined_encs, dim=0)
        # combined_encs has shape (#V+#E) x resample_size x out_length
        if self.rnn_type == 'RNN':
            output, hidden = self.rnn(combined_encs)
        elif self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.lstm(combined_encs)
        elif self.rnn_type == 'TransformerEncoder':
            output = self.transformer_encoder(combined_encs)
        if self.pooling == 'last' and self.rnn_type != 'TransformerEncoder':
            return output[-1]
        # output size will be resample_size x out_length
        else:
            output = output.permute(1, 0, 2)
            output = output.flatten(start_dim=1)
            return self.top_linear(output)


class CombinedConcatChainEncoder(nn.Module):
    '''
    without assumption that a vertex chain is of same length as an edge chain
    instead, each chain consists of one vertex and one edge
    chains are therefore still of same length
    '''
    def __init__(self, v_feature_lengths, e_feature_lengths, out_length,
                 rnn_type, pooling):
        super().__init__()
        feature_enc_length = sum(v_feature_lengths) + sum(e_feature_lengths)
        num_layers = 1
        self.rnn_type = rnn_type
        self.pooling = pooling
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=feature_enc_length,
                              hidden_size=out_length,
                              num_layers=num_layers)
        elif self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=feature_enc_length,
                                hidden_size=out_length,
                                num_layers=num_layers)

    def forward(self, input):
        '''
        input is a list of v_features, and e_features
        v_features is a list of num_vertices tuples
        each tuple is an N x d_in Variable, in which N is the batch size, and d_in is the feature length
        e_features is structured similarly
        '''

        ## (a1) the encoding of vertices and edges
        v_features, e_features = input
        v_encs = []
        for i in range(len(v_features)):
            v_enc = torch.cat(v_features[i], axis=1)
            v_encs.append(v_enc)
        e_encs = []
        for i in range(len(e_features)):
            e_enc = torch.cat(e_features[i], axis=1)
            e_encs.append(e_enc)

        ## (a2) concatenating vertex i with edge i-1
        dummy_edge_enc = torch.zeros(e_encs[0].shape)
        e_encs = [dummy_edge_enc] + e_encs  # adding dummy edge at the start
        concat_encs = []
        for v_enc, e_enc in zip(v_encs, e_encs):
            concatenated_enc = torch.cat([e_enc, v_enc], axis=1)
            concat_encs.append(concatenated_enc)

        ## (b) the encoding of the path
        combined_encs = torch.stack(concat_encs, dim=0)
        # combined_encs has shape (#V) x resample_size x feature_enc_length
        if self.rnn_type == 'RNN':
            output, hidden = self.rnn(combined_encs)
        elif self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.lstm(combined_encs)
        if self.pooling == 'last':
            return output[-1]
        else:
            return torch.mean(output, dim=0)


class Predictor(nn.Module):
    '''
    takes two feature vectors and produces a prediction
    '''
    def __init__(self, feature_len, use_multilayer=False):
        super().__init__()
        self.use_multilayer = use_multilayer
        self.linear = nn.Linear(feature_len, 1)
        self.linear1 = nn.Linear(feature_len, 100)
        self.relu = nn.ReLU(inplace=False)
        self.linear2 = nn.Linear(100, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, vec1, vec2):

        if self.use_multilayer:
            a = self.linear2(self.relu(self.linear1(vec1)))
            b = self.linear2(self.relu(self.linear1(vec2)))
        else:
            a = self.linear(vec1)
            b = self.linear(vec2)
        combined = torch.cat((a, b), dim=1)
        return self.logsoftmax(combined)
