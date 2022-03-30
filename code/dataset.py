from __future__ import division

import pickle, random
import numpy as np
from itertools import cycle

import torch
from torch.autograd import Variable

all_feature_lengths = {'v_enc_onehot': 100,
                       'v_enc_embedding': 300,
                       'v_enc_dim300': 300,
                       'v_enc_dim2': 2,
                       'v_enc_dim10': 10,
                       'v_enc_dim50': 50,
                       'v_enc_dim100': 100,
                       'v_freq_freq': 1,
                       'v_freq_rank': 1,
                       'v_deg': 1,
                       'v_sense': 1,
                       'e_vertexsim': 1,
                       'e_dir': 3,
                       'e_rel': 46,
                       'e_weight': 1,
                       'e_source': 6,
                       'e_weightsource': 6,
                       'e_srank_abs': 1,
                       'e_srank_rel': 1,
                       'e_trank_abs': 1,
                       'e_trank_rel': 1,
                       'e_sense': 1}
# TODO: @Wang Qian, add e_xx here
# TODO: @Jiayu, add heuristic extraction util here

class Dataset:
    def __init__(self, feature_names, heuristic_names, new_feature_names, train_test_split_fraction, gpu, features_src_path='features',
                 heuristics_src_path='heuristics', new_features_src_path='new_features', src_data_path='../../data/science'):
        self.feature_names = feature_names
        self.cached_features = dict()
        self.gpu = gpu
        for f in feature_names:
            print('loading '+f)
            self.cached_features[f] = pickle.load(
                open('%s/%s.pkl'%(features_src_path, f), 'rb'), encoding='latin1')
        sampled_problems = pickle.load(open(
            '%s/paths.pkl'%src_data_path, 'rb'), encoding='latin1')
        self.texts = dict()
        print('loading problem plain texts')
        for id_num in sampled_problems:
            f_short = sampled_problems[id_num]['forward']['short']
            r_short = sampled_problems[id_num]['reverse']['short']
            self.texts[id_num+'f'] = f_short
            self.texts[id_num+'r'] = r_short
        print('loading labeled pairs')
        self.all_pairs = [] # list of id tuples (good, bad)
        for l in open('%s/answers.txt'%src_data_path):
            first, second, good = l.strip().split('_')
            if first==good:
                bad = second
            elif second==good:
                bad = first
            g_len = (len(self.texts[good].strip().split(' '))+1)/2
            b_len = (len(self.texts[bad].strip().split(' '))+1)/2
            if g_len!=4 or b_len!=4:
                continue
            self.all_pairs.append((good, bad))
        random.shuffle(self.all_pairs)

        # add heuristics
        self.heuristic_names = heuristic_names
        self.cached_heuristics = dict()
        for h in self.heuristic_names:
            print('loading h', h)
            self.cached_heuristics[h] = pickle.load(open('%s/%s.pkl'%(heuristics_src_path, h), 'rb'))

        # add new features
        self.new_features_names = new_feature_names
        self.cached_new_features = dict()
        for n in self.new_features_names:
            print('loading n', n)
            self.cached_new_features[n] = pickle.load(open('%s/%s.pkl'%(new_features_src_path, n), 'rb'))


        split = int(train_test_split_fraction*len(self.all_pairs))
        self.train_pairs = self.all_pairs[:split]
        self.test_pairs = self.all_pairs[split:]

        self.train_pairs = self.train_pairs[:len(self.train_pairs)]
        self.cycled_train_pairs = cycle(self.train_pairs)

    def get_fea_len(self):
        return [all_feature_lengths[f] for f in self.feature_names]

    def get_v_fea_len(self):
        return [all_feature_lengths[f] for f in self.feature_names if f.startswith('v')]

    def get_e_fea_len(self):
        return [all_feature_lengths[f] for f in self.feature_names if f.startswith('e')]

    def get_chain_len(self, id):
        return len(self.get_features(id)[0])

    def get_features(self, id):
        v_features = []
        e_features = []
        for f in self.feature_names:
            if f.startswith('v'):
                v_features.append(self.cached_features[f][id])
            else:
                e_features.append(self.cached_features[f][id])
        v_features = zip(*v_features)
        e_features = zip(*e_features)
        heuristics = [self.cached_heuristics[h][id] for h in self.heuristic_names]
        new_features = [self.cached_new_features[n][id] for n in self.new_features_names]
        return v_features, e_features, heuristics, new_features

    def prepare_feature_placeholder(self, N):
        v_features = [[],[],[],[]]
        e_features = [[],[],[]]
        heuristics = []
        new_features = []

        for feature in v_features:
            for f in self.feature_names:
                if f.startswith('v'):
                    feature.append(
                        np.zeros((N, all_feature_lengths[f]), dtype='float32')
                    )
        for feature in e_features:
            for f in self.feature_names:
                if f.startswith('e'):
                    feature.append(
                        np.zeros((N, all_feature_lengths[f]), dtype='float32')
                    )
        for h in self.heuristic_names:
            heuristics.append(np.zeros((N, 1), dtype='float32'))
        for n in self.new_features_names:
            new_features.append(np.zeros((N, 1), dtype='float32'))
        return v_features, e_features, heuristics, new_features

    def get_train_pairs(self, N, randomize_dir=True):
        '''
        return a list of two lists, X_A and X_B, as well as a list y
        each list consists of two lists, which are vertex and edge representations
        each list consists of #V or #E lists, which are individual vertices/edges
        each list consists of several N x feature_len torch Variables, which are individual features
        currently only keeping chains of length 4
        if for i-th problem, the good chain is in X_A, then y[i]==1, else y[i]==0
        '''
        v_features_A, e_features_A, heuristics_A, new_features_A = self.prepare_feature_placeholder(N)
        v_features_B, e_features_B, heuristics_B, new_features_B = self.prepare_feature_placeholder(N)
        y = np.zeros(N, dtype='int64')

        for instance_idx in range(N):
            good, bad = next(self.cycled_train_pairs)
            if randomize_dir:
                good = good[:-1]+random.choice(['f','r'])
                bad = bad[:-1]+random.choice(['f','r'])
            v_good, e_good, h_good, n_good = self.get_features(good)
            v_bad, e_bad, h_bad, n_bad = self.get_features(bad)
            v_good, e_good, v_bad, e_bad = list(v_good), list(e_good), list(v_bad), list(e_bad)

            label = random.random()>0.5
            y[instance_idx] = label
            for v_idx in range(4):
                for v_fea_idx in range(len(v_good[v_idx])):
                    if label:
                        v_features_A[v_idx][v_fea_idx][instance_idx] = v_good[v_idx][v_fea_idx]
                        v_features_B[v_idx][v_fea_idx][instance_idx] = v_bad[v_idx][v_fea_idx]
                    else:
                        v_features_B[v_idx][v_fea_idx][instance_idx] = v_good[v_idx][v_fea_idx]
                        v_features_A[v_idx][v_fea_idx][instance_idx] = v_bad[v_idx][v_fea_idx]

            for e_idx in range(3):
                for e_fea_idx in range(len(e_good[e_idx])):
                    if label:
                        e_features_A[e_idx][e_fea_idx][instance_idx] = e_good[e_idx][e_fea_idx]
                        e_features_B[e_idx][e_fea_idx][instance_idx] = e_bad[e_idx][e_fea_idx]
                    else:
                        e_features_B[e_idx][e_fea_idx][instance_idx] = e_good[e_idx][e_fea_idx]
                        e_features_A[e_idx][e_fea_idx][instance_idx] = e_bad[e_idx][e_fea_idx]

            for h_idx in range(len(h_good)):
                if label:
                    heuristics_A[h_idx][instance_idx] = h_good[h_idx]
                    heuristics_B[h_idx][instance_idx] = h_bad[h_idx]
                else:
                    heuristics_B[h_idx][instance_idx] = h_good[h_idx]
                    heuristics_A[h_idx][instance_idx] = h_bad[h_idx]

            for n_idx in range(len(n_good)):
                if label:
                    new_features_A[n_idx][instance_idx] = n_good[n_idx]
                    new_features_B[n_idx][instance_idx] = n_bad[n_idx]
                else:
                    new_features_A[n_idx][instance_idx] = n_good[n_idx]
                    new_features_B[n_idx][instance_idx] = n_bad[n_idx]

        for features in [v_features_A, e_features_A, v_features_B, e_features_B]:
            for feature in features:
                for i in range(len(feature)):
                    feature[i] = Variable(torch.from_numpy(feature[i]))
                    if self.gpu:
                        feature[i] = feature[i].cuda()
        for heuristics in [heuristics_A, heuristics_B]:
            for i in range(len(heuristics)):
                heuristics[i] = Variable(torch.from_numpy(heuristics[i]))
                if self.gpu:
                    heuristics[i] = heuristics[i].cuda()

        for new_features in [new_features_A, new_features_B]:
            for i in range(len(new_features)):
                new_features[i] = Variable(torch.from_numpy(new_features[i]))
                if self.gpu:
                    new_features[i] = new_features[i].cuda()

        y = Variable(torch.from_numpy(y))
        if self.gpu:
            y = y.cuda()
        return ((v_features_A, e_features_A), heuristics_A, new_features_A, (v_features_B, e_features_B), heuristics_B, new_features_B, y)

    def get_test_pairs(self, randomize_dir=True, return_id=False):
        '''
        return a list of two lists, X_A and X_B, as well as a list y
        each list consists of two lists, which are vertex and edge representations
        each list consists of #V or #E lists, which are individual vertices/edges
        each list consists of several N x feature_len torch Variables, which are individual features
        currently only keeping chains of length 4
        if for i-th problem, the good chain is in X_A, then y[i]==1, else y[i]==0
        '''
        N = len(self.test_pairs)
        v_features_A, e_features_A, heuristics_A, new_features_A = self.prepare_feature_placeholder(N)
        v_features_B, e_features_B, heuristics_B, new_features_B = self.prepare_feature_placeholder(N)
        y = np.zeros(N, dtype='int64')
        if return_id:
            ids = [[], []]

        for instance_idx in range(N):
            good, bad = self.test_pairs[instance_idx]
            if randomize_dir:
                good = good[:-1]+random.choice(['f','r'])
                bad = bad[:-1]+random.choice(['f','r'])
            v_good, e_good, h_good, n_good = self.get_features(good)
            v_bad, e_bad, h_bad, n_bad = self.get_features(bad)
            v_good, e_good, v_bad, e_bad = list(v_good), list(e_good), list(v_bad), list(e_bad)

            label = random.random()>0.5
            y[instance_idx] = label
            if return_id:
                if label:
                    ids[0].append(good)
                    ids[1].append(bad)
                else:
                    ids[0].append(bad)
                    ids[1].append(good)
            for v_idx in range(4):
                for v_fea_idx in range(len(v_good[v_idx])):
                    if label:
                        v_features_A[v_idx][v_fea_idx][instance_idx] = v_good[v_idx][v_fea_idx]
                        v_features_B[v_idx][v_fea_idx][instance_idx] = v_bad[v_idx][v_fea_idx]
                    else:
                        v_features_B[v_idx][v_fea_idx][instance_idx] = v_good[v_idx][v_fea_idx]
                        v_features_A[v_idx][v_fea_idx][instance_idx] = v_bad[v_idx][v_fea_idx]

            for e_idx in range(3):
                for e_fea_idx in range(len(e_good[e_idx])):
                    if label:
                        e_features_A[e_idx][e_fea_idx][instance_idx] = e_good[e_idx][e_fea_idx]
                        e_features_B[e_idx][e_fea_idx][instance_idx] = e_bad[e_idx][e_fea_idx]
                    else:
                        e_features_B[e_idx][e_fea_idx][instance_idx] = e_good[e_idx][e_fea_idx]
                        e_features_A[e_idx][e_fea_idx][instance_idx] = e_bad[e_idx][e_fea_idx]

            for h_idx in range(len(h_good)):
                if label:
                    heuristics_A[h_idx][instance_idx] = h_good[h_idx]
                    heuristics_B[h_idx][instance_idx] = h_bad[h_idx]
                else:
                    heuristics_B[h_idx][instance_idx] = h_good[h_idx]
                    heuristics_A[h_idx][instance_idx] = h_bad[h_idx]

            for n_idx in range(len(n_good)):
                if label:
                    new_features_A[n_idx][instance_idx] = n_good[n_idx]
                    new_features_B[n_idx][instance_idx] = n_bad[n_idx]
                else:
                    new_features_B[n_idx][instance_idx] = n_good[n_idx]
                    new_features_A[n_idx][instance_idx] = n_bad[n_idx]


        for features in [v_features_A, e_features_A, v_features_B, e_features_B]:
            for feature in features:
                for i in range(len(feature)):
                    feature[i] = Variable(torch.from_numpy(feature[i]))
                    if self.gpu:
                        feature[i] = feature[i].cuda()
        for heuristics in [heuristics_A, heuristics_B]:
            for i in range(len(heuristics)):
                heuristics[i] = Variable(torch.from_numpy(heuristics[i]))
                if self.gpu:
                    heuristics[i] = heuristics[i].cuda()

        for features in [new_features_A, new_features_B]:
            for i in range(len(features)):
                features[i] = Variable(torch.from_numpy(features[i]))
                if self.gpu:
                    features[i] = features[i].cuda()

        y = Variable(torch.from_numpy(y))
        if self.gpu:
            y = y.cuda()
        if not return_id:
            return (v_features_A, e_features_A), heuristics_A, new_features_A, (v_features_B, e_features_B), heuristics_B, new_features_B, y
        else:
            return (v_features_A, e_features_A), heuristics_A, new_features_A, (v_features_B, e_features_B), heuristics_B, new_features_B, y, ids

    def get_pairs_for_ids(self, ids):
        '''
        ids are list of (first_chain, second_chain) tuples
        return a list of two lists, X_A and X_B
        each list consists of two lists, which are vertex and edge representations
        each list consists of #V or #E lists, which are individual vertices/edges
        each list consists of several N x feature_len torch Variables, which are individual features
        currently only keeping chains of length 4
        '''
        N = len(ids)
        v_features_A, e_features_A, heuristics_A, new_features_A = self.prepare_feature_placeholder(N)
        v_features_B, e_features_B, heuristics_B, new_features_B = self.prepare_feature_placeholder(N)

        for instance_idx, (first, second) in enumerate(ids):
            v_first, e_first, h_first, n_first = self.get_features(first)
            v_second, e_second, h_second, n_second = self.get_features(second)

            for v_idx in range(4):
                for v_fea_idx in range(len(v_first[v_idx])):
                    v_features_A[v_idx][v_fea_idx][instance_idx] = v_first[v_idx][v_fea_idx]
                    v_features_B[v_idx][v_fea_idx][instance_idx] = v_second[v_idx][v_fea_idx]

            for e_idx in range(3):
                for e_fea_idx in range(len(e_first[e_idx])):
                    e_features_A[e_idx][e_fea_idx][instance_idx] = e_first[e_idx][e_fea_idx]
                    e_features_B[e_idx][e_fea_idx][instance_idx] = e_second[e_idx][e_fea_idx]

            for h_idx in range(len(h_first)):
                heuristics_A[h_idx][instance_idx] = h_first[h_idx]
                heuristics_B[h_idx][instance_idx] = h_second[h_idx]

            for n_idx in range(len(n_first)):
                new_features_A[n_idx][instance_idx] = n_first[n_idx]
                new_features_B[n_idx][instance_idx] = n_second[n_idx]

        for features in [v_features_A, e_features_A, v_features_B, e_features_B]:
            for feature in features:
                for i in range(len(feature)):
                    feature[i] = Variable(torch.from_numpy(feature[i]))
                    if self.gpu:
                        feature[i] = feature[i].cuda()
        for heuristics in [heuristics_A, heuristics_B]:
            for i in range(len(heuristics)):
                heuristics[i] = Variable(torch.from_numpy(heuristics[i]))
                if self.gpu:
                    heuristics[i] = heuristics[i].cuda()

        for features in [new_features_A, new_features_B]:
            for i in range(len(features)):
                features[i] = Variable(torch.from_numpy(features[i]))
                if self.gpu:
                    features[i] = features[i].cuda()

        return ((v_features_A, e_features_A), heuristics_A, new_features_A, (v_features_B, e_features_B), heuristics_B, new_features_B)


class FeaturesPreprocessor:
    def __init__(self, name, src_data_path, tgt_dir_path):
        self.name = name
        sampled_problems = pickle.load(open(
            '%s/paths.pkl'%src_data_path, 'rb'), encoding='latin1')
        self.texts = dict()
        print('loading problem plain texts')
        for id_num in sampled_problems:
            f_short = sampled_problems[id_num]['forward']['short']
            r_short = sampled_problems[id_num]['reverse']['short']
            self.texts[id_num+'f'] = f_short
            self.texts[id_num+'r'] = r_short
        self.tgt_dir_path = tgt_dir_path

    def _eval_path(self, id_):
        raise NotImplementedError()

    def calc_features(self):
        scores = dict()
        for id_ in self.texts:
            scores[id_] = self._eval_path(id_)
        with open('%s/%s.pkl'%(self.tgt_dir_path, self.name), 'wb') as file:
            pickle.dump(scores, file)

class L1vertexendsExtractor(FeaturesPreprocessor):
    def __init__(self, src_data_path, tgt_dir_path, v_enc_path):
        super(L1vertexendsExtractor, self).__init__('L1vertexends', src_data_path, tgt_dir_path)
        with open(v_enc_path, 'rb') as file:
            self.v_emb = pickle.load(file, encoding='latin1')
        self.dist = torch.nn.PairwiseDistance(p=1.0)

    def _eval_path(self, id_):
        emb_data = self.v_emb[id_]
        score = self.dist(torch.tensor(emb_data[0]), torch.tensor(emb_data[len(emb_data)-1]))
        return score.item()



class HeuristicPreprocessor:
    def __init__(self, name, src_data_path, tgt_dir_path):
        self.name = name
        sampled_problems = pickle.load(open(
            '%s/paths.pkl'%src_data_path, 'rb'), encoding='latin1')
        self.texts = dict()
        print('loading problem plain texts')
        for id_num in sampled_problems:
            f_short = sampled_problems[id_num]['forward']['short']
            r_short = sampled_problems[id_num]['reverse']['short']
            self.texts[id_num+'f'] = f_short
            self.texts[id_num+'r'] = r_short
        self.tgt_dir_path = tgt_dir_path

    def _eval_path(self, id_):
        raise NotImplementedError()

    def calc_heuristics(self):
        scores = dict()
        for id_ in self.texts:
            scores[id_] = self._eval_path(id_)
        with open('%s/%s.pkl'%(self.tgt_dir_path, self.name), 'wb') as file:
            pickle.dump(scores, file)


class STHeuristicsExtractor(HeuristicPreprocessor):
    def __init__(self, src_data_path, tgt_dir_path, v_enc_path):
        super(STHeuristicsExtractor, self).__init__('st', src_data_path, tgt_dir_path)
        with open(v_enc_path, 'rb') as file:
            self.v_emb = pickle.load(file, encoding='latin1')
        self.cos = torch.nn.CosineSimilarity(dim=0)

    def _eval_path(self, id_):
        emb_data = self.v_emb[id_]
        score = self.cos(torch.tensor(emb_data[0]), torch.tensor(emb_data[len(emb_data)-1]))
        return score.item()


class PairwiseHeuristicsExtractor(HeuristicPreprocessor):
    def __init__(self, src_data_path, tgt_dir_path, v_enc_path):
        super(PairwiseHeuristicsExtractor, self).__init__('pairwise', src_data_path, tgt_dir_path)
        with open(v_enc_path, 'rb') as file:
            self.v_emb = pickle.load(file, encoding='latin1')
        self.cos = torch.nn.CosineSimilarity(dim=0)

    def _eval_path(self, id_):
        emb_data = self.v_emb[id_]
        size = len(emb_data)
        total_score = 0
        for i in range(size-1):
            total_score += self.cos(torch.tensor(emb_data[i]), torch.tensor(emb_data[i+1])).item()
        return total_score / (size+1)


class RFHeuristicsExtractor(HeuristicPreprocessor):
    # simplified: product of degrees (for easier standardization, take log)
    def __init__(self, src_data_path, tgt_dir_path, v_deg_path):
        super(RFHeuristicsExtractor, self).__init__('rf', src_data_path, tgt_dir_path)
        with open(v_deg_path, 'rb') as file:
            self.v_deg = pickle.load(file, encoding='latin1')

    def _eval_path(self, id_):
        score = 0
        deg = self.v_deg[id_]
        for d in deg:
            score += np.log(d[0])
        return score


class LengthHeuristicsExtractor(HeuristicPreprocessor):
    def __init__(self, src_data_path, tgt_dir_path):
        super(LengthHeuristicsExtractor, self).__init__('length', src_data_path, tgt_dir_path)

    def _eval_path(self, id_):
        path = self.texts[id_]
        tokens = path.split()
        score = len(tokens) // 2 + 1
        return score


if __name__ == '__main__':
    # prepare heuristics
    sthe = STHeuristicsExtractor('../data/science', '../prepare_data/heuristics', '../prepare_data/features/v_enc_embedding.pkl')
    sthe.calc_heuristics()
    phe = PairwiseHeuristicsExtractor('../data/science', '../prepare_data/heuristics', '../prepare_data/features/v_enc_embedding.pkl')
    phe.calc_heuristics()
    rfhe = RFHeuristicsExtractor('../data/science', '../prepare_data/heuristics', '../prepare_data/features/v_deg.pkl')
    rfhe.calc_heuristics()
    lhe = LengthHeuristicsExtractor('../data/science', '../prepare_data/heuristics')
    lhe.calc_heuristics()

    # prepare new features
    L1vertexdist = L1vertexendsExtractor('../data/science', '../prepare_data/new_features', '../prepare_data/features/v_enc_embedding.pkl')
    L1vertexdist.calc_features()
