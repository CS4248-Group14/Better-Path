# `dataset.py` is adapted from `code/science/dataset.py`.
# Main changes:
# 1. [DONE by Tian Fang] Extract methods related to loading pre-generated files, and fix Zhou's mistake of not closing file descriptor.
# 2. [DONE by Wang Qian] New features for edges: vertex norm

import pickle, random
import numpy as np
from itertools import cycle

import torch
from torch.autograd import Variable

from constants import RANDOM_SEED

all_feature_lengths = {
    'v_enc_onehot': 100,
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
    'e_vertexl1dist': 1,
    'e_vertexl2dist': 1,
    'e_dir': 3,
    'e_rel': 46,
    'e_weight': 1,
    'e_source': 6,
    'e_weightsource': 6,
    'e_srank_abs': 1,
    'e_srank_rel': 1,
    'e_trank_abs': 1,
    'e_trank_rel': 1,
    'e_sense': 1
}


class Dataset:
    def load_features(self, features_src_path):
        self.cached_features = dict()
        for feat_name in self.feature_names:
            print('loading', feat_name)
            with open(f'{features_src_path}/{feat_name}.pkl', 'rb') as f:
                self.cached_features[feat_name] = pickle.load(
                    f, encoding='latin1')

    def load_paths(self, src_data_path):
        self.texts = dict()
        with open(f'{src_data_path}/paths.pkl', 'rb') as f:
            sampled_problems = pickle.load(f, encoding='latin1')
            print('loading problem plain texts')
            for id_num in sampled_problems:
                f_short = sampled_problems[id_num]['forward']['short']
                r_short = sampled_problems[id_num]['reverse']['short']
                self.texts[id_num + 'f'] = f_short
                self.texts[id_num + 'r'] = r_short
            print(f'There are {len(self.texts)} sampled paths')

    def load_labeled_pairs(self, src_data_path):
        print('loading labeled pairs')
        self.all_pairs = []  # list of id tuples (good, bad)

        with open(f'{src_data_path}/answers.txt') as f:
            for l in f:
                first, second, good = l.strip().split('_')
                if first == good:
                    bad = second
                elif second == good:
                    bad = first
                g_len = (len(self.texts[good].strip().split(' ')) + 1) / 2
                b_len = (len(self.texts[bad].strip().split(' ')) + 1) / 2
                # We only keep paths with length 4
                if g_len != 4 or b_len != 4:
                    continue
                self.all_pairs.append((good, bad))
        random.seed(RANDOM_SEED)
        random.shuffle(self.all_pairs)

    def load_heuristics(self, heuristic_names, heuristics_src_path):
        self.heuristic_names = heuristic_names
        self.cached_heuristics = dict()
        for h in self.heuristic_names:
            with open(f'{heuristics_src_path}/{h}.pkl', 'rb') as f:
                print('loading heuristic:', h)
                self.cached_heuristics[h] = pickle.load(f)

    def __init__(self,
                 feature_names,
                 heuristic_names,
                 train_test_split_fraction,
                 use_gpu,
                 features_src_path='../prepare_data/features',
                 heuristics_src_path='../prepare_data/heuristics',
                 src_data_path='../data/science'):
        self.feature_names = feature_names
        self.use_gpu = use_gpu
        self.load_features(features_src_path)
        self.load_paths(src_data_path)
        self.load_labeled_pairs(src_data_path)
        self.load_heuristics(heuristic_names, heuristics_src_path)

        # train-test split
        split = int(train_test_split_fraction * len(self.all_pairs))
        self.train_pairs = self.all_pairs[:split]
        self.test_pairs = self.all_pairs[split:]
        print(f'There are {len(self.train_pairs)} training pairs')
        print(f'There are {len(self.test_pairs)} test pairs')
        # we resample from the training pairs
        self.cycled_train_pairs = cycle(self.train_pairs[:])

    def get_fea_len(self):
        return [all_feature_lengths[f] for f in self.feature_names]

    def get_v_fea_len(self):
        """
        Get vertex feature lengths corresponding to the input feature_names specified during initialization
        """
        return [
            all_feature_lengths[f] for f in self.feature_names
            if f.startswith('v')
        ]

    def get_e_fea_len(self):
        """
        Get edge feature lengths corresponding to the input feature_names specified during initialization
        """
        return [
            all_feature_lengths[f] for f in self.feature_names
            if f.startswith('e')
        ]

    def get_chain_len(self, id):
        return len(self.get_features(id)[0])

    def get_features(self, id):
        """
        Get features and heuristics of the path with the given id
        
        :param id: id of the path
        :return: a tuple containing (vertex features, edge features, heuristics)
        """
        v_features = []
        e_features = []
        for f in self.feature_names:
            if f.startswith('v'):
                v_features.append(self.cached_features[f][id])
            else:
                e_features.append(self.cached_features[f][id])
        v_features = zip(*v_features)
        e_features = zip(*e_features)
        heuristics = [
            self.cached_heuristics[h][id] for h in self.heuristic_names
        ]
        return v_features, e_features, heuristics

    def prepare_feature_placeholder(self, N):
        v_features = [[], [], [], []]
        e_features = [[], [], []]
        heuristics = []
        for feature in v_features:
            for f in self.feature_names:
                if f.startswith('v'):
                    feature.append(
                        np.zeros((N, all_feature_lengths[f]), dtype='float32'))
        for feature in e_features:
            for f in self.feature_names:
                if f.startswith('e'):
                    feature.append(
                        np.zeros((N, all_feature_lengths[f]), dtype='float32'))
        for h in self.heuristic_names:
            heuristics.append(np.zeros((N, 1), dtype='float32'))
        return v_features, e_features, heuristics

    def get_train_pairs(self, N, randomize_dir=True):
        '''
        return a list of two lists, X_A and X_B, as well as a list y
        each list consists of two lists, which are vertex and edge representations
        each list consists of #V or #E lists, which are individual vertices/edges
        each list consists of several N x feature_len torch Variables, which are individual features
        currently only keeping chains of length 4
        if for i-th problem, the good chain is in X_A, then y[i]==1, else y[i]==0
        '''
        v_features_A, e_features_A, heuristics_A = self.prepare_feature_placeholder(
            N)
        v_features_B, e_features_B, heuristics_B = self.prepare_feature_placeholder(
            N)
        y = np.zeros(N, dtype='int64')

        for instance_idx in range(N):
            good, bad = next(self.cycled_train_pairs)
            if randomize_dir:
                good = good[:-1] + random.choice(['f', 'r'])
                bad = bad[:-1] + random.choice(['f', 'r'])
            v_good, e_good, h_good = self.get_features(good)
            v_bad, e_bad, h_bad = self.get_features(bad)
            v_good, e_good, v_bad, e_bad = list(v_good), list(e_good), list(
                v_bad), list(e_bad)

            label = random.random() > 0.5
            y[instance_idx] = label
            for v_idx in range(4):
                for v_fea_idx in range(len(v_good[v_idx])):
                    if label:
                        v_features_A[v_idx][v_fea_idx][instance_idx] = v_good[
                            v_idx][v_fea_idx]
                        v_features_B[v_idx][v_fea_idx][instance_idx] = v_bad[
                            v_idx][v_fea_idx]
                    else:
                        v_features_B[v_idx][v_fea_idx][instance_idx] = v_good[
                            v_idx][v_fea_idx]
                        v_features_A[v_idx][v_fea_idx][instance_idx] = v_bad[
                            v_idx][v_fea_idx]

            for e_idx in range(3):
                for e_fea_idx in range(len(e_good[e_idx])):
                    if label:
                        e_features_A[e_idx][e_fea_idx][instance_idx] = e_good[
                            e_idx][e_fea_idx]
                        e_features_B[e_idx][e_fea_idx][instance_idx] = e_bad[
                            e_idx][e_fea_idx]
                    else:
                        e_features_B[e_idx][e_fea_idx][instance_idx] = e_good[
                            e_idx][e_fea_idx]
                        e_features_A[e_idx][e_fea_idx][instance_idx] = e_bad[
                            e_idx][e_fea_idx]

            for h_idx in range(len(h_good)):
                if label:
                    heuristics_A[h_idx][instance_idx] = h_good[h_idx]
                    heuristics_B[h_idx][instance_idx] = h_bad[h_idx]
                else:
                    heuristics_B[h_idx][instance_idx] = h_good[h_idx]
                    heuristics_A[h_idx][instance_idx] = h_bad[h_idx]

        for features in [
                v_features_A, e_features_A, v_features_B, e_features_B
        ]:
            for feature in features:
                for i in range(len(feature)):
                    feature[i] = Variable(torch.from_numpy(feature[i]))
                    if self.use_gpu:
                        feature[i] = feature[i].cuda()
        for heuristics in [heuristics_A, heuristics_B]:
            for i in range(len(heuristics)):
                heuristics[i] = Variable(torch.from_numpy(heuristics[i]))
                if self.use_gpu:
                    heuristics[i] = heuristics[i].cuda()
        y = Variable(torch.from_numpy(y))
        if self.use_gpu:
            y = y.cuda()
        return ((v_features_A, e_features_A), heuristics_A,
                (v_features_B, e_features_B), heuristics_B, y)

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
        v_features_A, e_features_A, heuristics_A = self.prepare_feature_placeholder(
            N)
        v_features_B, e_features_B, heuristics_B = self.prepare_feature_placeholder(
            N)
        y = np.zeros(N, dtype='int64')
        if return_id:
            ids = [[], []]

        for instance_idx in range(N):
            good, bad = self.test_pairs[instance_idx]
            if randomize_dir:
                good = good[:-1] + random.choice(['f', 'r'])
                bad = bad[:-1] + random.choice(['f', 'r'])
            v_good, e_good, h_good = self.get_features(good)
            v_bad, e_bad, h_bad = self.get_features(bad)
            v_good, e_good, v_bad, e_bad = list(v_good), list(e_good), list(
                v_bad), list(e_bad)

            label = random.random() > 0.5
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
                        v_features_A[v_idx][v_fea_idx][instance_idx] = v_good[
                            v_idx][v_fea_idx]
                        v_features_B[v_idx][v_fea_idx][instance_idx] = v_bad[
                            v_idx][v_fea_idx]
                    else:
                        v_features_B[v_idx][v_fea_idx][instance_idx] = v_good[
                            v_idx][v_fea_idx]
                        v_features_A[v_idx][v_fea_idx][instance_idx] = v_bad[
                            v_idx][v_fea_idx]

            for e_idx in range(3):
                for e_fea_idx in range(len(e_good[e_idx])):
                    if label:
                        e_features_A[e_idx][e_fea_idx][instance_idx] = e_good[
                            e_idx][e_fea_idx]
                        e_features_B[e_idx][e_fea_idx][instance_idx] = e_bad[
                            e_idx][e_fea_idx]
                    else:
                        e_features_B[e_idx][e_fea_idx][instance_idx] = e_good[
                            e_idx][e_fea_idx]
                        e_features_A[e_idx][e_fea_idx][instance_idx] = e_bad[
                            e_idx][e_fea_idx]

            for h_idx in range(len(h_good)):
                if label:
                    heuristics_A[h_idx][instance_idx] = h_good[h_idx]
                    heuristics_B[h_idx][instance_idx] = h_bad[h_idx]
                else:
                    heuristics_B[h_idx][instance_idx] = h_good[h_idx]
                    heuristics_A[h_idx][instance_idx] = h_bad[h_idx]

        for features in [
                v_features_A, e_features_A, v_features_B, e_features_B
        ]:
            for feature in features:
                for i in range(len(feature)):
                    feature[i] = Variable(torch.from_numpy(feature[i]))
                    if self.use_gpu:
                        feature[i] = feature[i].cuda()
        for heuristics in [heuristics_A, heuristics_B]:
            for i in range(len(heuristics)):
                heuristics[i] = Variable(torch.from_numpy(heuristics[i]))
                if self.use_gpu:
                    heuristics[i] = heuristics[i].cuda()
        y = Variable(torch.from_numpy(y))
        if self.use_gpu:
            y = y.cuda()
        if not return_id:
            return (v_features_A, e_features_A), heuristics_A, (
                v_features_B, e_features_B), heuristics_B, y
        else:
            return (v_features_A, e_features_A), heuristics_A, (
                v_features_B, e_features_B), heuristics_B, y, ids

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
        v_features_A, e_features_A, heuristics_A = self.prepare_feature_placeholder(
            N)
        v_features_B, e_features_B, heuristics_B = self.prepare_feature_placeholder(
            N)

        for instance_idx, (first, second) in enumerate(ids):
            v_first, e_first, h_first = self.get_features(first)
            v_second, e_second, h_second = self.get_features(second)

            for v_idx in range(4):
                for v_fea_idx in range(len(v_first[v_idx])):
                    v_features_A[v_idx][v_fea_idx][instance_idx] = v_first[
                        v_idx][v_fea_idx]
                    v_features_B[v_idx][v_fea_idx][instance_idx] = v_second[
                        v_idx][v_fea_idx]

            for e_idx in range(3):
                for e_fea_idx in range(len(e_first[e_idx])):
                    e_features_A[e_idx][e_fea_idx][instance_idx] = e_first[
                        e_idx][e_fea_idx]
                    e_features_B[e_idx][e_fea_idx][instance_idx] = e_second[
                        e_idx][e_fea_idx]

            for h_idx in range(len(h_first)):
                heuristics_A[h_idx][instance_idx] = h_first[h_idx]
                heuristics_B[h_idx][instance_idx] = h_second[h_idx]

        for features in [
                v_features_A, e_features_A, v_features_B, e_features_B
        ]:
            for feature in features:
                for i in range(len(feature)):
                    feature[i] = Variable(torch.from_numpy(feature[i]))
                    if self.use_gpu:
                        feature[i] = feature[i].cuda()
        for heuristics in [heuristics_A, heuristics_B]:
            for i in range(len(heuristics)):
                heuristics[i] = Variable(torch.from_numpy(heuristics[i]))
                if self.use_gpu:
                    heuristics[i] = heuristics[i].cuda()
        return ((v_features_A, e_features_A), heuristics_A,
                (v_features_B, e_features_B), heuristics_B)
