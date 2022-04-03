import pickle
import numpy as np

import torch


class HeuristicPreprocessor:
    def __init__(self, name, src_data_path, tgt_dir_path):
        self.name = name
        sampled_problems = pickle.load(open('%s/paths.pkl' % src_data_path,
                                            'rb'),
                                       encoding='latin1')
        self.texts = dict()
        print('loading problem plain texts')
        for id_num in sampled_problems:
            f_short = sampled_problems[id_num]['forward']['short']
            r_short = sampled_problems[id_num]['reverse']['short']
            self.texts[id_num + 'f'] = f_short
            self.texts[id_num + 'r'] = r_short
        self.tgt_dir_path = tgt_dir_path

    def _eval_path(self, id_):
        raise NotImplementedError()

    def calc_heuristics(self):
        scores = dict()
        for id_ in self.texts:
            scores[id_] = self._eval_path(id_)
        with open('%s/%s.pkl' % (self.tgt_dir_path, self.name), 'wb') as file:
            pickle.dump(scores, file)


class STHeuristicsExtractor(HeuristicPreprocessor):
    def __init__(self, src_data_path, tgt_dir_path, v_enc_path):
        super().__init__('st', src_data_path, tgt_dir_path)
        with open(v_enc_path, 'rb') as file:
            self.v_emb = pickle.load(file, encoding='latin1')
        self.cos = torch.nn.CosineSimilarity(dim=0)

    def _eval_path(self, id_):
        emb_data = self.v_emb[id_]
        score = self.cos(torch.tensor(emb_data[0]),
                         torch.tensor(emb_data[len(emb_data) - 1]))
        return score.item()


class PairwiseHeuristicsExtractor(HeuristicPreprocessor):
    def __init__(self, src_data_path, tgt_dir_path, v_enc_path):
        super().__init__('pairwise', src_data_path, tgt_dir_path)
        with open(v_enc_path, 'rb') as file:
            self.v_emb = pickle.load(file, encoding='latin1')
        self.cos = torch.nn.CosineSimilarity(dim=0)

    def _eval_path(self, id_):
        emb_data = self.v_emb[id_]
        size = len(emb_data)
        total_score = 0
        for i in range(size - 1):
            total_score += self.cos(torch.tensor(emb_data[i]),
                                    torch.tensor(emb_data[i + 1])).item()
        return total_score / (size + 1)


class RFHeuristicsExtractor(HeuristicPreprocessor):
    # simplified: product of degrees (for easier standardization, take log)
    def __init__(self, src_data_path, tgt_dir_path, v_deg_path):
        super(RFHeuristicsExtractor, self).__init__('rf', src_data_path,
                                                    tgt_dir_path)
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
        super().__init__('length', src_data_path, tgt_dir_path)

    def _eval_path(self, id_):
        path = self.texts[id_]
        tokens = path.split()
        score = len(tokens) // 2 + 1
        return score


if __name__ == '__main__':
    # prepare heuristics
    sthe = STHeuristicsExtractor(
        '../data/science', '../prepare_data/heuristics',
        '../prepare_data/features/v_enc_embedding.pkl')
    sthe.calc_heuristics()
    phe = PairwiseHeuristicsExtractor(
        '../data/science', '../prepare_data/heuristics',
        '../prepare_data/features/v_enc_embedding.pkl')
    phe.calc_heuristics()
    rfhe = RFHeuristicsExtractor('../data/science',
                                 '../prepare_data/heuristics',
                                 '../prepare_data/features/v_deg.pkl')
    rfhe.calc_heuristics()
    lhe = LengthHeuristicsExtractor('../data/science',
                                    '../prepare_data/heuristics')
    lhe.calc_heuristics()