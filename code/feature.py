# 1. [DONE by Wang Qian] Create L2 feature extractor
# 2. [DONE by Tian Fang] Create L1 feature extractor

import torch, pickle


class FeaturesPreprocessor:
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

    def calc_features(self):
        scores = dict()
        for id_ in self.texts:
            scores[id_] = self._eval_path(id_)
        with open('%s/%s.pkl' % (self.tgt_dir_path, self.name), 'wb') as file:
            pickle.dump(scores, file)


class EdgeL1DistanceExtractor(FeaturesPreprocessor):
    def __init__(self, src_data_path, tgt_dir_path, v_enc_path):
        super().__init__('e_vertexl1dist', src_data_path, tgt_dir_path)
        with open(v_enc_path, 'rb') as file:
            self.v_emb = pickle.load(file, encoding='latin1')

    def _eval_path(self, id_):
        emb_data = self.v_emb[id_]
        # each score is an embedding of edge's two ends' L1 distance
        size = len(emb_data)
        scores = [[], [], []]

        for i in range(size - 1):
            scores[i] = [
                torch.dist(torch.tensor(emb_data[i]),
                           torch.tensor(emb_data[i + 1]), 1).item()
            ]
        return scores


class EdgeL2DistanceExtractor(FeaturesPreprocessor):
    def __init__(self, src_data_path, tgt_dir_path, v_enc_path):
        super().__init__('e_vertexl2dist', src_data_path, tgt_dir_path)
        with open(v_enc_path, 'rb') as file:
            self.v_emb = pickle.load(file, encoding='latin1')

    def _eval_path(self, id_):
        emb_data = self.v_emb[id_]
        # each score is an embedding of edge's two ends' L2 distance
        size = len(emb_data)
        scores = [[], [], []]

        for i in range(size - 1):
            scores[i] = [
                torch.dist(torch.tensor(emb_data[i]),
                           torch.tensor(emb_data[i + 1]), 2).item()
            ]
        return scores


if __name__ == '__main__':
    L1_dist = EdgeL1DistanceExtractor(
        '../data/science', '../prepare_data/features',
        '../prepare_data/features/v_enc_embedding.pkl')
    L1_dist.calc_features()
    L2_dist = EdgeL2DistanceExtractor(
        '../data/science', '../prepare_data/features',
        '../prepare_data/features/v_enc_embedding.pkl')
    L2_dist.calc_features()
