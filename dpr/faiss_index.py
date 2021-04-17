import numpy as np
from dpr.simple_mmap_dataset import Corpus
from util.args_help import fill_from_args
import faiss
from util.reporting import Reporting


def l2_convert_indexed_vectors(vectors: np.ndarray, max_norm_sqrd: float):
    norms = np.linalg.norm(vectors, axis=1)
    aux_dims_sqrd = max_norm_sqrd - norms * norms
    if np.min(aux_dims_sqrd) < 0:
        print(f'WARNING: max_norm_sqrd = {max_norm_sqrd} but it was less '
              f'({np.min(aux_dims_sqrd)}) than a vectors norm_sqrd')
        aux_dims_sqrd = np.maximum(aux_dims_sqrd, 0)
    aux_dims = np.sqrt(aux_dims_sqrd)
    converted_vectors = np.hstack((vectors, aux_dims.reshape(-1, 1)))
    return converted_vectors


def l2_convert_query_vectors(vectors: np.ndarray):
    aux_dim = np.zeros(vectors.shape[0], dtype=np.float32)
    converted_vectors = np.hstack((vectors, aux_dim.reshape(-1, 1)))
    return converted_vectors


class ANNIndex:
    def __init__(self, index_file):
        self.index = faiss.read_index(index_file)

    def search(self, query_vectors, k):
        if type(self.index) == faiss.IndexHNSWSQ:
            query_vectors = l2_convert_query_vectors(query_vectors)
        return self.index.search(query_vectors, k)

    def dim(self):
        if type(self.index) == faiss.IndexHNSWSQ:
            return self.index.d - 1
        else:
            return self.index.d


class IndexOptions():
    def __init__(self):
        self.d = 768
        self.m = 128
        self.ef_search = 128
        self.ef_construction = 200
        self.index_batch_size = 100000
        self.scalar_quantizer = -1


def build_index(corpus_dir, output_file, opts: IndexOptions):
    corpus = Corpus(corpus_dir)

    if opts.scalar_quantizer > 0:
        if opts.scalar_quantizer == 16:
            sq = faiss.ScalarQuantizer.QT_fp16
        elif opts.scalar_quantizer == 8:
            sq = faiss.ScalarQuantizer.QT_8bit
        elif opts.scalar_quantizer == 4:
            sq = faiss.ScalarQuantizer.QT_4bit
        elif opts.scalar_quantizer == 6:
            sq = faiss.ScalarQuantizer.QT_6bit
        else:
            raise ValueError(f'unknown --scalar_quantizer {opts.scalar_quantizer}')
        # see https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py
        index = faiss.IndexHNSWSQ(opts.d+1, sq, opts.m)
    else:
        index = faiss.IndexHNSWFlat(opts.d, opts.m, faiss.METRIC_INNER_PRODUCT)

    # defaults are 16 and 40
    # print(f'ef search and construction: {index.hnsw.efSearch}; {index.hnsw.efConstruction}')
    index.hnsw.efSearch = opts.ef_search
    index.hnsw.efConstruction = opts.ef_construction

    if opts.scalar_quantizer > 0:
        max_norm = 0
        for psg in corpus:
            vector = psg['vector']
            max_norm = max(max_norm, np.linalg.norm(vector))
        print(f'found max norm = {max_norm}')
        max_norm_sqrd = max_norm * max_norm
    else:
        max_norm_sqrd = None

    vectors = np.zeros((opts.index_batch_size, opts.d), dtype=np.float32)
    vector_ndx = 0
    opts.is_trained = False

    def add_to_index(vectors):
        if opts.scalar_quantizer > 0:
            to_index = l2_convert_indexed_vectors(vectors, max_norm_sqrd)
            if not opts.is_trained:
                index.train(to_index)
                opts.is_trained = True
        else:
            to_index = vectors
        index.add(to_index)

    report = Reporting()
    for psg in corpus:
        if report.is_time():
            print(report.progress_str(instance_name='vector'))
        vector = psg['vector']
        vectors[vector_ndx] = vector
        vector_ndx += 1
        if vector_ndx == opts.index_batch_size:
            add_to_index(vectors)
            vector_ndx = 0
    if vector_ndx > 0:
        add_to_index(vectors[:vector_ndx])

    print(f'finished building index, writing index file to {output_file}')
    faiss.write_index(index, output_file)


if __name__ == "__main__":
    class CmdOptions(IndexOptions):
        def __init__(self):
            super().__init__()
            self.corpus_dir = ''
            self.output_file = ''
            self.__required_args__ = ['corpus_dir', 'output_file']

    opts = CmdOptions()
    fill_from_args(opts)

    build_index(opts.corpus_dir, opts.output_file, opts)
