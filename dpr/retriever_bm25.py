from transformers import RagTokenizer, RagConfig
import logging
import jnius_config
import multiprocessing
import functools

logger = logging.getLogger(__name__)


class BM25Hypers():
    def __init__(self):
        super().__init__()
        self.jar = ''
        self.anserini_index = ''
        self.n_docs = 10
        self.num_processes = multiprocessing.cpu_count()


def _retrieve_one(query_exclude, searcher, hypers, JString):
    query, exclude = query_exclude
    hits = searcher.search(JString(query.encode('utf-8')), hypers.n_docs + len(exclude))
    hits = [hit for hit in hits if hit.content not in exclude][:hypers.n_docs]
    if len(hits) == 0:
        # create dummy docs if no result
        doc_scores = [0.0] * hypers.n_docs
        docs = {'pid': ['N/A:0'] * hypers.n_docs,
                'title': ['title'] * hypers.n_docs,
                'text': ['text'] * hypers.n_docs}
        logger.warning(f'No results for {query}!')
        return doc_scores, docs
    if len(hits) < hypers.n_docs:
        # duplicate last doc if too few results
        logger.warning(f'Too few results for {query}! ({len(hits)})')
        hits.extend([hits[-1]] * (hypers.n_docs - len(hits)))
    assert len(hits) == hypers.n_docs
    doc_scores = [hit.score for hit in hits]
    titles = [hit.content[:hit.content.find('\n\n')] for hit in hits]
    texts = [hit.content[hit.content.find('\n\n') + 2:] for hit in hits]
    docs = {'pid': [hit.docid for hit in hits], 'title': titles, 'text': texts}
    return doc_scores, docs


class RetrieverBM25():
    def __init__(self, hypers: BM25Hypers):
        """
        :param hypers:
        """
        self.hypers = hypers
        jnius_config.set_classpath(hypers.jar)
        from jnius import autoclass
        self.JString = autoclass('java.lang.String')
        JSearcher = autoclass('io.anserini.search.SimpleSearcher')
        self.searcher = JSearcher(self.JString(hypers.anserini_index))
        self.config = RagConfig.from_pretrained('facebook/rag-token-nq')
        self.tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq', config=self.config)
        if hypers.num_processes > 1:
            # NOTE: only thread-based pooling works with the JSearcher
            self.pool = multiprocessing.pool.ThreadPool(processes=hypers.num_processes)
            logger.info(f'Using multiprocessing pool with {hypers.num_processes} workers')
        else:
            self.pool = None
        self._retrieve_one = functools.partial(_retrieve_one,
                                               searcher=self.searcher, hypers=self.hypers, JString=self.JString)

    def retrieve_forward(self, queries, *, exclude_by_content=None):
        """

        :param queries: list of queries to retrieve documents for
        :return: input for RAG: context_input_ids, context_attention_mask, doc_scores
          also docs and info-for-backward (when calling retrieve_backward)
        """
        if exclude_by_content is None:
            exclude_by_content = [set() for _ in range(len(queries))]
        else:
            assert len(exclude_by_content) == len(queries)

        if self.pool is not None:
            result_batch = self.pool.map(self._retrieve_one, zip(queries, exclude_by_content))
            docs = [r[1] for r in result_batch]
            doc_scores = [r[0] for r in result_batch]
        else:
            docs = []
            doc_scores = []
            for query, exclude in zip(queries, exclude_by_content):
                doc_scores_i, docs_i = self._retrieve_one((query, exclude))
                doc_scores.append(doc_scores_i)
                docs.append(docs_i)

        return doc_scores, docs
