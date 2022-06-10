from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, shuffled_writer
import jnius_config
import ujson as json
from generation.rag_util import _extractive_normalize as normalize
import multiprocessing
from multiprocessing.pool import ThreadPool
import functools
import logging
from util.reporting import Reporting
import math

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.num_processes = multiprocessing.cpu_count()
        self.max_negatives = 5  # number of negatives to gather for each positive
        self.allow_answer_bearing = False  # for datasets like FEVER, it doesn't matter if the passage contains 'SUPPORTS' or 'REFUTES'
        self.max_instance_expansion = 5  # create at most this many training instances per original instance (ignoring the other positives)


class Args(Options):
    def __init__(self):
        super().__init__()
        self.jar = ''
        self.anserini_index = ''
        self.positive_pids = ''
        self.output_dir = ''
        self.__required_args__ = ['anserini_index', 'positive_pids', 'jar', 'output_dir']


def _retrieve_one(query_tuple, searcher, JString, negatives_per_positive, max_instance_expansion):
    inst_id, query, positive_pids, positive_context, answers = query_tuple
    normed_positives = [' ' + normalize(p['title'] + '  ' + p['text']) + ' ' for p in positive_context]
    target_num_negatives = negatives_per_positive * min(max_instance_expansion, len(positive_pids))
    # initially fetch a factor of 3 more than we need, since we filter out some
    hits = searcher.search(JString(query.encode('utf-8')), 3 * target_num_negatives)
    negs = []
    for hit in hits:
        pid = hit.docid
        # CONSIDER: if max_bm25_score > 0 and hit.score >= max_bm25_score: continue
        if pid in positive_pids:
            continue
        title = hit.content[:hit.content.find('\n\n')]
        text = hit.content[hit.content.find('\n\n') + 2:]
        norm_context = ' ' + normalize(title + ' ' + text) + ' '
        if norm_context in normed_positives:
            continue  # exclude any passage with the same text as a positive from negatives
        answer_bearing = any([ans in norm_context for ans in answers])
        if answer_bearing:
            continue  # exclude answer bearing passages from negatives
        negs.append({'pid': pid, 'title': title, 'text': text})
        if len(negs) >= target_num_negatives:
            break
    return negs


class BM25forDPR:
    def __init__(self, jar: str, index: str, opts: Options):
        jnius_config.set_classpath(jar)
        from jnius import autoclass
        self.JString = autoclass('java.lang.String')
        JSearcher = autoclass('io.anserini.search.SimpleSearcher')
        self.searcher = JSearcher(self.JString(index))
        self.opts = opts
        # NOTE: only thread-based pooling works with the JSearcher
        self.pool = ThreadPool(processes=opts.num_processes)
        logger.info(f'Using multiprocessing pool with {opts.num_processes} workers')
        self.no_negative_skip_count = 0
        self._retrieve_one = functools.partial(_retrieve_one,
                                               searcher=self.searcher, JString=self.JString,
                                               negatives_per_positive=opts.max_negatives,
                                               max_instance_expansion=opts.max_instance_expansion)
        self.written = 0

    def _write_batch(self, out, query_tuples, all_negs):

        for query_tuple, negs in zip(query_tuples, all_negs):
            inst_id, query, positive_pids, positive_context, answers = query_tuple

            if len(negs) == 0:
                if self.no_negative_skip_count == 0:
                    logger.warning(f'No negatives for "{query}"\n   Answers: {answers}')
                self.no_negative_skip_count += 1
                continue

            num_instances = min(len(positive_context),
                                self.opts.max_instance_expansion,
                                int(math.ceil(len(negs) / self.opts.max_negatives)))
            for pndx, pos in enumerate(positive_context[:num_instances]):
                out.write(json.dumps({'id': inst_id, 'query': query,
                                      'positive': pos,
                                      'negatives': negs[pndx::num_instances][:self.opts.max_negatives]}) + '\n')
                self.written += 1

    def create(self, positive_pids_file, output_dir):
        report = Reporting()
        batch_size = 1024
        with shuffled_writer(output_dir) as out:
            query_tuples = []
            for line in jsonl_lines(positive_pids_file):
                jobj = json.loads(line)
                inst_id = jobj['id']
                query = jobj['query']
                positive_pids = jobj['positive_pids']
                positive_context = jobj['positive_passages']
                if self.opts.allow_answer_bearing:
                    answers = []
                else:
                    answers = [normalize(a) if len(normalize(a)) > 0 else a.strip() for a in jobj['answers']]
                    answers = [' '+a+' ' for a in answers if a]
                query_tuples.append((inst_id, query, positive_pids, positive_context, answers))
                if len(query_tuples) >= batch_size:
                    all_negs = self.pool.map(self._retrieve_one, query_tuples)
                    self._write_batch(out, query_tuples, all_negs)
                    query_tuples = []
                    if report.is_time():
                        instance_count = report.check_count*batch_size
                        logger.info(f'On instance {instance_count}, '
                                    f'{instance_count/report.elapsed_seconds()} instances per second')
                        if self.no_negative_skip_count > 0:
                            logger.info(f'{self.no_negative_skip_count} skipped for lack of negatives')

            if len(query_tuples) > 0:
                all_negs = self.pool.map(self._retrieve_one, query_tuples)
                self._write_batch(out, query_tuples, all_negs)
            instance_count = report.check_count * batch_size
            logger.info(f'Finished {instance_count} instances; wrote {self.written} training triples. '
                        f'{instance_count/report.elapsed_seconds()} instances per second')
            if self.no_negative_skip_count > 0:
                logger.info(f'{self.no_negative_skip_count} skipped for lack of negatives')


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = Args()
    fill_from_args(args)
    bm25dpr = BM25forDPR(args.jar, args.anserini_index, args)
    bm25dpr.create(args.positive_pids, args.output_dir)
