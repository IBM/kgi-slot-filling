# KGI (Knowledge Graph Induction) for slot filling
This is the code for our KILT leaderboard submission to the T-REx and zsRE tasks.  It includes code for training a DPR model then continuing training with RAG.


KGI model is described in: [Robust Retrieval Augmented Generation for Zero-shot Slot Filling](https://aclanthology.org/2021.emnlp-main.148/) (EMNLP 2021).  

# Available from Hugging Face as:
| Dataset | Type | Model Name | Tokenizer Name |
| ------- | ----- | ---- | --------- |
| T-REx   |  DPR (ctx)  | michaelrglass/dpr-ctx_encoder-multiset-base-kgi0-trex | facebook/dpr-ctx_encoder-multiset-base
| T-REx   |  RAG  | michaelrglass/rag-token-nq-kgi0-trex | rag-token-nq
| zsRE    |  DPR (ctx)  | michaelrglass/dpr-ctx_encoder-multiset-base-kgi0-zsre | facebook/dpr-ctx_encoder-multiset-base
| zsRE    |  RAG  | michaelrglass/rag-token-nq-kgi0-zsre | rag-token-nq

# Process to reproduce
Download the [KILT data and knowledge source](https://github.com/facebookresearch/KILT)
* T-REx: [train](http://dl.fbaipublicfiles.com/KILT/trex-train-kilt.jsonl), [dev](http://dl.fbaipublicfiles.com/KILT/trex-dev-kilt.jsonl)
* zsRE: [train](http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-train-kilt.jsonl), [dev](http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-dev-kilt.jsonl)
* [KILT Knowledge Source](http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json)

Segment the KILT Knowledge Source into passages:
```bash
python slot_filling/kilt_passage_corpus.py \
--kilt_corpus kilt_knowledgesource.json --output_dir kilt_passages --passage_ids passage_ids.txt
```

Generate the first phase of the DPR training data
```bash
python dpr/dpr_kilt_slot_filling_dataset.py \
--kilt_data structured_zeroshot-train-kilt.jsonl \
--passage_ids passage_ids.txt \
--output_file zsRE_train_positive_pids.jsonl

python dpr/dpr_kilt_slot_filling_dataset.py \
--kilt_data trex-train-kilt.jsonl \
--passage_ids passage_ids.txt \
--output_file trex_train_positive_pids.jsonl
```

Download and build [Anserini](https://github.com/castorini/anserini). 
You will need to have [Maven](https://maven.apache.org/index.html) and a [Java JDK](https://jdk.java.net/).
```bash
git clone https://github.com/castorini/anserini.git
cd anserini
# to use the 0.4.1 version dprBM25.jar is built for
git checkout 3a60106fdc83473d147218d78ae7dca7c3b6d47c
export JAVA_HOME=your JDK directory
mvn clean package appassembler:assemble
```

put the title/text into the training instance with hard negatives from BM25
```bash
python dpr/anserini_prep.py \
--input kilt_passages \
--output anserini_passages

sh Anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
-generator LuceneDocumentGenerator -threads 40 -input anserini_passages \
-index anserini_passage_index -storePositions -storeDocvectors -storeRawDocs

export CLASSPATH=jar/dprBM25.jar:Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar
java com.ibm.research.ai.pretraining.retrieval.DPRTrainingData \
-passageIndex anserini_passage_index \
-positivePidData ${dataset}_train_positive_pids.jsonl \
-trainingData ${dataset}_dpr_training_data.jsonl
```

Train DPR
```bash
# multi-gpu is not well supported
export CUDA_VISIBLE_DEVICES=0

python dpr/biencoder_trainer.py \
--train_dir zsRE_dpr_training_data.jsonl \
--output_dir models/DPR/zsRE \
--num_train_epochs 2 \
--num_instances 131610 \
--encoder_gpu_train_limit 32 \
--full_train_batch_size 128 \
--max_grad_norm 1.0 --learning_rate 5e-5

python dpr/biencoder_trainer.py \
--train_dir trex_dpr_training_data.jsonl \
--output_dir models/DPR/trex \
--num_train_epochs 2 \
--num_instances 2207953 \
--encoder_gpu_train_limit 32 \
--full_train_batch_size 128 \
--max_grad_norm 1.0 --learning_rate 5e-5
```

Put the trained DPR query encoder into the NQ RAG model (dataset = trex, zsRE)
```bash
python dpr/prepare_rag_model.py \
--save_dir models/RAG/${dataset}_dpr_rag_init  \
--qry_encoder_path models/DPR/${dataset}/qry_encoder
```

Encode the passages (dataset = trex, zsRE)
```bash
python dpr/index_simple_corpus.py \
--embed 1of2 \
--dpr_ctx_encoder_path models/DPR/${dataset}/ctx_encoder \
--corpus kilt_passages  \
--output_dir kilt_passages_${dataset}

python dpr/index_simple_corpus.py \
--embed 2of2 \
--dpr_ctx_encoder_path models/DPR/${dataset}/ctx_encoder \
--corpus kilt_passages \
--output_dir kilt_passages_${dataset}
```

Index the passage vectors (dataset = trex, zsRE)
```bash
python dpr/faiss_index.py \
--corpus_dir kilt_passages_${dataset} \
--scalar_quantizer 8 \
--output_file kilt_passages_${dataset}/index.faiss
```

Train RAG
```bash
python dataloader/file_splitter.py \
--input trex-train-kilt.jsonl \
--outdirs trex_training \
--file_counts 64

python slot_filling/rag_client_server_train.py \
  --kilt_data trex_training \
  --output models/RAG/trex_dpr_rag \
  --corpus_endpoint kilt_passages_trex \
  --model_name facebook/rag-token-nq \
  --model_path models/RAG/trex_dpr_rag_init \
  --num_instances 500000 --warmup_instances 10000  --num_train_epochs 1 \
  --learning_rate 3e-5 --full_train_batch_size 128 --gradient_accumulation_steps 64


python slot_filling/rag_client_server_train.py \
  --kilt_data structured_zeroshot-train-kilt.jsonl \
  --output models/RAG/zsRE_dpr_rag \
  --corpus_endpoint kilt_passages_zsRE \
  --model_name facebook/rag-token-nq \
  --model_path models/RAG/zsRE_dpr_rag_init \
  --num_instances 147909  --warmup_instances 10000 --num_train_epochs 1 \
  --learning_rate 3e-5 --full_train_batch_size 128 --gradient_accumulation_steps 64

```

Apply RAG (dev_file = trex-dev-kilt.jsonl, structured_zeroshot-dev-kilt.jsonl)
```bash
python slot_filling/rag_client_server_apply.py \
  --kilt_data ${dev_file} \
  --corpus_endpoint kilt_passages_${dataset} \
  --output predictions/${dataset}_dev.jsonl \
  --model_name facebook/rag-token-nq \
  --model_path models/RAG/${dataset}_dpr_rag

python eval/convert_for_kilt_eval.py \
--apply_file predictions/${dataset}_dev.jsonl \
--eval_file predictions/${dataset}_dev_kilt_format.jsonl

```

Run official evaluation script
```bash
# install KILT evaluation scripts
git clone https://github.com/facebookresearch/KILT.git
cd KILT
conda create -n kilt37 -y python=3.7 && conda activate kilt37
pip install -r requirements.txt
export PYTHONPATH=`pwd`

# run evaluation
python kilt/eval_downstream.py predictions/${dataset}_dev_kilt_format.jsonl ${dev_file}
```

## Publications

### Re2G (NAACL 2022)

 ```bibtex
@inproceedings{glass-etal-2022-re2g,
    title = "{R}e2{G}: Retrieve, Rerank, Generate",
    author = "Glass, Michael  and
      Rossiello, Gaetano  and
      Chowdhury, Md Faisal Mahbub  and
      Naik, Ankita  and
      Cai, Pengshan  and
      Gliozzo, Alfio",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.194",
    pages = "2701--2715",
}
```

### KGI (EMNLP 2021)
 
 ```bibtex
@inproceedings{glass-etal-2021-robust,
    title = "Robust Retrieval Augmented Generation for Zero-shot Slot Filling",
    author = "Glass, Michael  and
      Rossiello, Gaetano  and
      Chowdhury, Md Faisal Mahbub  and
      Gliozzo, Alfio",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.148",
    doi = "10.18653/v1/2021.emnlp-main.148",
    pages = "1939--1949",
}
```
