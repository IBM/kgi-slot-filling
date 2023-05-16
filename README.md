# Re2G (Retrieve, Rerank, Generate)
This is the code for our KILT leaderboard submission to the T-REx, Wizard of Wikipedia, FEVER, NQ, and TriviaQA tasks.  
It includes code for training a DPR model, a reranker, then continuing training with a RAG-like model incorporating both retrieval and reranking..

References:
- NAACL 2022: [Re2G: Retrieve, Rerank, Generate](https://aclanthology.org/2022.naacl-main.194/)
- EMNLP 2021: [Robust Retrieval Augmented Generation for Zero-shot Slot Filling](https://aclanthology.org/2021.emnlp-main.148/)


# Process to reproduce


## Download the [KILT data and knowledge source](https://github.com/facebookresearch/KILT)
* [KILT Knowledge Source](http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json)

This produces the files \\${dataset}-train-kilt.jsonl, \\${dataset}-dev-kilt.jsonl, and kilt_knowledgesource.json



## Corpus pre-processing
Segment the KILT Knowledge Source into passages:
```bash
python corpus/kilt_passage_corpus.py \
--kilt_corpus kilt_knowledgesource.json --output_dir kilt_passages --passage_ids passage_ids.txt
```
This produces the directory kilt_passages and the file passage_ids.txt

Generate the first phase of the DPR training data. The directory \\${dataset} should contain the KILT training and dev files (-train-kilt.jsonl, -dev-kilt.jsonl)
```bash
python dpr/kilt2positive_pids.py \
--kilt_data_dir ${dataset}  \
--passage_ids passage_ids.txt \
--kilt_passages kilt_passages
```
This produces the files \\${dataset}/train_positive_pids.jsonl and \\${dataset}/dev_positive_pids.jsonl

## BM25 index
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
This produces the programs Anserini/target/appassembler/bin/IndexCollection and Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar

Index the passages for BM25 search
```bash
python dpr/anserini_prep.py \
--input kilt_passages \
--output anserini_passages

sh Anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
-generator LuceneDocumentGenerator -threads 40 -input anserini_passages \
-index anserini_passage_index -storePositions -storeDocvectors -storeRawDocs
```
This produces the directory anserini_passage_index

Generate BM25 hard negatives
```bash
python dpr/dpr_bm25_negatives.py \
  --positive_pids ${dataset}/train_positive_pids.jsonl  \
  --jar Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --anserini_index anserini_passage_index \
  --output_dir ${dataset}_dpr_training_data
```
This produces the directory \\${dataset}_dpr_training_data

#### Optional: Establish BM25 baseline
```bash
python  dpr/bm25_apply.py \
  --kilt_data ${dataset}-dev-kilt.jsonl \
  --output predictions/bm25/dev.jsonl \
  --include_passages --n_docs 20 \
  --jar Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --anserini_index anserini_passage_index
```
This produces predictions/bm25/dev.jsonl


#### Train DPR
```bash
# multi-gpu is not well supported
export CUDA_VISIBLE_DEVICES=0

python dpr/biencoder_trainer.py \
--train_dir ${dataset}_dpr_training_data \
--output_dir models/DPR/${dataset} \
--num_train_epochs 2 --sample_negative_from_top_k 5 \
--encoder_gpu_train_limit 32 \
--full_train_batch_size 128 \
--max_grad_norm 1.0 --learning_rate 5e-5
```
This produces the model models/DPR/\\${dataset} (containing models/DPR/\\${dataset}/qry_encoder and models/DPR/\\${dataset}/ctx_encoder)

#### Optional: check DPR is training well by using it to rerank BM25 results
```bash
python dpr/dpr_quick_apply.py \
  --dpr_path models/DPR/${dataset} \
  --output predictions/dpr/dev_quick_bm25_dpr.jsonl \
  --kilt_data ${dataset}-dev-kilt.jsonl  \
  --initial_retrieval predictions/bm25/dev.jsonl
```

#### Put the trained DPR query encoder into the pre-trained RAG model
```bash
python dpr/prepare_rag_model.py \
--save_dir models/RAG/${dataset}_dpr_rag_init  \
--qry_encoder_path models/DPR/${dataset}/qry_encoder
```
This produces the model models/RAG/\\${dataset}_dpr_rag_init

#### Create the DPR indexed corpus
Encode the passages
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

Index the passage vectors
```bash
python dpr/faiss_index.py \
--corpus kilt_passages_${dataset}/passages_1_of_2.json.gz.records \
--scalar_quantizer 8

python dpr/faiss_index.py \
--corpus kilt_passages_${dataset}/passages_2_of_2.json.gz.records \
--scalar_quantizer 8
```
This produces the directory kilt_passages_\\${dataset}

#### Optional: Apply the trained DPR
```bash
python dpr/dpr_apply.py \
--kilt_data ${dev_file} \
--qry_encoder_path models/DPR/${dataset}/qry_encoder \
--corpus_endpoint kilt_passages_${dataset} \
--output predictions/${dataset}_dev_dpr.jsonl
```



## Optional: Further train DPR with negatives from index
Once you have built the kilt_passages_trex (or other dataset) including the faiss index, you can use
the initial index to gather negatives for further DPR training:
```bash
python dpr/negatives_from_index.py \
--corpus_endpoint kilt_passages_${dataset} \
--qry_encoder_path models/DPR/${dataset}/qry_encoder \
--dpr_training_data ${dataset}_dpr_training_data.jsonl \
--kilt_training_data ${dataset}-train-kilt.jsonl \
--output_file ${dataset}_dpr_training_data_nfi.jsonl
```

Then continue training your DPR model. The argument for resume_from should be the model you trained earlier.
```bash
python dpr/biencoder_trainer.py \
--resume_from models/DPR/${dataset}
--train_dir ${dataset}_dpr_training_data_nfi.jsonl \
--output_dir models/DPR/${dataset}_nfi \
--num_train_epochs 2 \
--encoder_gpu_train_limit 32 \
--full_train_batch_size 128 \
--max_grad_norm 1.0 --learning_rate 5e-5
```

Use the new DPR model to index_simple_corpus and faiss_index as above. 
You should also use this model to initialize your RAG model with prepare_rag_model.

## RAG
Train RAG
```bash
python dataloader/file_splitter.py \
--input ${dataset}-train-kilt.jsonl \
--outdirs ${dataset}_training \
--file_counts 8

python generation/kgi_train.py \
  --kilt_data ${dataset}_training \
  --output models/RAG/${dataset}_dpr_rag \
  --corpus_endpoint kilt_passages_${dataset} \
  --model_name facebook/rag-token-nq \
  --model_path models/RAG/${dataset}_dpr_rag_init \
  --warmup_fraction 0.05  --num_train_epochs 2 \
  --learning_rate 3e-5 --full_train_batch_size 128 --gradient_accumulation_steps 64
```
This produces the model models/RAG/\\${dataset}_dpr_rag

Apply RAG
```bash
python generation/kgi_apply.py \
  --kilt_data ${dataset}-dev-kilt.jsonl \
  --corpus_endpoint kilt_passages_${dataset} \
  --output_dir predictions/KGI/${dataset}_dev \
  --model_name facebook/rag-token-nq \
  --model_path models/RAG/${dataset}_dpr_rag
```

## Reranker

### Create initial retrieval files
RAG updated DPR results on train and dev
```bash
python dpr/dpr_apply.py \
  --kilt_data ${dataset}-dev-kilt.jsonl  \
  --output predictions/dprKGI0/dev.jsonl  --include_passages \
  --corpus_endpoint kilt_passages_${dataset} --n_docs_for_provenance 20 \
  --rag_model_path models/RAG/${dataset}_dpr_rag

python dpr/dpr_apply.py \
  --kilt_data ${dataset}-train-kilt.jsonl  \
  --output predictions/dprKGI0/train.jsonl  --include_passages \
  --corpus_endpoint kilt_passages_${dataset} --n_docs_for_provenance 20 \
  --rag_model_path models/RAG/${dataset}_dpr_rag
```

BM25 results on train and dev
```bash
python  dpr/bm25_apply.py \
  --kilt_data ${dataset}-dev-kilt.jsonl \
  --output predictions/bm25/dev.jsonl \
  --include_passages --n_docs 20 \
  --jar Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --anserini_index anserini_passage_index

python dpr/bm25_apply.py \
  --kilt_data ${dataset}-train-kilt.jsonl \
  --output predictions/bm25/train.jsonl \
  --include_passages --n_docs 20 \
  --jar Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --anserini_index anserini_passage_index
```

Merge DPR and BM25 results
```bash
python  dpr/merge_initial_retrieval_files.py \
 --kilt_data ${dataset}-dev-kilt.jsonl \
 --initial_retrievals predictions/bm25/dev.jsonl,predictions/dprKGI0/dev.jsonl \
 --output predictions/dpr_bm25/dev.jsonl 

python  dpr/merge_initial_retrieval_files.py \
 --kilt_data ${dataset}-train-kilt.jsonl \
 --initial_retrievals predictions/bm25/train.jsonl,predictions/dprKGI0/train.jsonl \
 --output predictions/dpr_bm25/train.jsonl 
```

This produces the files predictions/dpr_bm25/train.jsonl and predictions/dpr_bm25/dev.jsonl

### Train reranker stage 1 (isolated training)
```bash
python reranker/reranker_train.py \
  --model_type bert --model_name_or_path nboost/pt-bert-base-uncased-msmarco --do_lower_case \
  --positive_pids ${dataset}/train_positive_pids.jsonl \
  --initial_retrieval  predictions/dpr_bm25/train.jsonl  \
  --num_train_epochs 2 \
  --output_dir models/reranker_stage1
```

This produces the model models/reranker_stage1

Apply to check performance
```bash
python reranker/reranker_apply.py \
  --model_type bert --model_name_or_path  models/reranker_stage1 --do_lower_case \
  --kilt_data /data/KILT/qa/nq2/nq-dev-kilt.jsonl  \
  --initial_retrieval predictions/dpr_bm25/dev.jsonl   \
  --output predictions/rerank_dpr_bm25/dev.jsonl 
```

## Re2G

Train Re2G
```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--master_addr="127.0.1.1" --master_port=1234 --use_env --node_rank=0 \
generation/re2g_train.py \
  --reranker.model_name_or_path models/reranker_stage1  \
  --dpr.rag_model_path models/RAG/${dataset}_dpr_rag \
  --dpr.corpus_endpoint ${CORPUS} \
  --bm25.jar Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --bm25.anserini_index anserini_passage_index \
  --dpr.n_docs 12 --bm25.n_docs 12 \
  --kilt_data ${dataset}-train-kilt.jsonl   \
  --output_dir models/re2g \
  --model_name facebook/rag-token-nq \
  --model_path models/RAG/${dataset}_dpr_rag \
  --warmup_fraction 0.1 --num_train_epochs 1 \
  --positive_pids ${dataset}/train_positive_pids.jsonl \
  --learning_rate 3e-5 --full_train_batch_size 128
```

This produces the model models/re2g (also containing models/re2g/qry_encoder, models/re2g/reranker)

Apply Re2G
```bash
python generation/re2g_apply.py \
  --reranker.model_name_or_path models/re2g/reranker \
  --dpr.qry_encoder_path models/re2g/qry_encoder \
  --dpr.corpus_endpoint ${CORPUS} \
  --bm25.jar Anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar \
  --bm25.anserini_index anserini_passage_index \
  --dpr.n_docs 12 --bm25.n_docs 12 \
  --kilt_data ${dataset}-dev-kilt.jsonl  \
  --output predictions/re2g/dev_apply.jsonl \
  --model_name facebook/rag-token-nq \
  --model_path models/re2g \
  --positive_pids ${dataset}/dev_positive_pids.jsonl
```

When applying to test you will not have the positive_pids.jsonl, so just leave out that argument. 
It is only used to run the evaluation after the apply.

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

 ## License
  
 This work is released under Apache 2.0 license.
