# KGI (Knowledge Graph Induction) for slot filling
This is the code for our KILT leaderboard submission to the T-REx and zsRE tasks.  It includes code for training a DPR model then continuing training with RAG.


Our model is described in: [Zero-shot Slot Filling with DPR and RAG](https://arxiv.org/abs/2104.08610)  

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

download and build [Anserini](https://github.com/castorini/anserini)

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

python rag/dpr/index_simple_corpus.py \
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
