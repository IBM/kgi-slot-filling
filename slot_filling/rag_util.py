from typing import List, Optional
from transformers import RagTokenizer, RagConfig
import re
import string


# from kilt.kilt_utils.py in https://github.com/facebookresearch/KILT
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# from transformers.models.rag.retrieval_rag.py
# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def postprocess_docs(config: RagConfig, tokenizer: RagTokenizer, docs, input_strings, prefix, n_docs, return_tensors=None):
    r"""
    Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

    Args:
        docs  (:obj:`dict`):
            Retrieved documents.
        input_strings (:obj:`str`):
            Input strings decoded by ``preprocess_query``.
        prefix (:obj:`str`):
            Prefix added at the beginning of each input, typically used with T5-based models.

    Return:
        :obj:`tuple(tensors)`: a tuple consisting of two elements: contextualized ``input_ids`` and a compatible
        ``attention_mask``.
    """

    def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
        # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
        # TODO(piktus): better handling of truncation
        if doc_title.startswith('"'):
            doc_title = doc_title[1:]
        if doc_title.endswith('"'):
            doc_title = doc_title[:-1]
        if prefix is None:
            prefix = ""
        out = (prefix + doc_title + config.title_sep + doc_text + config.doc_sep + input_string).replace(
            "  ", " "
        )
        return out

    rag_input_strings = [
        cat_input_and_doc(
            docs[i]["title"][j],
            docs[i]["text"][j],
            input_strings[i],
            prefix,
        )
        for i in range(len(docs))
        for j in range(n_docs)
    ]

    contextualized_inputs = tokenizer.generator.batch_encode_plus(
        rag_input_strings,
        max_length=config.max_combined_length,
        return_tensors=return_tensors,
        padding="max_length",
        truncation=True,
    )

    return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]


# from a version of transformers.models.rag.tokenization_rag.py
# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def prepare_seq2seq_batch(
        tokenizer,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        truncation=True,
        **kwargs,
):
    if max_length is None:
        max_length = tokenizer.question_encoder.model_max_length
    model_inputs = tokenizer.question_encoder(
        src_texts,
        add_special_tokens=True,
        return_tensors=return_tensors,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        **kwargs,
    )
    if tgt_texts is None:
        return model_inputs
    # Process tgt_texts
    if max_target_length is None:
        max_target_length = tokenizer.generator.model_max_length
    labels = tokenizer.generator(
        tgt_texts,
        add_special_tokens=True,
        return_tensors=return_tensors,
        padding=padding,
        max_length=max_target_length,
        truncation=truncation,
        **kwargs,
    )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs
