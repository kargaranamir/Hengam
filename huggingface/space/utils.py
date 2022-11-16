"""
utils for Hengam inference
"""

"""### Import Libraries"""

# import primitive libraries
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

# import seqval to report classifier performance metrics
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.scheme import IOB2

# import torch related modules
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# import pytorch lightning library
import pytorch_lightning as pl
from torchcrf import CRF as SUPERCRF

# import NLTK to create better tokenizer
import nltk
from nltk.tokenize import RegexpTokenizer

# Transformers : Roberta Model
from transformers import XLMRobertaTokenizerFast
from transformers import XLMRobertaModel, XLMRobertaConfig


# import Typings
from typing import Union, Dict, List, Tuple, Any, Optional

import glob

# for sent tokenizer (nltk)
nltk.download('punkt')


"""## XLM-Roberta
### TokenFromSubtoken
- Code adapted from the following [file](https://github.com/deepmipt/DeepPavlov/blob/master/deeppavlov/models/torch_bert/torch_transformers_sequence_tagger.py)
- DeepPavlov is an popular open source library for deep learning end-to-end dialog systems and chatbots.
- Licensed under the Apache License, Version 2.0 (the "License");
"""

class TokenFromSubtoken(torch.nn.Module):

    def forward(self, units: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Assemble token level units from subtoken level units
        Args:
            units: torch.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
            mask: mask of token beginnings. For example: for tokens
                    [[``[CLS]`` ``My``, ``capybara``, ``[SEP]``],
                    [``[CLS]`` ``Your``, ``aar``, ``##dvark``, ``is``, ``awesome``, ``[SEP]``]]
                the mask will be
                    [[0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0]]
        Returns:
            word_level_units: Units assembled from ones in the mask. For the
                example above this units will correspond to the following
                    [[``My``, ``capybara``],
                    [``Your`, ``aar``, ``is``, ``awesome``,]]
                the shape of this tensor will be [batch_size, TOKEN_seq_length, n_features]
        """
        
        device = units.device
        nf_int = units.size()[-1]
        batch_size = units.size()[0]

        # number of TOKENS in each sentence
        token_seq_lengths = torch.sum(mask, 1).to(torch.int64)
        # number of words
        n_words = torch.sum(token_seq_lengths)
        # max token seq len
        max_token_seq_len = torch.max(token_seq_lengths)

        idxs = torch.stack(torch.nonzero(mask, as_tuple=True), dim=1)
        # padding is for computing change from one sample to another in the batch
        sample_ids_in_batch = torch.nn.functional.pad(input=idxs[:, 0], pad=[1, 0])
        
        a = (~torch.eq(sample_ids_in_batch[1:], sample_ids_in_batch[:-1])).to(torch.int64)
        
        # transforming sample start masks to the sample starts themselves
        q = a * torch.arange(n_words, device=device).to(torch.int64)
        count_to_substract = torch.nn.functional.pad(torch.masked_select(q, q.to(torch.bool)), [1, 0])

        new_word_indices = torch.arange(n_words, device=device).to(torch.int64) - count_to_substract[torch.cumsum(a, 0)]
        
        n_total_word_elements = max_token_seq_len*torch.ones_like(token_seq_lengths, device=device).sum()
        word_indices_flat = (idxs[:, 0] * max_token_seq_len + new_word_indices).to(torch.int64)
        #x_mask = torch.sum(torch.nn.functional.one_hot(word_indices_flat, n_total_word_elements), 0)
        #x_mask = x_mask.to(torch.bool)
        x_mask = torch.zeros(n_total_word_elements, dtype=torch.bool, device=device)
        x_mask[word_indices_flat] = torch.ones_like(word_indices_flat, device=device, dtype=torch.bool)
        # to get absolute indices we add max_token_seq_len:
        # idxs[:, 0] * max_token_seq_len -> [0, 0, 0, 1, 1, 2] * 2 = [0, 0, 0, 3, 3, 6]
        # word_indices_flat -> [0, 0, 0, 3, 3, 6] + [0, 1, 2, 0, 1, 0] = [0, 1, 2, 3, 4, 6]
        # total number of words in the batch (including paddings)
        # batch_size * max_token_seq_len -> 3 * 3 = 9
        # tf.one_hot(...) ->
        # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
        #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
        #  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]]
        #  x_mask -> [1, 1, 1, 1, 1, 0, 1, 0, 0]
        nonword_indices_flat = (~x_mask).nonzero().squeeze(-1)

        # get a sequence of units corresponding to the start subtokens of the words
        # size: [n_words, n_features]
        
        elements = units[mask.bool()]

        # prepare zeros for paddings
        # size: [batch_size * TOKEN_seq_length - n_words, n_features]
        paddings = torch.zeros_like(nonword_indices_flat, dtype=elements.dtype).unsqueeze(-1).repeat(1,nf_int).to(device)

        # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]
        tensor_flat_unordered = torch.cat([elements, paddings])
        _, order_idx = torch.sort(torch.cat([word_indices_flat, nonword_indices_flat]))
        tensor_flat = tensor_flat_unordered[order_idx]

        tensor = torch.reshape(tensor_flat, (-1, max_token_seq_len, nf_int))
        # tensor -> [[x, x, x],
        #            [x, x, 0],
        #            [x, 0, 0]]

        return tensor

"""### Conditional Random Field 
- Code adopted form [torchcrf library](https://pytorch-crf.readthedocs.io/en/stable/)
- we override veiterbi decoder in order to make it compatible with our code 
"""

class CRF(SUPERCRF):

    # override veiterbi decoder in order to make it compatible with our code 
    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        history = torch.stack(history, dim=0)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for i, hist in enumerate(torch.flip(history[:seq_ends[idx]], dims=(0,))):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag)

            best_tags = torch.stack(best_tags, dim=0)

            # Reverse the order because we start from the last timestep
            best_tags_list.append(torch.flip(best_tags, dims=(0,)))

        best_tags_list = nn.utils.rnn.pad_sequence(best_tags_list, batch_first=True, padding_value=0)

        return best_tags_list

"""### CRFLayer 
- Forward: decide output logits basaed on backbone network  
- Decode: decode based on CRF weights
"""

class CRFLayer(nn.Module):
    def __init__(self, embedding_size, n_labels):

        super(CRFLayer, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.output_dense = nn.Linear(embedding_size,n_labels)
        self.crf = CRF(n_labels, batch_first=True)
        self.token_from_subtoken = TokenFromSubtoken()

    # Forward: decide output logits basaed on backbone network  
    def forward(self, embedding, mask):
        logits = self.output_dense(self.dropout(embedding))
        logits = self.token_from_subtoken(logits, mask)
        pad_mask = self.token_from_subtoken(mask.unsqueeze(-1), mask).squeeze(-1).bool()
        return logits, pad_mask

    # Decode: decode based on CRF weights 
    def decode(self, logits, pad_mask):
        return self.crf.decode(logits, pad_mask)

    # Evaluation Loss: calculate mean log likelihood of CRF layer
    def eval_loss(self, logits, targets, pad_mask):
        mean_log_likelihood = self.crf(logits, targets, pad_mask, reduction='sum').mean()
        return -mean_log_likelihood

"""### NERModel
- Roberta Model with CRF Layer
"""

class NERModel(nn.Module):

    def __init__(self, n_labels:int, roberta_path:str):
        super(NERModel,self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(roberta_path)
        self.crf = CRFLayer(self.roberta.config.hidden_size, n_labels)

    # Forward: pass embedings to CRF layer in order to evaluate logits from suboword sequence
    def forward(self, 
                input_ids:torch.Tensor,
                attention_mask:torch.Tensor,
                token_type_ids:torch.Tensor,
                mask:torch.Tensor) -> torch.Tensor:

        embedding = self.roberta(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)[0]
        logits, pad_mask = self.crf(embedding, mask)
        return logits, pad_mask

    # Disable Gradient and Predict with model
    @torch.no_grad()
    def predict(self, inputs:Tuple[torch.Tensor]) -> torch.Tensor:
        input_ids, attention_mask, token_type_ids, mask = inputs
        logits, pad_mask = self(input_ids, attention_mask, token_type_ids, mask)
        decoded = self.crf.decode(logits, pad_mask)
        return decoded, pad_mask

    # Decode: pass to crf decoder and decode based on CRF weights 
    def decode(self, logits, pad_mask):
        """Decode logits using CRF weights 
        """
        return self.crf.decode(logits, pad_mask) 

    # Evaluation Loss: pass to crf eval_loss and calculate mean log likelihood of CRF layer
    def eval_loss(self, logits, targets, pad_mask):
        return self.crf.eval_loss(logits, targets, pad_mask)

    # Determine number of layers to be fine-tuned (!freeze) 
    def freeze_roberta(self, n_freeze:int=6):
        for param in self.roberta.parameters():
            param.requires_grad = False

        for param in self.roberta.encoder.layer[n_freeze:].parameters():
            param.requires_grad = True

"""### NERTokenizer
- NLTK tokenizer along with XLMRobertaTokenizerFast tokenizer
- Code adapted from the following [file](https://github.com/ugurcanozalp/multilingual-ner/blob/main/multiner/utils/custom_tokenizer.py)
"""

class NERTokenizer(object):

    MAX_LEN=512
    BATCH_LENGTH_LIMT = 380 # Max number of roberta tokens in one sentence.

    # Modified version of http://stackoverflow.com/questions/36353125/nltk-regular-expression-tokenizer
    PATTERN = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A. or U.S.A # 
        | (?:\d+\.)           # numbers
        | \w+(?:[-.]\w+)*     # words with optional internal hyphens
        | \$?\d+(?:.\d+)?%?   # currency and percentages, e.g. $12.40, 82%
        | \.\.\.              # ellipsis, and special chars below, includes ], [
        | [-\]\[.؟،؛;"'?,():_`“”/°º‘’″…#$%()*+<>=@\\^_{}|~❑&§\!]
        | \u200c
    '''

    def __init__(self, base_model:str, to_device:str='cpu'):
        super(NERTokenizer,self).__init__()
        self.roberta_tokenizer = XLMRobertaTokenizerFast.from_pretrained(base_model, do_lower_case=False, padding=True, truncation=True)
        self.to_device = to_device

        self.word_tokenizer = RegexpTokenizer(self.PATTERN)
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # tokenize batch of tokens
    def tokenize_batch(self, inputs, pad_to = None) -> torch.Tensor:
        batch = [inputs] if isinstance(inputs[0], str) else inputs
        
        input_ids, attention_mask, token_type_ids, mask = [], [], [], []
        for tokens in batch:
            input_ids_tmp, attention_mask_tmp, token_type_ids_tmp, mask_tmp = self._tokenize_words(tokens)
            input_ids.append(input_ids_tmp)
            attention_mask.append(attention_mask_tmp)
            token_type_ids.append(token_type_ids_tmp)
            mask.append(mask_tmp)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.roberta_tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        # truncate MAX_LEN
        if input_ids.shape[-1]>self.MAX_LEN:
            input_ids = input_ids[:,:,:self.MAX_LEN]
            attention_mask = attention_mask[:,:,:self.MAX_LEN]
            token_type_ids = token_type_ids[:,:,:self.MAX_LEN]
            mask = mask[:,:,:self.MAX_LEN]
        
        # extend pad 
        elif pad_to is not None and pad_to>input_ids.shape[1]:
            bs = input_ids.shape[0]
            padlen = pad_to-input_ids.shape[1]

            input_ids_append = torch.tensor([self.roberta_tokenizer.pad_token_id], dtype=torch.long).repeat([bs, padlen]).to(self.to_device)
            input_ids = torch.cat([input_ids, input_ids_append], dim=-1)

            attention_mask_append = torch.tensor([0], dtype=torch.long).repeat([bs, padlen]).to(self.to_device)
            attention_mask = torch.cat([attention_mask, attention_mask_append], dim=-1)

            token_type_ids_append = torch.tensor([0], dtype=torch.long).repeat([bs, padlen]).to(self.to_device)
            token_type_ids = torch.cat([token_type_ids, token_type_ids_append], dim=-1)

            mask_append = torch.tensor([0], dtype=torch.long).repeat([bs, padlen]).to(self.to_device)
            mask = torch.cat([mask, mask_append], dim=-1)

        # truncate pad
        elif pad_to is not None and pad_to<input_ids.shape[1]:
            input_ids = input_ids[:,:,:pad_to]
            attention_mask = attention_mask[:,:,:pad_to]
            token_type_ids = token_type_ids[:,:,:pad_to]
            mask = mask[:,:,:pad_to]

        if isinstance(inputs[0], str):
            return input_ids[0], attention_mask[0], token_type_ids[0], mask[0]
        else:
            return input_ids, attention_mask, token_type_ids, mask

    # tokenize list of words with roberta tokenizer
    def _tokenize_words(self, words):
        roberta_tokens = []
        mask = []
        for word in words:
            subtokens = self.roberta_tokenizer.tokenize(word)
            roberta_tokens+=subtokens
            n_subtoken = len(subtokens)
            if n_subtoken>=1:
                mask = mask + [1] + [0]*(n_subtoken-1)

        # add special tokens [CLS] and [SeP]
        roberta_tokens = [self.roberta_tokenizer.cls_token] + roberta_tokens + [self.roberta_tokenizer.sep_token]
        mask = [0] + mask + [0]
        input_ids = torch.tensor(self.roberta_tokenizer.convert_tokens_to_ids(roberta_tokens), dtype=torch.long).to(self.to_device)
        attention_mask = torch.ones(len(mask), dtype=torch.long).to(self.to_device)
        token_type_ids = torch.zeros(len(mask), dtype=torch.long).to(self.to_device)
        mask = torch.tensor(mask, dtype=torch.long).to(self.to_device)
        return input_ids, attention_mask, token_type_ids, mask

    # sent_to_token: yield each sentence token with positional span using nltk
    def sent_to_token(self, raw_text):
        for offset, ending in self.sent_tokenizer.span_tokenize(raw_text):
            sub_text = raw_text[offset:ending]
            words, spans = [], []
            flush = False
            total_subtoken = 0
            for start, end in self.word_tokenizer.span_tokenize(sub_text):
                flush = True
                start += offset
                end += offset
                words.append(raw_text[start:end])
                spans.append((start,end))
                total_subtoken += len(self.roberta_tokenizer.tokenize(words[-1]))
                if (total_subtoken > self.BATCH_LENGTH_LIMT): 
                    # Print
                    yield words[:-1],spans[:-1]
                    spans = spans[len(spans)-1:]
                    words = words[len(words)-1:]
                    total_subtoken = sum([len(self.roberta_tokenizer.tokenize(word)) for word in words])
                    flush = False

            if flush and len(spans) > 0:
                yield words,spans

    # Extract (batch words span() from a raw sentence
    def prepare_row_text(self, raw_text, batch_size=16):
        words_list, spans_list = [], []
        end_batch = False
        for words, spans in self.sent_to_token(raw_text):
            end_batch = True
            words_list.append(words)
            spans_list.append(spans)
            if len(spans_list) >= batch_size:
                input_ids, attention_mask, token_type_ids, mask = self.tokenize_batch(words_list)
                yield (input_ids, attention_mask, token_type_ids, mask), words_list, spans_list
                words_list, spans_list = [], []
        if end_batch and len(words_list) > 0:
            input_ids, attention_mask, token_type_ids, mask = self.tokenize_batch(words_list)
            yield (input_ids, attention_mask, token_type_ids, mask), words_list, spans_list

"""### NER
NER Interface : We Use this interface to infer sentence Time-Date tags.
"""

class NER(object):

    def __init__(self, model_path, model_name, tags):
        
        self.tags = tags
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load Pre-Trained model
        roberta_path = "xlm-roberta-base"
        self.model = NERModel(n_labels=len(self.tags), roberta_path=roberta_path).to(self.device)
        # Load Fine-Tuned model
        state_dict = torch.load(os.path.join(model_path, model_name))
        self.model.load_state_dict(state_dict, strict=False)
        # Enable Evaluation mode
        self.model.eval()
        self.tokenizer = NERTokenizer(base_model=roberta_path, to_device=self.device)

    # Predict and Pre/Post-Process the input/output
    @torch.no_grad()
    def __call__(self, raw_text):

        outputs_flat, spans_flat, entities = [], [], []
        for batch, words, spans in self.tokenizer.prepare_row_text(raw_text):
            output, pad_mask = self.model.predict(batch)
            outputs_flat.extend(output[pad_mask.bool()].reshape(-1).tolist())
            spans_flat += sum(spans, [])

        for tag_idx,(start,end) in zip(outputs_flat,spans_flat):
            tag = self.tags[tag_idx]
            # filter out O tags
            if tag != 'O':
                entities.append({'Text': raw_text[start:end], 
                                 'Tag': tag, 
                                 'Start':start,
                                 'End': end})

        return entities