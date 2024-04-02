
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import LongformerTokenizer, LongformerModel
import math
import random
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from scipy.optimize import linear_sum_assignment




'''hyper-parameters'''

MAX_LEN = 2048
batch_size = 1
num_epochs = 5
check_times = 10 * num_epochs

#coreference_train_method = "event_cluster"
coreference_train_method = "event_pairs"

event_weight_positive = 1
coreference_weight_positive = 1
temporal_weight_before = 1
temporal_weight_after = 1
temporal_weight_overlap = 1
causal_weight_cause = 1
causal_weight_caused = 1
subevent_weight_contain = 1
subevent_weight_contained = 1

no_decay = ['bias', 'LayerNorm.weight']
longformer_weight_decay = 1e-2
non_longformer_weight_decay = 1e-2
warmup_proportion = 0.1
non_longformer_lr = 1e-4
longformer_lr = 1e-5




'''creat the files paths for train dev test sets'''

def create_file_path(parent_path, file_names_list):
    file_paths_list = []
    for file_i in range(len(file_names_list)):
        file_name = file_names_list[file_i]
        file_path = parent_path + file_name
        file_paths_list.append(file_path)
    return file_paths_list





'''create custom dataset class, input is train/dev/test files paths list, output is getting info in one article json'''

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


class custom_dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, "r") as in_json:
            article_json = json.load(in_json)

        input_ids = []
        attention_mask = []

        label_event = []
        # label_event[i,:] = [start, end, label_event], where [start, end] is the corresponding index in input_ids of article_json['event_label'][i]
        # input_ids(range(start, end)) can extract the corresponding word
        # can down sampling 0 class by slicing label_event[:,2] == 0

        input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens = False)['input_ids']) # the <s> start token
        attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens = False)['attention_mask'])

        start = len(input_ids)
        word_encoding = tokenizer.encode_plus(article_json['event_label'][0]['token'], add_special_tokens = False)
        input_ids.extend(word_encoding['input_ids'])
        attention_mask.extend(word_encoding['attention_mask'])
        end = len(input_ids)
        label_event.append([start, end, article_json['event_label'][0]['event_label']])

        for word_i in range(1, len(article_json['event_label'])):
            start = len(input_ids)
            word_encoding = tokenizer.encode_plus(' ' + article_json['event_label'][word_i]['token'], add_special_tokens=False)
            input_ids.extend(word_encoding['input_ids'])
            attention_mask.extend(word_encoding['attention_mask'])
            end = len(input_ids)
            label_event.append([start, end, article_json['event_label'][word_i]['event_label']])

        input_ids.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['input_ids'])  # the </s> end token
        attention_mask.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['attention_mask'])
        num_pad = MAX_LEN - len(input_ids)
        if num_pad > 0:
            input_ids.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['input_ids']) # the <pad> padding token
            attention_mask.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['attention_mask'])


        if len(input_ids) > MAX_LEN: # the length of input_ids can be larger than 2048
            if tokenizer.encode_plus(" ".join(article_json['tokens_list']), add_special_tokens=True)['input_ids'] != input_ids:
                print("word tokenizer unmatched with article tokenizer in " + file_path)
        else:
            if tokenizer.encode_plus(" ".join(article_json['tokens_list']), add_special_tokens = True,
                                     max_length=MAX_LEN, padding='max_length', truncation=True)['input_ids'] != input_ids:
                print("word tokenizer unmatched with article tokenizer in " + file_path)

        if len(label_event) != len(article_json['event_label']):
            print("number of words unmatched")


        input_ids = torch.tensor(input_ids) # size = length of tokens
        attention_mask = torch.tensor(attention_mask)

        label_event = torch.tensor(label_event) # number of words * 3 (start in input_ids, end in input_ids, label_event)


        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_coreference = []
        label_temporal = []
        label_causal = []
        label_subevent = []

        for event_pair_i in range(len(article_json['relation_label'])):

            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_in_event_label'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_in_event_label']])

            label_coreference.append(article_json['relation_label'][event_pair_i]['label_coreference'])
            label_temporal.append(article_json['relation_label'][event_pair_i]['label_temporal'])
            label_causal.append(article_json['relation_label'][event_pair_i]['label_causal'])
            label_subevent.append(article_json['relation_label'][event_pair_i]['label_subevent'])


        event_pairs = torch.tensor(event_pairs) # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_coreference = torch.tensor(label_coreference) # size = number of event pairs, corresponded to event_pairs
        label_temporal = torch.tensor(label_temporal)
        label_causal = torch.tensor(label_causal)
        label_subevent = torch.tensor(label_subevent)


        dict = {"input_ids": input_ids, "attention_mask": attention_mask,
                "label_event": label_event, "event_pairs": event_pairs, "label_coreference": label_coreference,
                "label_temporal": label_temporal, "label_causal": label_causal, "label_subevent": label_subevent}

        return dict






'''model'''


def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def pad_and_stack(tensors, pad_size=None, value=0.0):
    sizes = [s.shape[0] for s in tensors]
    if not pad_size:
        pad_size = max(sizes)

    padded = []
    for tensor, size in zip(tensors, sizes):
        padded.append(torch.cat((tensor, to_cuda(torch.tensor(value).repeat(pad_size - size)))))
    padded = torch.stack(padded, dim = 0)

    return padded

def flatten(lists):
    return [item for l in lists for item in l]

def fill_expand(labels):
    event_num = max(flatten(labels)) + 1
    filled_labels = torch.zeros((event_num, event_num)) # account for dummy
    for gr in labels:
        if len(gr) > 1:
            sorted_gr = sorted(gr)
            for i in range(len(sorted_gr)):
                for j in range(i+1, len(sorted_gr)):
                    filled_labels[sorted_gr[j]][sorted_gr[i]] = 1
        else:
            try:
                filled_labels[gr[0]][gr[0]] = 1 # dummy default to same index as itself
            except:
                print(gr)
                raise ValueError
    return filled_labels



class Token_Embedding(nn.Module):

    # input: input_ids, attention_mask, 1 article * number of tokens
    # output: number of tokens * 768, dealing with one article at one time

    def __init__(self):
        super(Token_Embedding, self).__init__()

        self.longformermodel = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True, )

    def forward(self, input_ids, attention_mask):

        outputs = self.longformermodel(input_ids = input_ids, attention_mask = attention_mask)
        hidden_states = outputs[2]
        token_embeddings_layers = torch.stack(hidden_states, dim=0)  # 13 layer * batch_size (1) * number of tokens * 768
        token_embeddings_layers = token_embeddings_layers[:, 0, :, :] # 13 layer * number of tokens * 768
        token_embeddings = torch.sum(token_embeddings_layers[-4:, :, :], dim = 0) # sum up the last four layers, number of tokens * 768

        return token_embeddings



class Event_Relation_Graph(nn.Module):

    # input: label_event, number of tokens for event identification * 3 (start in input_ids, end in input_ids, label_event)
    #        event_pairs, number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
    #        label_relation, size = number of event pairs

    def __init__(self):
        super(Event_Relation_Graph, self).__init__()

        self.token_embedding = Token_Embedding()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)


        self.event_head_1 = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.event_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_1.bias)

        self.event_head_2 = nn.Linear(768, 2, bias=True)
        nn.init.xavier_uniform_(self.event_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_2.bias)

        if coreference_train_method == "event_pairs":

            self.coreference_head_1 = nn.Linear(768 * 4, 768, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_1.bias)

            self.coreference_head_2 = nn.Linear(768, 256, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_2.bias)

            self.coreference_head_3 = nn.Linear(256, 2, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_3.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_3.bias)

        else: # coreference_train_method == "event_cluster"

            self.coreference_head_1 = nn.Linear(768 * 4, 768, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_1.bias)

            self.coreference_head_2 = nn.Linear(768, 256, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_2.bias)

            self.coreference_head_3 = nn.Linear(256, 1, bias=True)
            nn.init.xavier_uniform_(self.coreference_head_3.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.coreference_head_3.bias)


        self.temporal_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_1.bias)

        self.temporal_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_2.bias)

        self.temporal_head_3 = nn.Linear(256, 4, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_3.bias)


        self.causal_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.causal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_1.bias)

        self.causal_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.causal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_2.bias)

        self.causal_head_3 = nn.Linear(256, 3, bias=True)
        nn.init.xavier_uniform_(self.causal_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_3.bias)


        self.subevent_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_1.bias)

        self.subevent_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_2.bias)

        self.subevent_head_3 = nn.Linear(256, 3, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_3.bias)


        self.relu = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none') # no reduction



    def forward(self, input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method):

        token_embeddings = self.token_embedding(input_ids, attention_mask) # number of tokens * 768

        # token-level bi-lstm layer
        token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])

        h0 = torch.zeros(2, 1, 384).cuda().requires_grad_()
        c0 = torch.zeros(2, 1, 384).cuda().requires_grad_()

        token_embeddings, (_, _) = self.bilstm(token_embeddings, (h0, c0)) # batch_size 1 * number of tokens * 768
        token_embeddings = token_embeddings[0, :, :] # number of tokens * 768


        # event identification task
        # event_embeddings: embeddings used for event identification, nrow = nrow(label_event), ncol = 768

        for token_i in range(label_event.shape[0]):
            if token_i == 0:
                start_in_input_ids = label_event[token_i, 0]
                end_in_input_ids = label_event[token_i, 1]
                event_embeddings = torch.mean(token_embeddings[start_in_input_ids: end_in_input_ids, :], dim = 0).view(1, 768)
            else:
                start_in_input_ids = label_event[token_i, 0]
                end_in_input_ids = label_event[token_i, 1]
                event_embeddings = torch.cat((event_embeddings, torch.mean(token_embeddings[start_in_input_ids: end_in_input_ids, :], dim = 0).view(1, 768)), dim = 0)

        event_raw_scores = self.event_head_2(self.relu(self.event_head_1(event_embeddings))) # nrow = nrow(label_events), ncol = 2
        event_loss = self.crossentropyloss(event_raw_scores, label_event[:,2]) # size = nrow(label_events)

        event_loss_weight_0 = (label_event[:, 2] == 0).int() # weight = 1 for negative examples with label_event = 0
        event_loss_0 = torch.mul(event_loss, event_loss_weight_0)

        event_loss_weight_1 = torch.mul(label_event[:,2], event_weight_positive) # weight = event_weight_positive for positive examples with label_event = 1
        event_loss_1 = torch.mul(event_loss, event_loss_weight_1)

        event_weighted_loss = torch.add(event_loss_0, event_loss_1)
        event_weighted_loss = torch.sum(event_weighted_loss)


        # convert label_coreference into label_coreference_cluster

        number_of_events = torch.sum((label_event[:, 2] == 1).int())

        label_coreference_cluster = []  # use label_coreference_cluster and label_coreference_cluster can calculate CoNLL metric
        for event_idx in range(number_of_events):

            event1_index_in_cluster = -1  # event_idx
            for cluster_i in range(len(label_coreference_cluster)):
                if event_idx in label_coreference_cluster[cluster_i]:
                    event1_index_in_cluster = cluster_i

            prefix_sum = int((2 * number_of_events - 1 - event_idx) * event_idx / 2)
            for j in range(0, number_of_events - 1 - event_idx):
                if label_coreference[prefix_sum + j] == 1:

                    event2_index_in_cluster = -1  # (event_idx + j + 1)
                    for cluster_i in range(len(label_coreference_cluster)):
                        if (event_idx + j + 1) in label_coreference_cluster[cluster_i]:
                            event2_index_in_cluster = cluster_i

                    if event1_index_in_cluster == -1 and event2_index_in_cluster == -1:
                        label_coreference_cluster.append([event_idx, event_idx + j + 1])
                        event1_index_in_cluster = len(label_coreference_cluster) - 1
                        event2_index_in_cluster = len(label_coreference_cluster) - 1
                    if event1_index_in_cluster != -1 and event2_index_in_cluster == -1:
                        label_coreference_cluster[event1_index_in_cluster].append(event_idx + j + 1)
                        event2_index_in_cluster = event1_index_in_cluster
                    if event1_index_in_cluster == -1 and event2_index_in_cluster != -1:
                        label_coreference_cluster[event2_index_in_cluster].append(event_idx)
                        event1_index_in_cluster = event2_index_in_cluster
                    if event1_index_in_cluster != -1 and event2_index_in_cluster != -1 and event1_index_in_cluster != event2_index_in_cluster:
                        label_coreference_cluster[event1_index_in_cluster].extend(label_coreference_cluster[event2_index_in_cluster])
                        label_coreference_cluster.pop(event2_index_in_cluster)
                        # recalculate event1_index_in_cluster for event_idx because one element pop out
                        for cluster_i in range(len(label_coreference_cluster)):
                            if event_idx in label_coreference_cluster[cluster_i]:
                                event1_index_in_cluster = cluster_i
                        event2_index_in_cluster = event1_index_in_cluster

            if event1_index_in_cluster == -1:  # singleton
                label_coreference_cluster.append([event_idx])

        for cluster_i in range(len(label_coreference_cluster)):
            label_coreference_cluster[cluster_i] = sorted(label_coreference_cluster[cluster_i])
        label_coreference_cluster = sorted(label_coreference_cluster, key = lambda x: x[0])



        # event_pair_embeddings: embeddings used for event pair relation task, nrow = nrow(event_pairs), ncol = 768 * 4

        for event_pair_i in range(event_pairs.shape[0]):
            if event_pair_i == 0:
                event_1_in_label_event = event_pairs[event_pair_i, 0]
                event_2_in_label_event = event_pairs[event_pair_i, 1]
                event_1_embedding = event_embeddings[event_1_in_label_event, :].view(1, 768)
                event_2_embedding = event_embeddings[event_2_in_label_event, :].view(1, 768)
                pair_element_wise_sub = torch.sub(event_1_embedding, event_2_embedding)
                pair_element_wise_mul = torch.mul(event_1_embedding, event_2_embedding)
                event_pair_embeddings = torch.cat((event_1_embedding, event_2_embedding, pair_element_wise_sub, pair_element_wise_mul), dim = 1) # 1 * (768 * 4)
            else:
                event_1_in_label_event = event_pairs[event_pair_i, 0]
                event_2_in_label_event = event_pairs[event_pair_i, 1]
                event_1_embedding = event_embeddings[event_1_in_label_event, :].view(1, 768)
                event_2_embedding = event_embeddings[event_2_in_label_event, :].view(1, 768)
                pair_element_wise_sub = torch.sub(event_1_embedding, event_2_embedding)
                pair_element_wise_mul = torch.mul(event_1_embedding, event_2_embedding)
                this_event_pair_embedding = torch.cat((event_1_embedding, event_2_embedding, pair_element_wise_sub, pair_element_wise_mul), dim = 1)
                event_pair_embeddings = torch.cat((event_pair_embeddings, this_event_pair_embedding), dim = 0)



        # event coreference relation task, training based on event pairs instead of based on event mention clusters

        if coreference_train_method == "event_pairs":

            coreference_raw_scores = self.coreference_head_3(self.relu(self.coreference_head_2(self.relu(self.coreference_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 2
            coreference_loss = self.crossentropyloss(coreference_raw_scores, label_coreference) # size = nrow(event_pairs)

            coreference_loss_weight_0 = (label_coreference == 0).int() # weight = 1 for negative examples with label_coreference = 0
            coreference_loss_0 = torch.mul(coreference_loss, coreference_loss_weight_0)

            coreference_loss_weight_1 = torch.mul(label_coreference, coreference_weight_positive) # weight = coreference_weight_positive for positive examples with label_coreference = 1
            coreference_loss_1 = torch.mul(coreference_loss, coreference_loss_weight_1)

            coreference_weighted_loss = torch.add(coreference_loss_0, coreference_loss_1)
            coreference_weighted_loss = torch.sum(coreference_weighted_loss)

            coreference_decision = torch.argmax(coreference_raw_scores, dim = 1)

            predicted_coreference_cluster = [] # use label_coreference_cluster and predicted_coreference_cluster can calculate CoNLL metric
            for event_idx in range(number_of_events):

                event1_index_in_cluster = -1  # event_idx
                for cluster_i in range(len(predicted_coreference_cluster)):
                    if event_idx in predicted_coreference_cluster[cluster_i]:
                        event1_index_in_cluster = cluster_i

                prefix_sum = int((2 * number_of_events - 1 - event_idx) * event_idx / 2)
                for j in range(0, number_of_events - 1 - event_idx):
                    if coreference_decision[prefix_sum + j] == 1:

                        event2_index_in_cluster = -1 # (event_idx + j + 1)
                        for cluster_i in range(len(predicted_coreference_cluster)):
                            if (event_idx + j + 1) in predicted_coreference_cluster[cluster_i]:
                                event2_index_in_cluster = cluster_i

                        if event1_index_in_cluster == -1 and event2_index_in_cluster == -1:
                            predicted_coreference_cluster.append([event_idx, event_idx + j + 1])
                            event1_index_in_cluster = len(predicted_coreference_cluster) - 1
                            event2_index_in_cluster = len(predicted_coreference_cluster) - 1
                        if event1_index_in_cluster != -1 and event2_index_in_cluster == -1:
                            predicted_coreference_cluster[event1_index_in_cluster].append(event_idx + j + 1)
                            event2_index_in_cluster = event1_index_in_cluster
                        if event1_index_in_cluster == -1 and event2_index_in_cluster != -1:
                            predicted_coreference_cluster[event2_index_in_cluster].append(event_idx)
                            event1_index_in_cluster = event2_index_in_cluster
                        if event1_index_in_cluster != -1 and event2_index_in_cluster != -1 and event1_index_in_cluster != event2_index_in_cluster:
                            predicted_coreference_cluster[event1_index_in_cluster].extend(predicted_coreference_cluster[event2_index_in_cluster])
                            predicted_coreference_cluster.pop(event2_index_in_cluster)
                            # recalculate event1_index_in_cluster for event_idx because one element pop out
                            for cluster_i in range(len(predicted_coreference_cluster)):
                                if event_idx in predicted_coreference_cluster[cluster_i]:
                                    event1_index_in_cluster = cluster_i
                            event2_index_in_cluster = event1_index_in_cluster

                if event1_index_in_cluster == -1: # singleton
                    predicted_coreference_cluster.append([event_idx])

            for cluster_i in range(len(predicted_coreference_cluster)):
                predicted_coreference_cluster[cluster_i] = sorted(predicted_coreference_cluster[cluster_i])
            predicted_coreference_cluster = sorted(predicted_coreference_cluster, key=lambda x: x[0])



        # event temporal relation task

        temporal_raw_scores = self.temporal_head_3(self.relu(self.temporal_head_2(self.relu(self.temporal_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 4
        temporal_loss = self.crossentropyloss(temporal_raw_scores, label_temporal) # size = nrow(event_pairs)

        temporal_loss_weight_0 = (label_temporal == 0).int() # weight = 1 for negative examples with label_temporal = 0
        temporal_loss_0 = torch.mul(temporal_loss, temporal_loss_weight_0)

        temporal_loss_weight_1 = torch.mul((label_temporal == 1).int(), temporal_weight_before)
        temporal_loss_1 = torch.mul(temporal_loss, temporal_loss_weight_1)

        temporal_loss_weight_2 = torch.mul((label_temporal == 2).int(), temporal_weight_after)
        temporal_loss_2 = torch.mul(temporal_loss, temporal_loss_weight_2)

        temporal_loss_weight_3 = torch.mul((label_temporal == 3).int(), temporal_weight_overlap)
        temporal_loss_3 = torch.mul(temporal_loss, temporal_loss_weight_3)

        temporal_weighted_loss = torch.add(torch.add(torch.add(temporal_loss_0, temporal_loss_1), temporal_loss_2), temporal_loss_3)
        temporal_weighted_loss = torch.sum(temporal_weighted_loss)


        # event causal relation task

        causal_raw_scores = self.causal_head_3(self.relu(self.causal_head_2(self.relu(self.causal_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 3
        causal_loss = self.crossentropyloss(causal_raw_scores, label_causal) # size = nrow(event_pairs)

        causal_loss_weight_0 = (label_causal == 0).int() # weight = 1 for negative examples with label_causal = 0
        causal_loss_0 = torch.mul(causal_loss, causal_loss_weight_0)

        causal_loss_weight_1 = torch.mul((label_causal == 1).int(), causal_weight_cause)
        causal_loss_1 = torch.mul(causal_loss, causal_loss_weight_1)

        causal_loss_weight_2 = torch.mul((label_causal == 2).int(), causal_weight_caused)
        causal_loss_2 = torch.mul(causal_loss, causal_loss_weight_2)

        causal_weighted_loss = torch.add(torch.add(causal_loss_0, causal_loss_1), causal_loss_2)
        causal_weighted_loss = torch.sum(causal_weighted_loss)


        # event subevent relation task

        subevent_raw_scores = self.subevent_head_3(self.relu(self.subevent_head_2(self.relu(self.subevent_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 3
        subevent_loss = self.crossentropyloss(subevent_raw_scores, label_subevent) # size = nrow(event_pairs)

        subevent_loss_weight_0 = (label_subevent == 0).int() # weight = 1 for negative examples with label_subevent = 0
        subevent_loss_0 = torch.mul(subevent_loss, subevent_loss_weight_0)

        subevent_loss_weight_1 = torch.mul((label_subevent == 1).int(), subevent_weight_contain)
        subevent_loss_1 = torch.mul(subevent_loss, subevent_loss_weight_1)

        subevent_loss_weight_2 = torch.mul((label_subevent == 2).int(), subevent_weight_contained)
        subevent_loss_2 = torch.mul(subevent_loss, subevent_loss_weight_2)

        subevent_weighted_loss = torch.add(torch.add(subevent_loss_0, subevent_loss_1), subevent_loss_2)
        subevent_weighted_loss = torch.sum(subevent_weighted_loss)


        # event coreference relation task, training based on event mention clusters instead of based on event pairs

        if coreference_train_method == "event_cluster":

            event_pairs_sorted_by_event2 = torch.stack(sorted(event_pairs, key = lambda x: (x[1], x[0])))

            for event_pair_i in range(event_pairs_sorted_by_event2.shape[0]):
                if event_pair_i == 0:
                    event_1_in_label_event = event_pairs_sorted_by_event2[event_pair_i, 0]
                    event_2_in_label_event = event_pairs_sorted_by_event2[event_pair_i, 1]
                    event_1_embedding = event_embeddings[event_1_in_label_event, :].view(1, 768)
                    event_2_embedding = event_embeddings[event_2_in_label_event, :].view(1, 768)
                    pair_element_wise_sub = torch.sub(event_1_embedding, event_2_embedding)
                    pair_element_wise_mul = torch.mul(event_1_embedding, event_2_embedding)
                    event_pair_embeddings = torch.cat((event_1_embedding, event_2_embedding, pair_element_wise_sub, pair_element_wise_mul), dim = 1) # 1 * (768 * 4)
                else:
                    event_1_in_label_event = event_pairs_sorted_by_event2[event_pair_i, 0]
                    event_2_in_label_event = event_pairs_sorted_by_event2[event_pair_i, 1]
                    event_1_embedding = event_embeddings[event_1_in_label_event, :].view(1, 768)
                    event_2_embedding = event_embeddings[event_2_in_label_event, :].view(1, 768)
                    pair_element_wise_sub = torch.sub(event_1_embedding, event_2_embedding)
                    pair_element_wise_mul = torch.mul(event_1_embedding, event_2_embedding)
                    this_event_pair_embedding = torch.cat((event_1_embedding, event_2_embedding, pair_element_wise_sub, pair_element_wise_mul), dim = 1)
                    event_pair_embeddings = torch.cat((event_pair_embeddings, this_event_pair_embedding), dim = 0)

            coreference_raw_scores = self.coreference_head_3(self.relu(self.coreference_head_2(self.relu(self.coreference_head_1(event_pair_embeddings))))) # nrow = nrow(event_pairs), ncol = 2
            coreference_raw_scores = coreference_raw_scores[:,0]
            split_scores = [to_cuda(torch.tensor([]))] + list(torch.split(coreference_raw_scores, [i for i in range(number_of_events) if i], dim=0))  # first event has no valid antecedent
            epsilon = to_var(torch.tensor([0.]))  # dummy score default to 0.0
            with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]  # dummy index default to same index as itself
            coreference_probs = [F.softmax(tensor, dim=0) for tensor in with_epsilon]
            coreference_probs = pad_and_stack(coreference_probs, value = -100.0) # use label_coreference_cluster and coreference_probs can calculate CoNLL metric

            filled_labels = fill_expand(label_coreference_cluster)
            filled_labels = to_cuda(filled_labels)
            eps = 1e-8
            prob_sum = torch.sum(torch.clamp(torch.mul(coreference_probs, filled_labels), eps, 1-eps), dim=1)
            coreference_cluster_loss = torch.sum(torch.log(prob_sum)) * -1 # torch.mean(torch.log(prob_sum)) * -1



        if coreference_train_method == "event_pairs":
            return event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
                   subevent_weighted_loss, subevent_raw_scores, coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster

        if coreference_train_method == "event_cluster":
            return event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
                   subevent_weighted_loss, subevent_raw_scores, coreference_cluster_loss, coreference_probs, label_coreference_cluster






'''evaluate'''

def get_event2cluster(clusters):
    event2cluster = {}
    for cluster in clusters:
        for eid in cluster:
            event2cluster[eid] = tuple(cluster)
    return event2cluster

def get_clusters(event2cluster):
    clusters = list(set(event2cluster.values()))
    return clusters

def get_predicted_clusters(prob):
    predicted_antecedents = torch.argmax(prob, dim=-1).cpu().numpy().tolist()
    idx_to_clusters = {}
    for i in range(len(predicted_antecedents)):
        idx_to_clusters[i] = set([i])

    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index >= i:
            assert predicted_index == i
            continue
        else:
            union_cluster = idx_to_clusters[predicted_index] | idx_to_clusters[i]
            for j in union_cluster:
                idx_to_clusters[j] = union_cluster
    idx_to_clusters = {i: tuple(sorted(idx_to_clusters[i])) for i in idx_to_clusters}
    predicted_clusters = get_clusters(idx_to_clusters)
    return predicted_clusters, idx_to_clusters


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem

def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p

def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

def ceafe(clusters, gold_clusters):
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_id, col_id = linear_sum_assignment(-scores)
    similarity = sum(scores[row_id, col_id])
    return similarity, len(clusters), similarity, len(gold_clusters)

def blanc(mention_to_cluster, mention_to_gold):
    rc = 0
    wc = 0
    rn = 0
    wn = 0
    assert len(mention_to_cluster) == len(mention_to_gold)
    mentions = list(mention_to_cluster.keys())
    for i in range(len(mentions)):
        for j in range(i + 1, len(mentions)):
            if mention_to_cluster[mentions[i]] == mention_to_cluster[mentions[j]]:
                if mention_to_gold[mentions[i]] == mention_to_gold[mentions[j]]:
                    rc += 1
                else:
                    wc += 1
            else:
                if mention_to_gold[mentions[i]] == mention_to_gold[mentions[j]]:
                    wn += 1
                else:
                    rn += 1
    return rc, wc, rn, wn


class MUC:
    def __init__(self, beta = 1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = muc
        self.beta = beta
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster):
        pn, pd = self.metric(pred_cluster, gold_event2cluster)
        rn, rd = self.metric(gold_cluster, pred_event2cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class B_CUBED:
    def __init__(self, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = b_cubed
        self.beta = beta
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster):
        pn, pd = self.metric(pred_cluster, gold_event2cluster)
        rn, rd = self.metric(gold_cluster, pred_event2cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class CEAFE:
    def __init__(self, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = ceafe
        self.beta = beta
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster):
        pn, pd, rn, rd = self.metric(pred_cluster, gold_cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class BLANC:
    def __init__(self, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = blanc
        self.beta = beta
        self.rc = 0
        self.wc = 0
        self.rn = 0
        self.wn = 0

    def update(self, gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster):
        rc, wc, rn, wn = self.metric(pred_event2cluster, gold_event2cluster)
        self.rc += rc
        self.wc += wc
        self.rn += rn
        self.wn += wn

    def get_f1(self):
        return (f1(self.rc, self.rc+self.wc, self.rc, self.rc+self.wn, beta=self.beta) + f1(self.rn, self.rn+self.wn, self.rn, self.rn+self.wc, beta=self.beta)) / 2

    def get_recall(self):
        return (self.rc/(self.rc+self.wn+1e-6) + self.rn/(self.rn+self.wc+1e-6)) / 2

    def get_precision(self):
        return (self.rc/(self.rc+self.wc+1e-6) + self.rn/(self.rn+self.wn+1e-6)) / 2

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()



def evaluate(event_relation_graph, eval_dataloader, verbose):

    event_relation_graph.eval()

    if coreference_train_method == "event_pairs":

        muc_evaluator = MUC()
        bcubed_evaluator = B_CUBED()
        ceafe_evaluator = CEAFE()
        blanc_evaluator = BLANC()

        for step, batch in enumerate(eval_dataloader):

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label_event = batch['label_event']
            event_pairs= batch['event_pairs']
            label_coreference = batch['label_coreference']
            label_temporal = batch['label_temporal']
            label_causal = batch['label_causal']
            label_subevent = batch['label_subevent']

            label_event = label_event[0, :, :]
            number_of_events = torch.sum((label_event[:, 2] == 1).int())
            if number_of_events == 1:
                continue
            event_pairs = event_pairs[0, :, :]
            label_coreference = label_coreference[0, :]
            label_temporal = label_temporal[0, :]
            label_causal = label_causal[0, :]
            label_subevent = label_subevent[0, :]

            input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent = \
                input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), \
                label_coreference.to(device), label_temporal.to(device), label_causal.to(device), label_subevent.to(device)

            with torch.no_grad():
                event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
                subevent_weighted_loss, subevent_raw_scores, coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster = \
                    event_relation_graph(input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method)


            gold_event2cluster = get_event2cluster(label_coreference_cluster)
            gold_cluster = label_coreference_cluster
            pred_event2cluster = get_event2cluster(predicted_coreference_cluster)
            pred_cluster = predicted_coreference_cluster

            muc_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            bcubed_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            ceafe_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            blanc_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)


            decision_event = torch.argmax(event_raw_scores, dim = 1).view(event_raw_scores.shape[0], 1) # batch_size * 1
            true_label_event = label_event[:,2].view(event_raw_scores.shape[0], 1)

            if step == 0:
                decision_event_onetest = decision_event
                true_label_event_onetest = true_label_event
            else:
                decision_event_onetest = torch.cat((decision_event_onetest, decision_event), dim=0)
                true_label_event_onetest = torch.cat((true_label_event_onetest, true_label_event), dim=0)


            decision_temporal = torch.argmax(temporal_raw_scores, dim = 1).view(temporal_raw_scores.shape[0], 1) # batch_size * 1
            true_label_temporal = label_temporal.view(temporal_raw_scores.shape[0], 1)

            if step == 0:
                decision_temporal_onetest = decision_temporal
                true_label_temporal_onetest = true_label_temporal
            else:
                decision_temporal_onetest = torch.cat((decision_temporal_onetest, decision_temporal), dim=0)
                true_label_temporal_onetest = torch.cat((true_label_temporal_onetest, true_label_temporal), dim=0)


            decision_causal = torch.argmax(causal_raw_scores, dim = 1).view(causal_raw_scores.shape[0], 1) # batch_size * 1
            true_label_causal = label_causal.view(causal_raw_scores.shape[0], 1)

            if step == 0:
                decision_causal_onetest = decision_causal
                true_label_causal_onetest = true_label_causal
            else:
                decision_causal_onetest = torch.cat((decision_causal_onetest, decision_causal), dim=0)
                true_label_causal_onetest = torch.cat((true_label_causal_onetest, true_label_causal), dim=0)


            decision_subevent = torch.argmax(subevent_raw_scores, dim = 1).view(subevent_raw_scores.shape[0], 1) # batch_size * 1
            true_label_subevent = label_subevent.view(subevent_raw_scores.shape[0], 1)

            if step == 0:
                decision_subevent_onetest = decision_subevent
                true_label_subevent_onetest = true_label_subevent
            else:
                decision_subevent_onetest = torch.cat((decision_subevent_onetest, decision_subevent), dim=0)
                true_label_subevent_onetest = torch.cat((true_label_subevent_onetest, true_label_subevent), dim=0)


            decision_coreference = torch.argmax(coreference_raw_scores, dim = 1).view(coreference_raw_scores.shape[0], 1) # batch_size * 1
            true_label_coreference = label_coreference.view(coreference_raw_scores.shape[0], 1)

            if step == 0:
                decision_coreference_onetest = decision_coreference
                true_label_coreference_onetest = true_label_coreference
            else:
                decision_coreference_onetest = torch.cat((decision_coreference_onetest, decision_coreference), dim=0)
                true_label_coreference_onetest = torch.cat((true_label_coreference_onetest, true_label_coreference), dim=0)



        decision_event_onetest = decision_event_onetest.to('cpu').numpy()
        true_label_event_onetest = true_label_event_onetest.to('cpu').numpy()

        if verbose:
            print("======== Event Identification Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average=None)[:3])

        macro_F_event = precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average='macro')[2]


        decision_temporal_onetest = decision_temporal_onetest.to('cpu').numpy()
        true_label_temporal_onetest = true_label_temporal_onetest.to('cpu').numpy()

        if verbose:
            print("======== Temporal Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average=None)[:3])

        macro_F_temporal = precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average='macro')[2]


        decision_causal_onetest = decision_causal_onetest.to('cpu').numpy()
        true_label_causal_onetest = true_label_causal_onetest.to('cpu').numpy()

        if verbose:
            print("======== Causal Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average=None)[:3])

        macro_F_causal = precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average='macro')[2]


        decision_subevent_onetest = decision_subevent_onetest.to('cpu').numpy()
        true_label_subevent_onetest = true_label_subevent_onetest.to('cpu').numpy()

        if verbose:
            print("======== Subevent Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average=None)[:3])

        macro_F_subevent = precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average='macro')[2]


        decision_coreference_onetest = decision_coreference_onetest.to('cpu').numpy()
        true_label_coreference_onetest = true_label_coreference_onetest.to('cpu').numpy()

        if verbose:
            print("======== Coreference Relation Task - event pairs ========")
            print("Macro: ", precision_recall_fscore_support(true_label_coreference_onetest, decision_coreference_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_coreference_onetest, decision_coreference_onetest, average=None)[:3])

        macro_F_coreference = precision_recall_fscore_support(true_label_coreference_onetest, decision_coreference_onetest, average='macro')[2]


        muc_precision, muc_recall, muc_F = muc_evaluator.get_prf()
        bcubed_precision, bcubed_recall, bcubed_F = bcubed_evaluator.get_prf()
        ceafe_precision, ceafe_recall, ceafe_F = ceafe_evaluator.get_prf()
        blanc_precision, blanc_recall, blanc_F = blanc_evaluator.get_prf()
        conll_F_coreference = (muc_F + bcubed_F + ceafe_F + blanc_F) / 4

        if verbose:
            print("======== Coreference Relation Task - event mention clusters ========")
            print('MUC precision: {:.3f}, MUC recall: {:.3f}, MUC F: {:.3f}'.format(muc_precision, muc_recall, muc_F))
            print('BCUBED precision: {:.3f}, BCUBED recall: {:.3f}, BCUBED F: {:.3f}'.format(bcubed_precision, bcubed_recall, bcubed_F))
            print('CEAFE precision: {:.3f}, CEAFE recall: {:.3f}, CEAFE F: {:.3f}'.format(ceafe_precision, ceafe_recall, ceafe_F))
            print('BLANC precision: {:.3f}, BLANC recall: {:.3f}, BLANC F: {:.3f}'.format(blanc_precision, blanc_recall, blanc_F))


        macro_F_graph = (macro_F_event + macro_F_temporal + macro_F_causal + macro_F_subevent + conll_F_coreference) / 5

        if verbose:
            print('macro_F_graph: {:.3f}'.format(macro_F_graph))


        return macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph




    if coreference_train_method == "event_cluster":

        muc_evaluator = MUC()
        bcubed_evaluator = B_CUBED()
        ceafe_evaluator = CEAFE()
        blanc_evaluator = BLANC()

        for step, batch in enumerate(eval_dataloader):

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label_event = batch['label_event']
            event_pairs = batch['event_pairs']
            label_coreference = batch['label_coreference']
            label_temporal = batch['label_temporal']
            label_causal = batch['label_causal']
            label_subevent = batch['label_subevent']

            label_event = label_event[0, :, :]
            number_of_events = torch.sum((label_event[:, 2] == 1).int())
            if number_of_events == 1:
                continue
            event_pairs = event_pairs[0, :, :]
            label_coreference = label_coreference[0, :]
            label_temporal = label_temporal[0, :]
            label_causal = label_causal[0, :]
            label_subevent = label_subevent[0, :]

            input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent = \
                input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), \
                label_coreference.to(device), label_temporal.to(device), label_causal.to(device), label_subevent.to(device)

            with torch.no_grad():
                event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
                subevent_weighted_loss, subevent_raw_scores, coreference_cluster_loss, coreference_probs, label_coreference_cluster = \
                    event_relation_graph(input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method)


            pred_cluster, pred_event2cluster = get_predicted_clusters(coreference_probs)
            gold_event2cluster = get_event2cluster(label_coreference_cluster)
            gold_cluster = label_coreference_cluster

            muc_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            bcubed_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            ceafe_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)
            blanc_evaluator.update(gold_cluster, gold_event2cluster, pred_cluster, pred_event2cluster)


            decision_event = torch.argmax(event_raw_scores, dim = 1).view(event_raw_scores.shape[0], 1) # batch_size * 1
            true_label_event = label_event[:,2].view(event_raw_scores.shape[0], 1)

            if step == 0:
                decision_event_onetest = decision_event
                true_label_event_onetest = true_label_event
            else:
                decision_event_onetest = torch.cat((decision_event_onetest, decision_event), dim=0)
                true_label_event_onetest = torch.cat((true_label_event_onetest, true_label_event), dim=0)


            decision_temporal = torch.argmax(temporal_raw_scores, dim = 1).view(temporal_raw_scores.shape[0], 1) # batch_size * 1
            true_label_temporal = label_temporal.view(temporal_raw_scores.shape[0], 1)

            if step == 0:
                decision_temporal_onetest = decision_temporal
                true_label_temporal_onetest = true_label_temporal
            else:
                decision_temporal_onetest = torch.cat((decision_temporal_onetest, decision_temporal), dim=0)
                true_label_temporal_onetest = torch.cat((true_label_temporal_onetest, true_label_temporal), dim=0)


            decision_causal = torch.argmax(causal_raw_scores, dim = 1).view(causal_raw_scores.shape[0], 1) # batch_size * 1
            true_label_causal = label_causal.view(causal_raw_scores.shape[0], 1)

            if step == 0:
                decision_causal_onetest = decision_causal
                true_label_causal_onetest = true_label_causal
            else:
                decision_causal_onetest = torch.cat((decision_causal_onetest, decision_causal), dim=0)
                true_label_causal_onetest = torch.cat((true_label_causal_onetest, true_label_causal), dim=0)


            decision_subevent = torch.argmax(subevent_raw_scores, dim = 1).view(subevent_raw_scores.shape[0], 1) # batch_size * 1
            true_label_subevent = label_subevent.view(subevent_raw_scores.shape[0], 1)

            if step == 0:
                decision_subevent_onetest = decision_subevent
                true_label_subevent_onetest = true_label_subevent
            else:
                decision_subevent_onetest = torch.cat((decision_subevent_onetest, decision_subevent), dim=0)
                true_label_subevent_onetest = torch.cat((true_label_subevent_onetest, true_label_subevent), dim=0)



        decision_event_onetest = decision_event_onetest.to('cpu').numpy()
        true_label_event_onetest = true_label_event_onetest.to('cpu').numpy()

        if verbose:
            print("======== Event Identification Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average=None)[:3])

        macro_F_event = precision_recall_fscore_support(true_label_event_onetest, decision_event_onetest, average='macro')[2]


        decision_temporal_onetest = decision_temporal_onetest.to('cpu').numpy()
        true_label_temporal_onetest = true_label_temporal_onetest.to('cpu').numpy()

        if verbose:
            print("======== Temporal Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average=None)[:3])

        macro_F_temporal = precision_recall_fscore_support(true_label_temporal_onetest, decision_temporal_onetest, average='macro')[2]


        decision_causal_onetest = decision_causal_onetest.to('cpu').numpy()
        true_label_causal_onetest = true_label_causal_onetest.to('cpu').numpy()

        if verbose:
            print("======== Causal Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average=None)[:3])

        macro_F_causal = precision_recall_fscore_support(true_label_causal_onetest, decision_causal_onetest, average='macro')[2]


        decision_subevent_onetest = decision_subevent_onetest.to('cpu').numpy()
        true_label_subevent_onetest = true_label_subevent_onetest.to('cpu').numpy()

        if verbose:
            print("======== Subevent Relation Task ========")
            print("Macro: ", precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average='macro'))
            print("None: ", precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average=None)[:3])

        macro_F_subevent = precision_recall_fscore_support(true_label_subevent_onetest, decision_subevent_onetest, average='macro')[2]


        muc_precision, muc_recall, muc_F = muc_evaluator.get_prf()
        bcubed_precision, bcubed_recall, bcubed_F = bcubed_evaluator.get_prf()
        ceafe_precision, ceafe_recall, ceafe_F = ceafe_evaluator.get_prf()
        blanc_precision, blanc_recall, blanc_F = blanc_evaluator.get_prf()
        conll_F_coreference = (muc_F + bcubed_F + ceafe_F + blanc_F) / 4

        if verbose:
            print("======== Coreference Relation Task - event mention clusters ========")
            print('MUC precision: {:.3f}, MUC recall: {:.3f}, MUC F: {:.3f}'.format(muc_precision, muc_recall, muc_F))
            print('BCUBED precision: {:.3f}, BCUBED recall: {:.3f}, BCUBED F: {:.3f}'.format(bcubed_precision, bcubed_recall, bcubed_F))
            print('CEAFE precision: {:.3f}, CEAFE recall: {:.3f}, CEAFE F: {:.3f}'.format(ceafe_precision, ceafe_recall, ceafe_F))
            print('BLANC precision: {:.3f}, BLANC recall: {:.3f}, BLANC F: {:.3f}'.format(blanc_precision, blanc_recall, blanc_F))


        macro_F_graph = (macro_F_event + macro_F_temporal + macro_F_causal + macro_F_subevent + conll_F_coreference) / 5

        if verbose:
            print('macro_F_graph: {:.3f}'.format(macro_F_graph))


        return macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph







'''train'''

import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()




seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


event_relation_graph = Event_Relation_Graph()
event_relation_graph.cuda()


param_all = list(event_relation_graph.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('longformer' in n))],
     'lr': longformer_lr, 'weight_decay': longformer_weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'longformer' in n))],
     'lr': non_longformer_lr, 'weight_decay': non_longformer_weight_decay},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('longformer' in n))],
     'lr': longformer_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'longformer' in n))],
     'lr': non_longformer_lr, 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)


train_path = "./MAVEN_ERE/train/"
train_file_names = os.listdir(train_path)
train_file_paths = []
train_file_paths = create_file_path(train_path, train_file_names)

dev_path = "./MAVEN_ERE/dev/"
dev_file_names = os.listdir(dev_path)
dev_file_paths = []
dev_file_paths = create_file_path(dev_path, dev_file_names)

test_path = "./MAVEN_ERE/test/"
test_file_names = os.listdir(test_path)
test_file_paths = []
test_file_paths = create_file_path(test_path, test_file_names)

train_dataset = custom_dataset(train_file_paths)
dev_dataset = custom_dataset(dev_file_paths)
test_dataset = custom_dataset(test_file_paths)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_train_steps = num_epochs * len(train_dataloader)
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)


best_macro_F_event = 0
best_macro_F_coreference = 0
best_conll_F_coreference = 0 # average of MUC, B_cube, CEAF_e, blanc F1 scores
best_macro_F_temporal = 0
best_macro_F_causal = 0
best_macro_F_subevent = 0
best_macro_F_graph = 0



for epoch_i in range(num_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    total_event_loss = 0
    total_coreference_loss = 0
    total_temporal_loss = 0
    total_causal_loss = 0
    total_subevent_loss = 0
    num_batch = 0

    for step, batch in enumerate(train_dataloader):

        if step % ((len(train_dataloader) * num_epochs) // check_times) == 0:

            elapsed = format_time(time.time() - t0)

            if num_batch != 0:
                avg_event_loss = total_event_loss / num_batch
                avg_coreference_loss = total_coreference_loss / num_batch
                avg_temporal_loss = total_temporal_loss / num_batch
                avg_causal_loss = total_causal_loss / num_batch
                avg_subevent_loss = total_subevent_loss / num_batch

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Event Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_event_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Coreference Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_coreference_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Temporal Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_temporal_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Causal Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_causal_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Subevent Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_subevent_loss))

            else:
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            total_event_loss = 0
            total_coreference_loss = 0
            total_temporal_loss = 0
            total_causal_loss = 0
            total_subevent_loss = 0
            num_batch = 0

            # evaluate on dev set

            if coreference_train_method == "event_pairs":
                macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
                    evaluate(event_relation_graph, dev_dataloader, verbose = 1)

                if macro_F_coreference > best_macro_F_coreference:
                    torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_coreference.ckpt')
                    best_macro_F_coreference = macro_F_coreference

            if coreference_train_method == "event_cluster":
                macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
                    evaluate(event_relation_graph, dev_dataloader, verbose = 1)

            if macro_F_event > best_macro_F_event:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_event.ckpt')
                best_macro_F_event = macro_F_event
            if macro_F_temporal > best_macro_F_temporal:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_temporal.ckpt')
                best_macro_F_temporal = macro_F_temporal
            if macro_F_causal > best_macro_F_causal:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_causal.ckpt')
                best_macro_F_causal = macro_F_causal
            if macro_F_subevent > best_macro_F_subevent:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_subevent.ckpt')
                best_macro_F_subevent = macro_F_subevent
            if conll_F_coreference > best_conll_F_coreference:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_conll_F_coreference.ckpt')
                best_conll_F_coreference = conll_F_coreference
            if macro_F_graph > best_macro_F_graph:
                torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_graph.ckpt')
                best_macro_F_graph = macro_F_graph




        # train

        event_relation_graph.train()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label_event = batch['label_event']
        event_pairs = batch['event_pairs']
        label_coreference = batch['label_coreference']
        label_temporal = batch['label_temporal']
        label_causal = batch['label_causal']
        label_subevent = batch['label_subevent']

        label_event = label_event[0, :, :]
        number_of_events = torch.sum((label_event[:, 2] == 1).int())
        if number_of_events == 1:
            continue
        event_pairs = event_pairs[0, :, :]
        label_coreference = label_coreference[0, :]
        label_temporal = label_temporal[0, :]
        label_causal = label_causal[0, :]
        label_subevent = label_subevent[0, :]

        input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent = \
            input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), \
            label_coreference.to(device), label_temporal.to(device), label_causal.to(device), label_subevent.to(device)

        optimizer.zero_grad()


        if coreference_train_method == "event_pairs":

            event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
            subevent_weighted_loss, subevent_raw_scores, coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster = \
                event_relation_graph(input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method)

            total_event_loss += event_weighted_loss.item()
            total_coreference_loss += coreference_weighted_loss.item()
            total_temporal_loss += temporal_weighted_loss.item()
            total_causal_loss += causal_weighted_loss.item()
            total_subevent_loss += subevent_weighted_loss.item()
            num_batch += 1

            event_weighted_loss.backward(retain_graph = True)
            coreference_weighted_loss.backward(retain_graph = True)
            temporal_weighted_loss.backward(retain_graph = True)
            causal_weighted_loss.backward(retain_graph = True)
            subevent_weighted_loss.backward()

            torch.nn.utils.clip_grad_norm_(event_relation_graph.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if coreference_train_method == "event_cluster":

            event_weighted_loss, event_raw_scores, temporal_weighted_loss, temporal_raw_scores, causal_weighted_loss, causal_raw_scores, \
            subevent_weighted_loss, subevent_raw_scores, coreference_cluster_loss, coreference_probs, label_coreference_cluster = \
                event_relation_graph(input_ids, attention_mask, label_event, event_pairs, label_coreference, label_temporal, label_causal, label_subevent, coreference_train_method)

            total_event_loss += event_weighted_loss.item()
            total_coreference_loss += coreference_cluster_loss.item()
            total_temporal_loss += temporal_weighted_loss.item()
            total_causal_loss += causal_weighted_loss.item()
            total_subevent_loss += subevent_weighted_loss.item()
            num_batch += 1

            event_weighted_loss.backward(retain_graph = True)
            coreference_cluster_loss.backward(retain_graph = True)
            temporal_weighted_loss.backward(retain_graph = True)
            causal_weighted_loss.backward(retain_graph = True)
            subevent_weighted_loss.backward()

            torch.nn.utils.clip_grad_norm_(event_relation_graph.parameters(), 1.0)
            optimizer.step()
            scheduler.step()



    elapsed = format_time(time.time() - t0)

    if num_batch != 0:
        avg_event_loss = total_event_loss / num_batch
        avg_coreference_loss = total_coreference_loss / num_batch
        avg_temporal_loss = total_temporal_loss / num_batch
        avg_causal_loss = total_causal_loss / num_batch
        avg_subevent_loss = total_subevent_loss / num_batch

        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Event Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_event_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Coreference Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_coreference_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Temporal Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_temporal_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Causal Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_causal_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Subevent Training Loss Average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_subevent_loss))

    else:
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

    total_event_loss = 0
    total_coreference_loss = 0
    total_temporal_loss = 0
    total_causal_loss = 0
    total_subevent_loss = 0
    num_batch = 0

    # evaluate on dev set

    if coreference_train_method == "event_pairs":
        macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
            evaluate(event_relation_graph, dev_dataloader, verbose = 1)

        if macro_F_coreference > best_macro_F_coreference:
            torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_coreference.ckpt')
            best_macro_F_coreference = macro_F_coreference

    if coreference_train_method == "event_cluster":
        macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
            evaluate(event_relation_graph, dev_dataloader, verbose = 1)

    if macro_F_event > best_macro_F_event:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_event.ckpt')
        best_macro_F_event = macro_F_event
    if macro_F_temporal > best_macro_F_temporal:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_temporal.ckpt')
        best_macro_F_temporal = macro_F_temporal
    if macro_F_causal > best_macro_F_causal:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_causal.ckpt')
        best_macro_F_causal = macro_F_causal
    if macro_F_subevent > best_macro_F_subevent:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_subevent.ckpt')
        best_macro_F_subevent = macro_F_subevent
    if conll_F_coreference > best_conll_F_coreference:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_conll_F_coreference.ckpt')
        best_conll_F_coreference = conll_F_coreference
    if macro_F_graph > best_macro_F_graph:
        torch.save(event_relation_graph.state_dict(),'./saved_models/event_relation_graph/best_macro_F_graph.ckpt')
        best_macro_F_graph = macro_F_graph




# test

print("Testing...")

print("best_macro_F_graph on dev is: {:}".format(best_macro_F_graph))

event_relation_graph = Event_Relation_Graph()
event_relation_graph.cuda()
event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_graph.ckpt', map_location=device))

if coreference_train_method == "event_pairs":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)

if coreference_train_method == "event_cluster":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)


print("best_macro_F_event on dev is: {:}".format(best_macro_F_event))

event_relation_graph = Event_Relation_Graph()
event_relation_graph.cuda()
event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_event.ckpt', map_location=device))

if coreference_train_method == "event_pairs":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)

if coreference_train_method == "event_cluster":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)


print("best_macro_F_temporal on dev is: {:}".format(best_macro_F_temporal))

event_relation_graph = Event_Relation_Graph()
event_relation_graph.cuda()
event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_temporal.ckpt', map_location=device))

if coreference_train_method == "event_pairs":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)

if coreference_train_method == "event_cluster":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)


print("best_macro_F_causal on dev is: {:}".format(best_macro_F_causal))

event_relation_graph = Event_Relation_Graph()
event_relation_graph.cuda()
event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_causal.ckpt', map_location=device))

if coreference_train_method == "event_pairs":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)

if coreference_train_method == "event_cluster":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)


print("best_macro_F_subevent on dev is: {:}".format(best_macro_F_subevent))

event_relation_graph = Event_Relation_Graph()
event_relation_graph.cuda()
event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_subevent.ckpt', map_location=device))

if coreference_train_method == "event_pairs":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)

if coreference_train_method == "event_cluster":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)


print("best_conll_F_coreference on dev is: {:}".format(best_conll_F_coreference))

event_relation_graph = Event_Relation_Graph()
event_relation_graph.cuda()
event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_conll_F_coreference.ckpt', map_location=device))

if coreference_train_method == "event_pairs":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)

if coreference_train_method == "event_cluster":
    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)


if coreference_train_method == "event_pairs":

    print("best_macro_F_coreference on dev is: {:}".format(best_macro_F_coreference))

    event_relation_graph = Event_Relation_Graph()
    event_relation_graph.cuda()
    event_relation_graph.load_state_dict(torch.load('./saved_models/event_relation_graph/best_macro_F_coreference.ckpt', map_location=device))

    macro_F_event, macro_F_temporal, macro_F_causal, macro_F_subevent, macro_F_coreference, conll_F_coreference, macro_F_graph = \
        evaluate(event_relation_graph, test_dataloader, verbose=1)




# stop here
