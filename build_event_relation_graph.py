
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
import nltk
nltk.download('punkt')



'''hyper-parameters'''

MAX_LEN = 2048
event_weight_positive = 1
coreference_weight_positive = 1
temporal_weight_before = 1 # temporal label 1 before
temporal_weight_after = 1 # temporal label 2 after
temporal_weight_overlap = 1 # temporal label 3 overlap
causal_weight_cause = 1 # causal label 1 cause
causal_weight_caused = 1 # causal label 2 caused by
subevent_weight_contain = 1 # subevent label 1 contain
subevent_weight_contained = 1 # subevent label 2 contained by



softmax = nn.Softmax(dim = 1)
softmax.cuda()
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()




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

class Event_Identification(nn.Module):

    # input: label_event, number of tokens for event identification * 3 (start in input_ids, end in input_ids, label_event)
    #        event_pairs, number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
    #        label_relation, size = number of event pairs

    def __init__(self):
        super(Event_Identification, self).__init__()

        self.token_embedding = Token_Embedding()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)


        self.event_head_1 = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.event_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_1.bias)

        self.event_head_2 = nn.Linear(768, 2, bias=True)
        nn.init.xavier_uniform_(self.event_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_2.bias)

        self.relu = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none') # no reduction


    def forward(self, input_ids, attention_mask, label_event):

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

        return event_weighted_loss, event_raw_scores

event_identification = Event_Identification()
event_identification.cuda()
event_identification.load_state_dict(torch.load('./saved_models/event_relation_graph/best_positive_F_event.ckpt', map_location=device))
event_identification.eval()


class Event_Coreference(nn.Module):

    # input: label_event, number of tokens for event identification * 3 (start in input_ids, end in input_ids, label_event)
    #        event_pairs, number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
    #        label_relation, size = number of event pairs

    def __init__(self):
        super(Event_Coreference, self).__init__()

        self.token_embedding = Token_Embedding()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)

        self.coreference_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.coreference_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.coreference_head_1.bias)

        self.coreference_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.coreference_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.coreference_head_2.bias)

        self.coreference_head_3 = nn.Linear(256, 2, bias=True)
        nn.init.xavier_uniform_(self.coreference_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.coreference_head_3.bias)


        self.relu = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none') # no reduction



    def forward(self, input_ids, attention_mask, label_event, event_pairs, label_coreference):

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


        return coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster

event_coreference = Event_Coreference()
event_coreference.cuda()
event_coreference.load_state_dict(torch.load('./saved_models/event_relation_graph/best_positive_F_coreference.ckpt', map_location=device))
event_coreference.eval()


class Event_Temporal(nn.Module):

    # input: label_event, number of tokens for event identification * 3 (start in input_ids, end in input_ids, label_event)
    #        event_pairs, number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
    #        label_relation, size = number of event pairs

    def __init__(self):
        super(Event_Temporal, self).__init__()

        self.token_embedding = Token_Embedding()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)

        self.temporal_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_1.bias)

        self.temporal_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_2.bias)

        self.temporal_head_3 = nn.Linear(256, 4, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_3.bias)


        self.relu = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none') # no reduction


    def forward(self, input_ids, attention_mask, label_event, event_pairs, label_temporal):

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


        return temporal_weighted_loss, temporal_raw_scores

event_temporal = Event_Temporal()
event_temporal.cuda()
event_temporal.load_state_dict(torch.load('./saved_models/event_relation_graph/best_positive_F_temporal.ckpt', map_location=device))
event_temporal.eval()


class Event_Causal(nn.Module):

    # input: label_event, number of tokens for event identification * 3 (start in input_ids, end in input_ids, label_event)
    #        event_pairs, number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
    #        label_relation, size = number of event pairs

    def __init__(self):
        super(Event_Causal, self).__init__()

        self.token_embedding = Token_Embedding()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)

        self.causal_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.causal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_1.bias)

        self.causal_head_2 = nn.Linear(768, 256, bias=True)
        nn.init.xavier_uniform_(self.causal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_2.bias)

        self.causal_head_3 = nn.Linear(256, 3, bias=True)
        nn.init.xavier_uniform_(self.causal_head_3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_3.bias)

        self.relu = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none') # no reduction


    def forward(self, input_ids, attention_mask, label_event, event_pairs, label_causal):

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

        return causal_weighted_loss, causal_raw_scores

event_causal = Event_Causal()
event_causal.cuda()
event_causal.load_state_dict(torch.load('./saved_models/event_relation_graph/best_positive_F_causal.ckpt', map_location=device))
event_causal.eval()


class Event_Subevent(nn.Module):

    # input: label_event, number of tokens for event identification * 3 (start in input_ids, end in input_ids, label_event)
    #        event_pairs, number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
    #        label_relation, size = number of event pairs

    def __init__(self):
        super(Event_Subevent, self).__init__()

        self.token_embedding = Token_Embedding()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)

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


    def forward(self, input_ids, attention_mask, label_event, event_pairs, label_subevent):

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

        return subevent_weighted_loss, subevent_raw_scores

event_subevent = Event_Subevent()
event_subevent.cuda()
event_subevent.load_state_dict(torch.load('./saved_models/event_relation_graph/best_positive_F_subevent.ckpt', map_location=device))
event_subevent.eval()





''' build event relation graph on BASIL '''


in_file_path = "./BASIL"
file_names = os.listdir(in_file_path)

for file_idx in range(len(file_names)):

    print(file_names[file_idx])

    with open(in_file_path + "/" + file_names[file_idx], "r") as injson:
        original_article_json = json.load(injson)


    ''' tokenize the article '''

    article_json = {}
    article_json['uuid'] = original_article_json['uuid']
    article_json['triplet-uuid'] = original_article_json['triplet-uuid']
    article_json['media'] = original_article_json['media']
    article_json['date'] = original_article_json['date']
    article_json['url'] = original_article_json['url']
    article_json['main-entities'] = original_article_json['main-entities']
    article_json['main-event'] = original_article_json['main-event']

    article_json['sentences'] = []
    index_of_token = 0
    # deal with the title first
    sent_dict = {'sentence_id': 'title', 'sentence_text': original_article_json['title'] + '.', 'tokens': [], 'label_info_bias': -1, 'label_info_lex_bias': -1} # title do not have label
    tokens_list = nltk.tokenize.TreebankWordTokenizer().tokenize(sent_dict['sentence_text'])
    for token_i in range(len(tokens_list)):
        token_dict = {'index_of_token': index_of_token, 'token_text': tokens_list[token_i], 'prob_event': [0, 0], 'label_event': 0}
        sent_dict['tokens'].append(token_dict)
        index_of_token += 1
    article_json['sentences'].append(sent_dict)
    # then deal with sentences
    for sent_i in range(len(original_article_json['basil_sentence_level_annotations'])):
        sent_dict = {'sentence_id': original_article_json['basil_sentence_level_annotations'][sent_i]['sentence_id'],
                     'sentence_text': original_article_json['basil_sentence_level_annotations'][sent_i]['sentence_text'],
                     'tokens': []}
        tokens_list = nltk.tokenize.TreebankWordTokenizer().tokenize(sent_dict['sentence_text'])
        for token_i in range(len(tokens_list)):
            token_dict = {'index_of_token': index_of_token, 'token_text': tokens_list[token_i], 'prob_event': [0, 0], 'label_event': 0}
            sent_dict['tokens'].append(token_dict)
            index_of_token += 1

        label_info_bias = 0
        label_info_lex_bias = 0
        for ann_i in range(len(original_article_json['basil_sentence_level_annotations'][sent_i]['basil_ann'])):
            if original_article_json['basil_sentence_level_annotations'][sent_i]['basil_ann'][ann_i]['bias'] == 'inf':
                label_info_bias = 1
                label_info_lex_bias = 1
            if original_article_json['basil_sentence_level_annotations'][sent_i]['basil_ann'][ann_i]['bias'] == 'lex':
                label_info_lex_bias = 1

        sent_dict['label_info_bias'] = label_info_bias
        sent_dict['label_info_lex_bias'] = label_info_lex_bias

        article_json['sentences'].append(sent_dict)


    ''' event identification '''

    input_ids = []
    attention_mask = []

    label_event = []
    # label_event[i,:] = [start, end, label_event], where [start, end] is the corresponding index in input_ids of ith index of token
    # input_ids(range(start, end)) can extract the corresponding word

    stop_flag = 0

    input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])  # the <s> start token
    attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])

    for sent_i in range(len(article_json['sentences'])):
        if stop_flag == 1:
            break
        for token_i in range(len(article_json['sentences'][sent_i]['tokens'])):
            token_text = article_json['sentences'][sent_i]['tokens'][token_i]['token_text']
            start = len(input_ids)
            word_encoding = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)

            if start + len(word_encoding['input_ids']) < MAX_LEN: # truncate to MAX_LEN
                input_ids.extend(word_encoding['input_ids'])
                attention_mask.extend(word_encoding['attention_mask'])
                end = len(input_ids)
                label_event.append([start, end, article_json['sentences'][sent_i]['tokens'][token_i]['label_event']])
            else:
                stop_flag = 1
                break

    input_ids.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['input_ids'])  # the </s> end token
    attention_mask.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['attention_mask'])

    num_pad = MAX_LEN - len(input_ids)
    if num_pad > 0:
        input_ids.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['input_ids']) # the <pad> padding token
        attention_mask.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['attention_mask'])

    input_ids = torch.tensor(input_ids)  # size = length of tokens
    attention_mask = torch.tensor(attention_mask)
    label_event = torch.tensor(label_event)  # number of words * 3 (start in input_ids, end in input_ids, label_event)

    input_ids = input_ids.view(1, input_ids.shape[0])
    attention_mask = attention_mask.view(1, attention_mask.shape[0])

    input_ids, attention_mask, label_event = input_ids.to(device), attention_mask.to(device), label_event.to(device)

    with torch.no_grad():
        event_weighted_loss, event_raw_scores = event_identification(input_ids, attention_mask, label_event)

    event_predicted_prob = softmax(event_raw_scores)
    event_predicted_label = torch.argmax(event_predicted_prob, dim = 1)

    event_predicted_prob = event_predicted_prob.to('cpu').numpy()
    event_predicted_label = event_predicted_label.to('cpu').numpy()


    event_tokens = []
    stop_flag = 0
    for sent_i in range(len(article_json['sentences'])):
        if stop_flag == 1:
            break
        for token_i in range(len(article_json['sentences'][sent_i]['tokens'])):
            index_of_token = article_json['sentences'][sent_i]['tokens'][token_i]['index_of_token']
            if index_of_token < event_predicted_prob.shape[0]:
                article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][0] = event_predicted_prob[index_of_token][0].astype(np.float)
                article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][1] = event_predicted_prob[index_of_token][1].astype(np.float)

                if article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][0] < 0.5:
                    article_json['sentences'][sent_i]['tokens'][token_i]['label_event'] = 1
                    event_tokens.append(article_json['sentences'][sent_i]['tokens'][token_i])

            else:
                stop_flag = 1
                break

    article_json['event_tokens'] = event_tokens


    ''' form and initialize event pairs '''

    number_of_events = len(event_tokens)
    article_json['number_of_events'] = number_of_events

    if number_of_events > 1:

        relation_label = []
        for i in range(number_of_events):
            for j in range(i + 1, number_of_events):
                relation_label.append({'event_1': event_tokens[i], 'event_2': event_tokens[j],
                                       'prob_coreference': [0, 0], 'label_coreference': 0,
                                       'prob_temporal': [0, 0, 0, 0], 'label_temporal': 0,
                                       'prob_causal': [0, 0, 0], 'label_causal': 0,
                                       'prob_subevent': [0, 0, 0], 'label_subevent': 0,})

        article_json['relation_label'] = relation_label


        ''' event coreference relation '''

        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_coreference = []

        for event_pair_i in range(len(article_json['relation_label'])):
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_of_token'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_of_token']])

            label_coreference.append(article_json['relation_label'][event_pair_i]['label_coreference'])

        event_pairs = torch.tensor(event_pairs)  # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_coreference = torch.tensor(label_coreference)  # size = number of event pairs, corresponded to event_pairs

        input_ids, attention_mask, label_event, event_pairs, label_coreference = \
                    input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), label_coreference.to(device)

        with torch.no_grad():
            coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster = \
                event_coreference(input_ids, attention_mask, label_event, event_pairs, label_coreference)

        coreference_predicted_prob = softmax(coreference_raw_scores)
        coreference_predicted_label = torch.argmax(coreference_predicted_prob, dim = 1)

        coreference_predicted_prob = coreference_predicted_prob.to('cpu').numpy()
        coreference_predicted_label = coreference_predicted_label.to('cpu').numpy()

        for event_pair_i in range(len(article_json['relation_label'])):
            article_json['relation_label'][event_pair_i]['prob_coreference'][0] = coreference_predicted_prob[event_pair_i][0].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_coreference'][1] = coreference_predicted_prob[event_pair_i][1].astype(np.float)
            article_json['relation_label'][event_pair_i]['label_coreference'] = int(coreference_predicted_label[event_pair_i])


        ''' event temporal relation '''

        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_temporal = []

        for event_pair_i in range(len(article_json['relation_label'])):
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_of_token'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_of_token']])

            label_temporal.append(article_json['relation_label'][event_pair_i]['label_temporal'])

        event_pairs = torch.tensor(event_pairs)  # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_temporal = torch.tensor(label_temporal)  # size = number of event pairs, corresponded to event_pairs

        input_ids, attention_mask, label_event, event_pairs, label_temporal = \
            input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), label_temporal.to(device)

        with torch.no_grad():
            temporal_weighted_loss, temporal_raw_scores = event_temporal(input_ids, attention_mask, label_event, event_pairs, label_temporal)

        temporal_predicted_prob = softmax(temporal_raw_scores)
        temporal_predicted_label = torch.argmax(temporal_predicted_prob, dim = 1)

        temporal_predicted_prob = temporal_predicted_prob.to('cpu').numpy()
        temporal_predicted_label = temporal_predicted_label.to('cpu').numpy()

        for event_pair_i in range(len(article_json['relation_label'])):
            article_json['relation_label'][event_pair_i]['prob_temporal'][0] = temporal_predicted_prob[event_pair_i][0].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_temporal'][1] = temporal_predicted_prob[event_pair_i][1].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_temporal'][2] = temporal_predicted_prob[event_pair_i][2].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_temporal'][3] = temporal_predicted_prob[event_pair_i][3].astype(np.float)
            article_json['relation_label'][event_pair_i]['label_temporal'] = int(temporal_predicted_label[event_pair_i])


        ''' event causal relation '''

        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_causal = []

        for event_pair_i in range(len(article_json['relation_label'])):
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_of_token'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_of_token']])

            label_causal.append(article_json['relation_label'][event_pair_i]['label_causal'])

        event_pairs = torch.tensor(event_pairs)  # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_causal = torch.tensor(label_causal)  # size = number of event pairs, corresponded to event_pairs

        input_ids, attention_mask, label_event, event_pairs, label_causal = \
            input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), label_causal.to( device)

        with torch.no_grad():
            causal_weighted_loss, causal_raw_scores = event_causal(input_ids, attention_mask, label_event, event_pairs, label_causal)

        causal_predicted_prob = softmax(causal_raw_scores)
        causal_predicted_label = torch.argmax(causal_predicted_prob, dim = 1)

        causal_predicted_prob = causal_predicted_prob.to('cpu').numpy()
        causal_predicted_label = causal_predicted_label.to('cpu').numpy()

        for event_pair_i in range(len(article_json['relation_label'])):
            article_json['relation_label'][event_pair_i]['prob_causal'][0] = causal_predicted_prob[event_pair_i][0].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_causal'][1] = causal_predicted_prob[event_pair_i][1].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_causal'][2] = causal_predicted_prob[event_pair_i][2].astype(np.float)
            article_json['relation_label'][event_pair_i]['label_causal'] = int(causal_predicted_label[event_pair_i])


        ''' event subevent relation '''

        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_subevent = []

        for event_pair_i in range(len(article_json['relation_label'])):
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_of_token'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_of_token']])

            label_subevent.append(article_json['relation_label'][event_pair_i]['label_subevent'])

        event_pairs = torch.tensor(event_pairs)  # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_subevent = torch.tensor(label_subevent)  # size = number of event pairs, corresponded to event_pairs

        input_ids, attention_mask, label_event, event_pairs, label_subevent = \
            input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), label_subevent.to(device)

        with torch.no_grad():
            subevent_weighted_loss, subevent_raw_scores = event_subevent(input_ids, attention_mask, label_event, event_pairs, label_subevent)

        subevent_predicted_prob = softmax(subevent_raw_scores)
        subevent_predicted_label = torch.argmax(subevent_predicted_prob, dim = 1)

        subevent_predicted_prob = subevent_predicted_prob.to('cpu').numpy()
        subevent_predicted_label = subevent_predicted_label.to('cpu').numpy()

        for event_pair_i in range(len(article_json['relation_label'])):
            article_json['relation_label'][event_pair_i]['prob_subevent'][0] = subevent_predicted_prob[event_pair_i][0].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_subevent'][1] = subevent_predicted_prob[event_pair_i][1].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_subevent'][2] = subevent_predicted_prob[event_pair_i][2].astype(np.float)
            article_json['relation_label'][event_pair_i]['label_subevent'] = int(subevent_predicted_label[event_pair_i])


    ''' save article_json with event relation graph '''


    out_file_path = "./BASIL_event_graph"
    with open(out_file_path + "/" + file_names[file_idx][:-5] + "_event_graph.json", "w", encoding="utf-8") as outjson:
        json.dump(article_json, outjson)





''' build event relation graph on BiasedSents '''


in_file_path = "./BiasedSents"
file_names = os.listdir(in_file_path)

for file_idx in range(len(file_names)):

    print(file_names[file_idx])

    with open(in_file_path + "/" + file_names[file_idx], "r") as injson:
        original_article_json = json.load(injson)


    ''' tokenize the article '''

    article_json = {}
    article_json['event'] = original_article_json['event']
    article_json['date_event'] = original_article_json['date_event']
    article_json['id_article'] = original_article_json['id_article']
    article_json['source'] = original_article_json['source']
    article_json['source_bias'] = original_article_json['source_bias']
    article_json['url'] = original_article_json['url']

    article_json['sentences'] = []
    index_of_token = 0
    # deal with the ref_title first
    sent_dict = {'sentence_id': 'ref_title', 'sentence_text': original_article_json['ref_title'] + '.', 'tokens': [], 'label_bias': -1} # ref_title do not have label
    tokens_list = nltk.tokenize.TreebankWordTokenizer().tokenize(sent_dict['sentence_text'])
    for token_i in range(len(tokens_list)):
        token_dict = {'index_of_token': index_of_token, 'token_text': tokens_list[token_i], 'prob_event': [0, 0], 'label_event': 0}
        sent_dict['tokens'].append(token_dict)
        index_of_token += 1
    article_json['sentences'].append(sent_dict)
    # then deal with the title
    sent_dict = {'sentence_id': 0, 'sentence_text': original_article_json['title'] + '.', 'tokens': [], 'label_bias': original_article_json['ann_title']} # ref_title do not have label
    tokens_list = nltk.tokenize.TreebankWordTokenizer().tokenize(sent_dict['sentence_text'])
    for token_i in range(len(tokens_list)):
        token_dict = {'index_of_token': index_of_token, 'token_text': tokens_list[token_i], 'prob_event': [0, 0], 'label_event': 0}
        sent_dict['tokens'].append(token_dict)
        index_of_token += 1
    article_json['sentences'].append(sent_dict)
    # then deal with sentences
    for sent_i in range(len(original_article_json['body'])):
        sent_dict = {'sentence_id': original_article_json['body'][sent_i]['sentence_index'] + 1,
                     'sentence_text': original_article_json['body'][sent_i]['sentence'],
                     'tokens': [], 'label_bias': original_article_json['body'][sent_i]['ann']}
        tokens_list = nltk.tokenize.TreebankWordTokenizer().tokenize(sent_dict['sentence_text'])
        for token_i in range(len(tokens_list)):
            token_dict = {'index_of_token': index_of_token, 'token_text': tokens_list[token_i], 'prob_event': [0, 0], 'label_event': 0}
            sent_dict['tokens'].append(token_dict)
            index_of_token += 1
        article_json['sentences'].append(sent_dict)


    ''' event identification '''

    input_ids = []
    attention_mask = []

    label_event = []
    # label_event[i,:] = [start, end, label_event], where [start, end] is the corresponding index in input_ids of ith index of token
    # input_ids(range(start, end)) can extract the corresponding word

    stop_flag = 0

    input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])  # the <s> start token
    attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])

    for sent_i in range(len(article_json['sentences'])):
        if stop_flag == 1:
            break
        for token_i in range(len(article_json['sentences'][sent_i]['tokens'])):
            token_text = article_json['sentences'][sent_i]['tokens'][token_i]['token_text']
            start = len(input_ids)
            word_encoding = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)

            if start + len(word_encoding['input_ids']) < MAX_LEN: # truncate to MAX_LEN
                input_ids.extend(word_encoding['input_ids'])
                attention_mask.extend(word_encoding['attention_mask'])
                end = len(input_ids)
                label_event.append([start, end, article_json['sentences'][sent_i]['tokens'][token_i]['label_event']])
            else:
                stop_flag = 1
                break

    input_ids.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['input_ids'])  # the </s> end token
    attention_mask.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['attention_mask'])

    num_pad = MAX_LEN - len(input_ids)
    if num_pad > 0:
        input_ids.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['input_ids']) # the <pad> padding token
        attention_mask.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['attention_mask'])

    input_ids = torch.tensor(input_ids)  # size = length of tokens
    attention_mask = torch.tensor(attention_mask)
    label_event = torch.tensor(label_event)  # number of words * 3 (start in input_ids, end in input_ids, label_event)

    input_ids = input_ids.view(1, input_ids.shape[0])
    attention_mask = attention_mask.view(1, attention_mask.shape[0])

    input_ids, attention_mask, label_event = input_ids.to(device), attention_mask.to(device), label_event.to(device)

    with torch.no_grad():
        event_weighted_loss, event_raw_scores = event_identification(input_ids, attention_mask, label_event)

    event_predicted_prob = softmax(event_raw_scores)
    event_predicted_label = torch.argmax(event_predicted_prob, dim = 1)

    event_predicted_prob = event_predicted_prob.to('cpu').numpy()
    event_predicted_label = event_predicted_label.to('cpu').numpy()


    event_tokens = []
    stop_flag = 0
    for sent_i in range(len(article_json['sentences'])):
        if stop_flag == 1:
            break
        for token_i in range(len(article_json['sentences'][sent_i]['tokens'])):
            index_of_token = article_json['sentences'][sent_i]['tokens'][token_i]['index_of_token']
            if index_of_token < event_predicted_prob.shape[0]:
                article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][0] = event_predicted_prob[index_of_token][0].astype(np.float)
                article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][1] = event_predicted_prob[index_of_token][1].astype(np.float)

                if article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][0] < 0.5:
                    article_json['sentences'][sent_i]['tokens'][token_i]['label_event'] = 1
                    event_tokens.append(article_json['sentences'][sent_i]['tokens'][token_i])

            else:
                stop_flag = 1
                break

    article_json['event_tokens'] = event_tokens


    ''' form and initialize event pairs '''

    number_of_events = len(event_tokens)
    article_json['number_of_events'] = number_of_events

    if number_of_events > 1:

        relation_label = []
        for i in range(number_of_events):
            for j in range(i + 1, number_of_events):
                relation_label.append({'event_1': event_tokens[i], 'event_2': event_tokens[j],
                                       'prob_coreference': [0, 0], 'label_coreference': 0,
                                       'prob_temporal': [0, 0, 0, 0], 'label_temporal': 0,
                                       'prob_causal': [0, 0, 0], 'label_causal': 0,
                                       'prob_subevent': [0, 0, 0], 'label_subevent': 0,})

        article_json['relation_label'] = relation_label


        ''' event coreference relation '''

        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_coreference = []

        for event_pair_i in range(len(article_json['relation_label'])):
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_of_token'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_of_token']])

            label_coreference.append(article_json['relation_label'][event_pair_i]['label_coreference'])

        event_pairs = torch.tensor(event_pairs)  # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_coreference = torch.tensor(label_coreference)  # size = number of event pairs, corresponded to event_pairs

        input_ids, attention_mask, label_event, event_pairs, label_coreference = \
                    input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), label_coreference.to(device)

        with torch.no_grad():
            coreference_weighted_loss, coreference_raw_scores, predicted_coreference_cluster, label_coreference_cluster = \
                event_coreference(input_ids, attention_mask, label_event, event_pairs, label_coreference)

        coreference_predicted_prob = softmax(coreference_raw_scores)
        coreference_predicted_label = torch.argmax(coreference_predicted_prob, dim = 1)

        coreference_predicted_prob = coreference_predicted_prob.to('cpu').numpy()
        coreference_predicted_label = coreference_predicted_label.to('cpu').numpy()

        for event_pair_i in range(len(article_json['relation_label'])):
            article_json['relation_label'][event_pair_i]['prob_coreference'][0] = coreference_predicted_prob[event_pair_i][0].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_coreference'][1] = coreference_predicted_prob[event_pair_i][1].astype(np.float)
            article_json['relation_label'][event_pair_i]['label_coreference'] = int(coreference_predicted_label[event_pair_i])


        ''' event temporal relation '''

        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_temporal = []

        for event_pair_i in range(len(article_json['relation_label'])):
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_of_token'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_of_token']])

            label_temporal.append(article_json['relation_label'][event_pair_i]['label_temporal'])

        event_pairs = torch.tensor(event_pairs)  # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_temporal = torch.tensor(label_temporal)  # size = number of event pairs, corresponded to event_pairs

        input_ids, attention_mask, label_event, event_pairs, label_temporal = \
            input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), label_temporal.to(device)

        with torch.no_grad():
            temporal_weighted_loss, temporal_raw_scores = event_temporal(input_ids, attention_mask, label_event, event_pairs, label_temporal)

        temporal_predicted_prob = softmax(temporal_raw_scores)
        temporal_predicted_label = torch.argmax(temporal_predicted_prob, dim = 1)

        temporal_predicted_prob = temporal_predicted_prob.to('cpu').numpy()
        temporal_predicted_label = temporal_predicted_label.to('cpu').numpy()

        for event_pair_i in range(len(article_json['relation_label'])):
            article_json['relation_label'][event_pair_i]['prob_temporal'][0] = temporal_predicted_prob[event_pair_i][0].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_temporal'][1] = temporal_predicted_prob[event_pair_i][1].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_temporal'][2] = temporal_predicted_prob[event_pair_i][2].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_temporal'][3] = temporal_predicted_prob[event_pair_i][3].astype(np.float)
            article_json['relation_label'][event_pair_i]['label_temporal'] = int(temporal_predicted_label[event_pair_i])


        ''' event causal relation '''

        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_causal = []

        for event_pair_i in range(len(article_json['relation_label'])):
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_of_token'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_of_token']])

            label_causal.append(article_json['relation_label'][event_pair_i]['label_causal'])

        event_pairs = torch.tensor(event_pairs)  # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_causal = torch.tensor(label_causal)  # size = number of event pairs, corresponded to event_pairs

        input_ids, attention_mask, label_event, event_pairs, label_causal = \
            input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), label_causal.to( device)

        with torch.no_grad():
            causal_weighted_loss, causal_raw_scores = event_causal(input_ids, attention_mask, label_event, event_pairs, label_causal)

        causal_predicted_prob = softmax(causal_raw_scores)
        causal_predicted_label = torch.argmax(causal_predicted_prob, dim = 1)

        causal_predicted_prob = causal_predicted_prob.to('cpu').numpy()
        causal_predicted_label = causal_predicted_label.to('cpu').numpy()

        for event_pair_i in range(len(article_json['relation_label'])):
            article_json['relation_label'][event_pair_i]['prob_causal'][0] = causal_predicted_prob[event_pair_i][0].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_causal'][1] = causal_predicted_prob[event_pair_i][1].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_causal'][2] = causal_predicted_prob[event_pair_i][2].astype(np.float)
            article_json['relation_label'][event_pair_i]['label_causal'] = int(causal_predicted_label[event_pair_i])


        ''' event subevent relation '''

        event_pairs = []
        # event_pairs[i, :] = [event 1 row in label_event, event 2 row in label_event], ith event pair
        label_subevent = []

        for event_pair_i in range(len(article_json['relation_label'])):
            event_pairs.append([article_json['relation_label'][event_pair_i]['event_1']['index_of_token'],
                                article_json['relation_label'][event_pair_i]['event_2']['index_of_token']])

            label_subevent.append(article_json['relation_label'][event_pair_i]['label_subevent'])

        event_pairs = torch.tensor(event_pairs)  # number of event pairs * 2 (event 1 row in label_event, event 2 row in label_event)
        label_subevent = torch.tensor(label_subevent)  # size = number of event pairs, corresponded to event_pairs

        input_ids, attention_mask, label_event, event_pairs, label_subevent = \
            input_ids.to(device), attention_mask.to(device), label_event.to(device), event_pairs.to(device), label_subevent.to(device)

        with torch.no_grad():
            subevent_weighted_loss, subevent_raw_scores = event_subevent(input_ids, attention_mask, label_event, event_pairs, label_subevent)

        subevent_predicted_prob = softmax(subevent_raw_scores)
        subevent_predicted_label = torch.argmax(subevent_predicted_prob, dim = 1)

        subevent_predicted_prob = subevent_predicted_prob.to('cpu').numpy()
        subevent_predicted_label = subevent_predicted_label.to('cpu').numpy()

        for event_pair_i in range(len(article_json['relation_label'])):
            article_json['relation_label'][event_pair_i]['prob_subevent'][0] = subevent_predicted_prob[event_pair_i][0].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_subevent'][1] = subevent_predicted_prob[event_pair_i][1].astype(np.float)
            article_json['relation_label'][event_pair_i]['prob_subevent'][2] = subevent_predicted_prob[event_pair_i][2].astype(np.float)
            article_json['relation_label'][event_pair_i]['label_subevent'] = int(subevent_predicted_label[event_pair_i])


    ''' save article_json with event relation graph '''


    out_file_path = "./BiasedSents_event_graph"
    with open(out_file_path + "/" + file_names[file_idx][:-5] + "_event_graph.json", "w", encoding="utf-8") as outjson:
        json.dump(article_json, outjson)











# stop here
