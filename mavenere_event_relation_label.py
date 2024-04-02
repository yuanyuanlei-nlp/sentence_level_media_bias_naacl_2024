
import random
import json
import os



''' extract event labels and event relations labels from original json data'''

def extract_label(line_json, out_path):

    article_json = {}
    article_json['article_id'] = line_json['id']
    article_json['title'] = line_json['title']
    article_json['article_txt'] = " ".join(line_json['sentences'])

    # tokens_list

    article_json['tokens_list'] = []
    num_tokens_in_sentence = []
    prefix_sum_num_tokens = [0]
    sum = 0

    for sent_i in range(len(line_json['tokens'])):
        num_tokens_in_sentence.append(len(line_json['tokens'][sent_i]))
        sum += len(line_json['tokens'][sent_i])
        prefix_sum_num_tokens.append(sum)
        article_json['tokens_list'].extend(line_json['tokens'][sent_i])


    # event_mentions, list of dict with keys: event_id, mention_id, trigger_word, index_in_tokens_list, index_in_event_label

    article_json['event_mentions'] = []
    for event_i in range(len(line_json['events'])):
        for mention_i in range(len(line_json['events'][event_i]['mention'])):
            event_dict = {}
            event_dict['event_id'] = line_json['events'][event_i]['id']
            event_dict['mention_id'] = line_json['events'][event_i]['mention'][mention_i]['id']
            event_dict['trigger_word'] = line_json['events'][event_i]['mention'][mention_i]['trigger_word']
            index_in_tokens_list = []
            for token_i in range(line_json['events'][event_i]['mention'][mention_i]['offset'][0], line_json['events'][event_i]['mention'][mention_i]['offset'][1]):
                index_in_tokens_list.append(prefix_sum_num_tokens[line_json['events'][event_i]['mention'][mention_i]['sent_id']] + token_i)

            trigger_word = []
            for i in range(len(index_in_tokens_list)):
                trigger_word.append(article_json['tokens_list'][index_in_tokens_list[i]])
            if " ".join(trigger_word) != event_dict['trigger_word']:
                print("event trigger word not match")

            event_dict['index_in_tokens_list'] = index_in_tokens_list
            article_json['event_mentions'].append(event_dict)

    article_json['event_mentions'] = sorted(article_json['event_mentions'], key = lambda d: d['index_in_tokens_list'][0]) # sorted in natural textual order


    # event_label, list of dict with keys: token, index_in_tokens_list, event_label, index_in_event_label

    article_json['event_label'] = []
    point_token = 0 # point to article_json['tokens_list']
    point_event = 0 # point to article_json['event_mentions']
    point_event_label = 0 # point to article_json['event_label']

    while point_token < len(article_json['tokens_list']) and point_event <= len(article_json['event_mentions']):
        if point_event < len(article_json['event_mentions']):
            if point_token < article_json['event_mentions'][point_event]['index_in_tokens_list'][0]:
                token_dict = {}
                token_dict['token'] = article_json['tokens_list'][point_token]
                token_dict['index_in_tokens_list'] = [point_token] # index in article_json['tokens_list']
                token_dict['event_label'] = 0 # not event
                token_dict['index_in_event_label'] = point_event_label # index in article_json['event_label']
                article_json['event_label'].append(token_dict)
                point_token += 1
                point_event_label += 1
            else:
                token_dict = {}
                token_dict['token'] = article_json['event_mentions'][point_event]['trigger_word']
                token_dict['index_in_tokens_list'] = article_json['event_mentions'][point_event]['index_in_tokens_list']
                token_dict['event_label'] = 1 # event
                token_dict['index_in_event_label'] = point_event_label
                article_json['event_label'].append(token_dict)
                article_json['event_mentions'][point_event]['index_in_event_label'] = point_event_label
                point_token = article_json['event_mentions'][point_event]['index_in_tokens_list'][-1] + 1
                point_event += 1
                point_event_label += 1
        else: # all event words have been matched
            token_dict = {}
            token_dict['token'] = article_json['tokens_list'][point_token]
            token_dict['index_in_tokens_list'] = [point_token]
            token_dict['event_label'] = 0  # not event
            token_dict['index_in_event_label'] = point_event_label
            article_json['event_label'].append(token_dict)
            point_token += 1
            point_event_label += 1



    # relation_label, list of dict with keys: event_1, event_2, label_coreference, label_temporal, label_causal, label_subevent

    article_json['relation_label'] = []
    num_events = len(article_json['event_mentions']) # number of event mentions

    for i in range(0, num_events - 1):
        for j in range(i + 1, num_events):
            event_pair = {}
            event_pair['event_1'] = article_json['event_mentions'][i]
            event_pair['event_2'] = article_json['event_mentions'][j]

            event_pair['label_coreference'] = 0 # by default is 0, no such relation
            event_pair['label_temporal'] = 0
            event_pair['label_causal'] = 0
            event_pair['label_subevent'] = 0

            if event_pair['event_1']['event_id'] == event_pair['event_2']['event_id']:
                event_pair['label_coreference'] = 1

            article_json['relation_label'].append(event_pair)

    if len(article_json['relation_label']) != (1 + num_events - 1) * (num_events - 1) // 2:
        print("wrong number of event pairs")


    # temporal relation
    for relation_i in range(len(line_json['temporal_relations']['BEFORE'])):
        relation_1 = line_json['temporal_relations']['BEFORE'][relation_i][0] # event_id
        relation_2 = line_json['temporal_relations']['BEFORE'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal BEFORE relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 1 # event_1 before event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal BEFORE relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 2 # event_1 after event_2



    for relation_i in range(len(line_json['temporal_relations']['OVERLAP'])):
        relation_1 = line_json['temporal_relations']['OVERLAP'][relation_i][0] # event_id
        relation_2 = line_json['temporal_relations']['OVERLAP'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal OVERLAP relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal OVERLAP relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2



    for relation_i in range(len(line_json['temporal_relations']['CONTAINS'])):
        relation_1 = line_json['temporal_relations']['CONTAINS'][relation_i][0] # event_id
        relation_2 = line_json['temporal_relations']['CONTAINS'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal CONTAINS relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal CONTAINS relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2



    for relation_i in range(len(line_json['temporal_relations']['SIMULTANEOUS'])):
        relation_1 = line_json['temporal_relations']['SIMULTANEOUS'][relation_i][0] # event_id
        relation_2 = line_json['temporal_relations']['SIMULTANEOUS'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal SIMULTANEOUS relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal SIMULTANEOUS relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2



    for relation_i in range(len(line_json['temporal_relations']['ENDS-ON'])):
        relation_1 = line_json['temporal_relations']['ENDS-ON'][relation_i][0] # event_id
        relation_2 = line_json['temporal_relations']['ENDS-ON'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal ENDS-ON relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal ENDS-ON relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2



    for relation_i in range(len(line_json['temporal_relations']['BEGINS-ON'])):
        relation_1 = line_json['temporal_relations']['BEGINS-ON'][relation_i][0] # event_id
        relation_2 = line_json['temporal_relations']['BEGINS-ON'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in temporal BEGINS-ON relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in temporal BEGINS-ON relation")

                        article_json['relation_label'][index_in_relation_label]['label_temporal'] = 3 # event_1 overlap event_2



    # causal relation
    for relation_i in range(len(line_json['causal_relations']['CAUSE'])):
        relation_1 = line_json['causal_relations']['CAUSE'][relation_i][0] # event_id
        relation_2 = line_json['causal_relations']['CAUSE'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in causal CAUSE relation")

                        article_json['relation_label'][index_in_relation_label]['label_causal'] = 1 # event_1 causes event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in causal CAUSE relation")

                        article_json['relation_label'][index_in_relation_label]['label_causal'] = 2 # event_1 caused by event_2



    for relation_i in range(len(line_json['causal_relations']['PRECONDITION'])):
        relation_1 = line_json['causal_relations']['PRECONDITION'][relation_i][0] # event_id
        relation_2 = line_json['causal_relations']['PRECONDITION'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in causal PRECONDITION relation")

                        article_json['relation_label'][index_in_relation_label]['label_causal'] = 1 # event_1 causes event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in causal PRECONDITION relation")

                        article_json['relation_label'][index_in_relation_label]['label_causal'] = 2 # event_1 caused by event_2




    # subevent relations
    for relation_i in range(len(line_json['subevent_relations'])):
        relation_1 = line_json['subevent_relations'][relation_i][0] # event_id
        relation_2 = line_json['subevent_relations'][relation_i][1]

        if relation_1[:5] == 'EVENT' and relation_2[:5] == 'EVENT': # exclude relation between Time expressions

            relation_1_event = []
            relation_1_index_in_event_mentions = []
            relation_2_event = []
            relation_2_index_in_event_mentions = []
            for event_i in range(len(article_json['event_mentions'])):
                if article_json['event_mentions'][event_i]['event_id'] == relation_1:
                    relation_1_event.append(article_json['event_mentions'][event_i])
                    relation_1_index_in_event_mentions.append(event_i)
                if article_json['event_mentions'][event_i]['event_id'] == relation_2:
                    relation_2_event.append(article_json['event_mentions'][event_i])
                    relation_2_index_in_event_mentions.append(event_i)

            for i in range(len(relation_1_event)):
                for j in range(len(relation_2_event)):
                    if relation_1_event[i]['index_in_event_label'] < relation_2_event[j]['index_in_event_label']:
                        min = relation_1_index_in_event_mentions[i]
                        max = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_1_event[i] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_2_event[j]:
                            print("unmatched event pair in subevent relation")

                        article_json['relation_label'][index_in_relation_label]['label_subevent'] = 1 # event_1 contains event_2

                    else:
                        max = relation_1_index_in_event_mentions[i]
                        min = relation_2_index_in_event_mentions[j]
                        index_in_relation_label = ((num_events - 1 + num_events - min) * min // 2) - 1 + (max - min)

                        if article_json['relation_label'][index_in_relation_label]['event_1'] != relation_2_event[j] or article_json['relation_label'][index_in_relation_label]['event_2'] != relation_1_event[i]:
                            print("unmatched event pair in subevent relation")

                        article_json['relation_label'][index_in_relation_label]['label_subevent'] = 2 # event_1 contained by event_2


    with open(out_path, "w") as f:
        json.dump(article_json, f)

# end of def extract_label




''' write json with event label and event relation label into new json file '''


line_id = 0
with open("./MAVEN_ERE/valid.jsonl", "r") as f:
    while line_id < 710:
        line = f.readline()
        line_json = json.loads(line)
        out_path = "./MAVEN_ERE/dev/" + line_json['id'] + ".json"
        extract_label(line_json, out_path)
        line_id += 1


line_id = 0
with open("./MAVEN_ERE/train.jsonl", "r") as f:
    while line_id < 2913:
        print(line_id)
        line = f.readline()
        line_json = json.loads(line)
        out_path = "./MAVEN_ERE/train/" + line_json['id'] + ".json"
        extract_label(line_json, out_path)
        line_id += 1







# stop here
