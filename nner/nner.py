import torch
import torch.nn.functional as F
from torch import nn

from transformers import *
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
import json

class TaggingEval(object):
    def __init__(self, ref='dev'):
        self.golds = []
        self.preds = []

        self.prepare_golds(ref)

    def prepare_golds(self, ref):
        # load json file
        with open(ref, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # extract golden spans + type
        for r in data:
            tokens = r['words']
            ents = r['golden-entity-mentions']

            tmp = []

            for ent in ents:
                s = ent['start']
                f = ent['end']

                ent_type = ent['entity-type'].split(':')[0]
                if ent_type not in ['Contact-Info', 'Crime', 'Job-Title', 'Numeric', 'Sentence', 'TIM']:
                    tmp.append((s, f-1, ent_type))

            self.golds.append(tmp)

    def add(self, pred):
        self.preds.append(pred)

    def get_span_labels(self, sentence_tags):
        span_labels = []
        tmp_dic = {} # ent_type: {start_pos1, start_pos2,...}
                    # when find 'O', clean the tmp_dic

        for idx, tag in enumerate(sentence_tags):

            tags = tag.split('|')

            if tags == ['O']:
                # clean the tmp_dic
                tmp_dic = {}
            else:

                tags = set(tags) # remove duplicate, from golden-label. useless for inference
                i_list = [] # create a list to collect all I- type
                for t in tags:

                    pos, tp = t.split('-')

                    # pos: B, I, U, L
                    if pos == 'U':
                        # unit length entity
                        span_labels.append((idx, idx, tp))
                    elif pos == 'L':
                        # end of entity, collect all head to this stop
                        # use tp as key

                        try:
                            starting_positions = tmp_dic[tp]

                            for sp in starting_positions:
                                span_labels.append((sp, idx, tp))
                        except:
                            # cant find head, wrong inference
                            pass
                    elif pos == 'B':
                        if tp in tmp_dic:
                            # if there are head already
                            tmp_dic[tp].append(idx)
                        else:
                            # initialize the head position
                            tmp_dic[tp] = [idx]

                    else:
                        i_list.append(tp)

                # pos == 'I'
                # make sure every ent type in the tmp_dic has I in this token,
                # O/W, the sequence broken
                # remove this sequence | add it to span_labels then remove
                remove_k = []
                for k, v in tmp_dic.items():
                    if k in i_list:
                        pass
                    else:
                        # check if this head was just added
                        if tmp_dic[k][0] != idx:
                        # add to label, and then remove
                        # for start_p in v:
                        #     span_labels.append((start_p, idx, k))

                        # remove, since it breaks
                            remove_k.append(k)
                for rk in remove_k:
                    tmp_dic.pop(rk, None)

        return span_labels

    def get_results(self):
        n_correct, n_predicted, n_gold = 0, 0, 0
        assert len(self.golds) == len(self.preds)

        for gold, pred in zip(self.golds, self.preds):
            gold_spans = set(gold)

            pred_spans = set(self.get_span_labels(pred))


            #print('GOLD:', gold_spans)
            #print('PRED:', pred_spans)

            n_correct += len(gold_spans & pred_spans)
            n_gold += len(gold_spans)
            n_predicted += len(pred_spans)

        if n_correct == 0:
            p, r, f1 = 0, 0, 0
        else:
            p = 100.0 * n_correct / n_predicted
            r = 100.0 * n_correct / n_gold
            f1 = 2 * p * r / (p + r)

        return p, r, f1


class Example(object):
    def __init__(self, text, label=None):
        self.text = text
        self.label = label


class Features(object):
    def __init__(self, input_ids, input_mask, segment_ids,
                 label_id, valid_ids=None, label_mask=None, sen_id=None,
                 eval_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.sen_id = sen_id
        self.eval_ids = eval_ids


class ACEProcessor(object):
    '''Processor for ACE2005 data set.'''

    def get_examples(self, data_dir):
        tsv = self.read_tsv(data_dir)
        examples = self.create_examples(tsv)
        return examples

    def get_labels(self):
        return ["O",
                "B-PER", "I-PER", "L-PER", "U-PER",
                "B-ORG", "I-ORG", "L-ORG", "U-ORG",
                "B-GPE", "I-GPE", "L-GPE", "U-GPE",
                "B-LOC", "I-LOC", "L-LOC", "U-LOC",
                "B-FAC", "I-FAC", "L-FAC", "U-FAC",
                "B-VEH", "I-VEH", "L-VEH", "U-VEH",
                "B-WEA", "I-WEA", "L-WEA", "U-WEA"]

    def create_examples(self, lines):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            examples.append(Example(text=sentence, label=label))
        return examples

    def read_tsv(self, filename):
        '''
        read file
        '''
        print('Reading file: ', filename)
        f = open(filename, encoding='utf-8')
        data = []

        for line in f:
            sentence, label = line.strip().split('\t')
            data.append((sentence, label))
        print('Data size: ', len(data))
        return data


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, n_duplicate):
    '''
    sentence: ['I', 'like', 'apples', 'juice']
    labels: ['O', 'O', 'A-B', 'A-I']
    tokens: ['I', 'like', 'app', '##les', 'ju', '##ice']
    label id: [0, 0, 1, 2] # label id still the same as labels
    valid: [1, 1, 1, 0, 1, 0] # only first token in a word matters for prediction
    ========================================================================
    input tokens: ['[CLS]', 'I', 'like', 'app', '##les', 'ju', '##ice', '[SEP]']
    input padding: ['[CLS]', 'I', 'like', 'app', '##les', 'ju', '##ice', '[SEP]', '[PAD]', '[PAD]']
    input masking: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] # [CLS] and [SEP] is important
    valid: [0, 1, 1, 1, 0, 1, 0, 0, 0, 0] # get tokens for prediction
    label padding: [{'O', 'O', 'A-B', 'A-I'},['O', 'O', 'O', 'O', 'O', 'O']] # use O as padding in []
    # we have labels in {}
    label masking: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    ==================================
    in prediction
    valid_logits: ['I', 'like', 'app',  'ju', 0, 0, 0, 0, 0, 0]
    label padding: ['O', 'O', 'A-B', 'A-I', 'O', 'O', 'O', 'O', 'O', 'O']
    --> get cross entropy loss
    --> apply label masking
    --> gradient on
    gradient: ['I', 'like', 'app',  'ju'] vs ['O', 'O', 'A-B', 'A-I']
    ===================================
    '''
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (sen_id, example) in enumerate(examples):
        text_list = example.text.split(' ')
        label_list = example.label.split(' ')
        total_tokens = []
        total_labels = []
        total_valid = []

        for i, word in enumerate(text_list):
            token = tokenizer.tokenize(word)
            total_tokens.extend(token)
            label_tmp = label_list[i]
            # label only applied to the first token in a word
            # e.g., lamb --> la ##mb
            #       valid_id = [1, 0]
            # so you only need hidden vector of la to make prediction on label on lamb
            for m in range(len(token)):
                if m == 0:
                    total_labels.append(label_tmp)
                    total_valid.append(1)

                else:
                    total_labels.append('*')  # special padding, will remove it after split long sentences
                    total_valid.append(0)

        # trunct sentence to max length
        tokens_split = []
        labels_split = []
        valid_split = []
        start = 0
        end = max_seq_length - 2
        while end < len(total_tokens):
            tokens_split.append(total_tokens[start: end])
            labels_split.append(total_labels[start: end])
            valid_split.append(total_valid[start: end])

            start += max_seq_length - 2 - n_duplicate
            end += max_seq_length - 2 - n_duplicate

        tokens_split.append(total_tokens[start:])
        labels_split.append(total_labels[start:])
        valid_split.append(total_valid[start:])

        for split_idx in range(len(tokens_split)):
            tokens = tokens_split[split_idx]
            labels = labels_split[split_idx]
            eval_idx = [1 for _ in range(len(labels))]

            # for next split, the first n_duplicate token are ignored
            if split_idx != 0:
                for eval_i in range(n_duplicate):
                    eval_idx[eval_i] = 0

            # clean label
            clean_labels = []
            clean_eval_idx = []
            for idx, l in enumerate(labels):
                if l != '*':  # remove '*"
                    clean_labels.append(l)
                    clean_eval_idx.append(eval_idx[idx])

            labels = clean_labels
            valid = valid_split[split_idx]
            # add special tokens
            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append(tokenizer.bos_token)
            segment_ids.append(0)
            valid.insert(0, 0)  # [CLS] is not valid for prediction

            for token in tokens:
                ntokens.append(token)
                segment_ids.append(0)

            '''
            assigning multiple labels
            '''
            for label_i in labels:
                # convert to one-hot encoding
                tmp_lab = [0] * len(label_map)
                labels_i = label_i.split('|')
                for l_i in labels_i:
                    l_idx = label_map[l_i]
                    tmp_lab[l_idx] = 1

                label_ids.append(tmp_lab)

            ntokens.append(tokenizer.eos_token)
            segment_ids.append(0)
            valid.append(0)  # [SEP] is not valid for prediction
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)  # label_ids stays the same, no change

            # padding inputs
            while len(input_ids) < max_seq_length:
                input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append([0] * len(label_map))
                valid.append(0)  # padding is not taken into consideration of prediction
                label_mask.append(0)
                clean_eval_idx.append(0)

            # padding labels
            while len(label_ids) < max_seq_length:
                label_ids.append([0] * len(label_map))
                label_mask.append(0)
                clean_eval_idx.append(0)

            sen_id_list = [sen_id] * max_seq_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length
            assert len(clean_eval_idx) == max_seq_length

            if sen_id < 0:
                print("============ Example ============")
                print("guid: %s" % (example.guid))
                print("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            features.append(
                Features(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_ids,
                         valid_ids=valid,
                         label_mask=label_mask,
                         sen_id=sen_id_list,
                         eval_ids=clean_eval_idx))
    return features


def create_dataloader(data_dir, set_type='train', batchsize=32, max_seq_length=128, tokenizer=None,
                      num_duplicate=None, num_exp=0):
    proc = ACEProcessor()

    data_examples = proc.get_examples(data_dir)

    if num_exp != 0:
        random.shuffle(data_examples)
        length = int(len(data_examples) * num_exp)
        data_examples = data_examples[:length]

    data_size = len(data_examples)

    print('Dataset set: %s, orig size: %s' % (set_type, data_size))
    data_size_total = data_size

    data_features = convert_examples_to_features(data_examples, proc.get_labels(),
                                                 max_seq_length, tokenizer, num_duplicate)
    all_input_ids = torch.tensor([f.input_ids for f in data_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in data_features], dtype=torch.long, requires_grad=False)
    all_segment_ids = torch.tensor([f.segment_ids for f in data_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in data_features], dtype=torch.float32)
    all_valid_ids = torch.tensor([f.valid_ids for f in data_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in data_features], dtype=torch.long)
    all_sen_ids = torch.tensor([f.sen_id for f in data_features], dtype=torch.long)
    all_eval_ids = torch.tensor([f.eval_ids for f in data_features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                            all_lmask_ids, all_eval_ids, all_sen_ids)
    if set_type == 'train':
        data_sampler = RandomSampler(dataset)
    else:
        data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batchsize)

    return dataloader, data_size_total


class BertForNER(BertForTokenClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, label_mask=None):
        # get bert output
        output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # last_hidden_states, pooler_output, hidden_states
        sequence_output = output[0]
        # valid output: get all valid hidden vectors
        # e.g., lamb --> la ##mb
        # we only get hidden vector of la
        valid_output = self.valid_word_output(sequence_output, valid_ids)
        # valid_output = [1,1,1,1,0,0,0,0,0,0,0] vs label [1,1,1,1,0,0,0,0,0,0,0] match valid

        # apply dropout and get logits on each time steps
        # label <-> input is 1 to 1 mapping now with some padding, mask out padding later
        sequence_output = self.dropout(valid_output)
        # run through dnn layer to get logits
        logits = self.classifier(sequence_output)
        # calculate loss
        loss = F.binary_cross_entropy(torch.sigmoid(logits), labels, reduction='none')
        # change dtype to float
        input_mask = label_mask.float()
        # apply mask on time domain so padding on label has loss = 0
        loss = loss.mean(dim=-1) * input_mask
        # get avg loss for each example
        loss_batch = loss.mean(-1)
        # get avg loss for minibatch
        avg_loss = loss_batch.mean()
        return avg_loss, logits

    def valid_word_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=self.device)

        # valid output: get all valid hidden vectors
        # e.g., lamb --> la ##mb
        # we only get hidden vector of la
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        return valid_output

    def set_label_map(self, label_list):
        label_map = {i: label for i, label in enumerate(label_list)}
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device

class XLMRobertaForNER(XLMRobertaForTokenClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, label_mask=None):
        # get bert output
        output = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # last_hidden_states, pooler_output, hidden_states
        sequence_output = output[0]
        # valid output: get all valid hidden vectors
        # e.g., lamb --> la ##mb
        # we only get hidden vector of la
        valid_output = self.valid_word_output(sequence_output, valid_ids)
        # valid_output = [1,1,1,1,0,0,0,0,0,0,0] vs label [1,1,1,1,0,0,0,0,0,0,0] match valid

        # apply dropout and get logits on each time steps
        # label <-> input is 1 to 1 mapping now with some padding, mask out padding later
        sequence_output = self.dropout(valid_output)
        # run through dnn layer to get logits
        logits = self.classifier(sequence_output)
        # calculate loss
        loss = F.binary_cross_entropy(torch.sigmoid(logits), labels, reduction='none')
        # change dtype to float
        input_mask = label_mask.float()
        # apply mask on time domain so padding on label has loss = 0
        loss = loss.mean(dim=-1) * input_mask
        # get avg loss for each example
        loss_batch = loss.mean(-1)
        # get avg loss for minibatch
        avg_loss = loss_batch.mean()
        return avg_loss, logits

    def valid_word_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=self.device)

        # valid output: get all valid hidden vectors
        # e.g., lamb --> la ##mb
        # we only get hidden vector of la
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        return valid_output

    def set_label_map(self, label_list):
        label_map = {i: label for i, label in enumerate(label_list)}
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device

class XLMForNER(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLMModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, label_mask=None):
        # get bert output
        output = self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # last_hidden_states, pooler_output, hidden_states
        sequence_output = output[0]
        # valid output: get all valid hidden vectors
        # e.g., lamb --> la ##mb
        # we only get hidden vector of la
        valid_output = self.valid_word_output(sequence_output, valid_ids)
        # valid_output = [1,1,1,1,0,0,0,0,0,0,0] vs label [1,1,1,1,0,0,0,0,0,0,0] match valid

        # apply dropout and get logits on each time steps
        # label <-> input is 1 to 1 mapping now with some padding, mask out padding later
        sequence_output = self.dropout(valid_output)
        # run through dnn layer to get logits
        logits = self.classifier(sequence_output)
        # calculate loss
        loss = F.binary_cross_entropy(torch.sigmoid(logits), labels, reduction='none')
        # change dtype to float
        input_mask = label_mask.float()
        # apply mask on time domain so padding on label has loss = 0
        loss = loss.mean(dim=-1) * input_mask
        # get avg loss for each example
        loss_batch = loss.mean(-1)
        # get avg loss for minibatch
        avg_loss = loss_batch.mean()
        return avg_loss, logits

    def valid_word_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=self.device)

        # valid output: get all valid hidden vectors
        # e.g., lamb --> la ##mb
        # we only get hidden vector of la
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        return valid_output

    def set_label_map(self, label_list):
        label_map = {i: label for i, label in enumerate(label_list)}
        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device



def get_label(label_id, label_map):
    '''
    inference multiple label on a single token, separate by |
    set default threshold to 0.5
    '''
    p_tmp = ''
    for dd_idx, dd in enumerate(label_id):
        if dd >= 0.5 and dd_idx != 0:
            tag_tmp = label_map[dd_idx]
            p_tmp += tag_tmp + '|'
    if p_tmp == '':
        p_tmp = 'O'
    if p_tmp[-1] == '|':
        p_tmp = p_tmp[:-1]

    return p_tmp


def evaluate(model, eval_dataloader, data_size, ref='dev_path'):
    model.eval()
    device = model.get_device()
    label_map = model.get_label_map()
    y_true = [[] for _ in range(data_size)]
    y_pred = [[] for _ in range(data_size)]
    dev_loss = 0
    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, eval_idx, sen_id in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)
        eval_idx = eval_idx.to('cpu').numpy()
        sen_id = sen_id.to('cpu').numpy()

        with torch.no_grad():
            loss, logits = model(input_ids, segment_ids, input_mask, labels=label_ids, valid_ids=valid_ids,
                                 label_mask=l_mask)

        dev_loss += loss.item()
        # get prediction with argmax on seqence domain
        # note here, this logits is valid logits, check definition of valid in function convert_examples_to_features

        # with threshold, get multiple labels as well
        # take sigmoid
        logits = torch.sigmoid(logits)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        l_mask = l_mask.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if l_mask[i][j] == 0:
                    # stop when label mask = 0
                    s_id = sen_id[i][0]
                    y_true[s_id].extend(temp_1)
                    y_pred[s_id].extend(temp_2)
                    break
                else:
                    if eval_idx[i][j] == 1:
                        # get all prediction when label mask == 1
                        t_tmp = get_label(label_ids[i][j], label_map)
                        p_tmp = get_label(logits[i][j], label_map)
                        temp_1.append(t_tmp)
                        temp_2.append(p_tmp)

    # For evaluation, we should use the original json file
    # get golden span and labels here

    f1_scorer = TaggingEval(ref=ref)
    for _, y_pred_i in zip(y_true, y_pred):
        f1_scorer.add(pred=y_pred_i)

    _, _, f1_cur = f1_scorer.get_results()
    avg_dev_loss = dev_loss / len(eval_dataloader)
    return f1_cur, avg_dev_loss


def train(model, train_dataloader, dev_size, dev_dataloader, optimizer, scheduler, max_epochs, save_ckpt, save_config, dev_ref):
    device = model.get_device()
    best_f1 = 0

    for epoch in range(max_epochs):
        train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, _, _ = batch

            loss, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                            labels=label_ids, valid_ids=valid_ids, label_mask=l_mask)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            print('Epoch: %d, step: %d, training loss: %.5f' % (epoch, step, loss.item()))
            train_loss += loss.item()

        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = save_ckpt
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = save_config
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        avg_train_loss = train_loss / len(train_dataloader)
        print("Epoch: %d, average training loss: %.5f" % (epoch, avg_train_loss))
        f1_cur, avg_dev_loss = evaluate(model, dev_dataloader, dev_size, dev_ref)
        print("Epoch: %d, F1: %.5f, average dev loss: %.5f" % (epoch, f1_cur, avg_dev_loss))

        # Save best model
        if f1_cur > best_f1:
            best_f1 = f1_cur
            # save best chpt
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = 'best_' + save_ckpt
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = 'best_' + save_config
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

    return model