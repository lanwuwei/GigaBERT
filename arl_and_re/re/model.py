import torch
import torch.nn.functional as F
from transformers import *
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch import nn
import json
import copy

class Example(object):
    def __init__(self, text, label=None, t_span=None, e_span=None):
        self.text = text
        self.label = label
        self.t_span = t_span
        self.e_span = e_span

class ACEProcessor(object):
    '''Processor for CoNLL-2003 data set.'''

    def get_examples(self, data_dir):
        res = self.read_json(data_dir)
        examples = self.create_examples(res)
        return examples

    def get_labels(self):
        return ['ART:User-Owner-Inventor-Manufacturer',
                 'GEN-AFF:Citizen-Resident-Religion-Ethnicity',
                 'GEN-AFF:Org-Location',
                 'ORG-AFF:Employment',
                 'ORG-AFF:Founder',
                 'ORG-AFF:Investor-Shareholder',
                 'ORG-AFF:Membership',
                 'ORG-AFF:Ownership',
                 'ORG-AFF:Sports-Affiliation',
                 'ORG-AFF:Student-Alum',
                 'PART-WHOLE:Artifact',
                 'PART-WHOLE:Geographical',
                 'PART-WHOLE:Subsidiary',
                 'PER-SOC:Business',
                 'PER-SOC:Family',
                 'PER-SOC:Lasting-Personal',
                 'PHYS:Located',
                 'PHYS:Near',
                 'O']

    def get_entity_label(self):
        return ["PER", "ORG", "GPE", "LOC", "FAC", "VEH", "WEA"]

    def create_examples(self, lines):
        examples = []

        # get tokens
        # get argument class labels

        for i, sent in enumerate(lines):
            tokens = sent['tokens']
            label = sent['label']
            arg1_span = sent['arg1-span']
            arg2_span = sent['arg2-span']
            if arg1_span != arg2_span:
                tokens = self.insert_entity_marker(tokens, arg1_span, arg2_span)

            examples.append(
                Example(text=tokens, label=label))

        return examples

    def insert_entity_marker(self, tokens, arg1_span, arg2_span):
        arg1_start, arg1_end = arg1_span
        arg2_start, arg2_end = arg2_span

     
            

        arg1_start_token = tokens[arg1_start]
        arg1_end_token = tokens[arg1_end - 1]
        arg1 = tokens[arg1_start:arg1_end]
        arg2_start_token = tokens[arg2_start]
        arg2_end_token = tokens[arg2_end - 1]
        arg2 = tokens[arg2_start:arg2_end]

        #print(arg1, arg2, tokens)
        # arg1 s, arg2 s, arg2 e, arg1 e
        if arg1_start <= arg2_start < arg2_end <= arg1_end:
            tokens.insert(arg1_start, '<arg1_start>')
            tokens.insert(arg2_start + 1, '<arg2_start>')
            tokens.insert(arg2_end + 2, '<arg2_end>')
            tokens.insert(arg1_end + 3, '<arg1_end>')
            #print('1')
        #elif arg1_start <= arg2_start and arg2_ned > arg1_end:
        # arg1 s, arg2 s, arg1 e, arg2 e
        elif arg1_start < arg2_start < arg1_end < arg2_end:
            tokens.insert(arg1_start, '<arg1_start>')
            tokens.insert(arg2_start + 1, '<arg2_start>')
            tokens.insert(arg1_end + 2, '<arg1_end>')
            tokens.insert(arg2_end + 3, '<arg2_end>')
            #print('2')

        # arg2 s, arg1 s, arg2 e, arg1 e

        elif arg1_end <= arg2_start:
            tokens.insert(arg1_start, '<arg1_start>')
            tokens.insert(arg1_end + 1, '<arg1_end>')
            tokens.insert(arg2_start + 2, '<arg2_start>')
            tokens.insert(arg2_end + 3, '<arg2_end>')
            #print('3')
        # arg1 s, arg1 e, arg2 s, arg2 e
        # arg2 s, arg2 e, arg1 s, arg1 e
        elif arg2_end <= arg1_start:
            tokens.insert(arg2_start, '<arg2_start>')
            tokens.insert(arg2_end + 1, '<arg2_end>')
            tokens.insert(arg1_start + 2, '<arg1_start>')
            tokens.insert(arg1_end + 3, '<arg1_end>')
            #print('4')
        elif arg2_start < arg1_start < arg2_end < arg1_end:
            tokens.insert(arg2_start, '<arg2_start>')
            tokens.insert(arg1_start + 1, '<arg1_start>')
            tokens.insert(arg2_end + 2, '<arg2_end>')
            tokens.insert(arg1_end + 3, '<arg1_end>')
            #print('5')
        # arg2 s, arg1 s, arg1 e, arg2 e
        elif arg2_start <= arg1_start < arg1_end <= arg2_end:
            tokens.insert(arg2_start, '<arg2_start>')
            tokens.insert(arg1_start + 1, '<arg1_start>')
            tokens.insert(arg1_end + 2, '<arg1_end>')
            tokens.insert(arg2_end + 3, '<arg2_end>')
            #print('6')
       # print(tokens)
        s1 = tokens.index('<arg1_start>')
        e1 = tokens.index('<arg1_end>')
        s2 = tokens.index('<arg2_start>')
        e2 = tokens.index('<arg2_end>')

        arg1_new = tokens[s1 + 1: e1]

        for i in ['<arg2_start>', '<arg2_end>']:
            if i in arg1_new:
                arg1_new.remove(i)

        arg2_new = tokens[s2 + 1: e2]
        for i in ['<arg1_start>', '<arg1_end>']:
            if i in arg2_new:
                arg2_new.remove(i)
        
        assert ''.join(arg1) == ''.join(arg1_new)
        
        assert ''.join(arg2) == ''.join(arg2_new)

        return tokens

    def create_input_token(self, tokens, etype, span):
        copy_tokens = copy.deepcopy(tokens)
        for i in range(span[0], span[1]):
            copy_tokens[i] = '[' + etype + ']'
        return copy_tokens

    def read_json(self, filename):
        '''
        read file
        '''
        print('Reading file: ', filename)
        with open(filename, encoding='utf-8') as json_file:
            data = json.load(json_file)
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
        t_tokens = ' '.join(example.text)
        label = label_map[example.label]
        total_tokens = tokenizer.tokenize(t_tokens)

        if len(total_tokens) > max_seq_length - 2:
            total_tokens = total_tokens[:max_seq_length - 2]

        ntokens = []
        segment_ids = []
        ntokens.append(tokenizer.bos_token)
        segment_ids.append(0)

        for token in total_tokens:
            ntokens.append(token)
            segment_ids.append(0)

        ntokens.append(tokenizer.eos_token)
        segment_ids.append(0)
        try:
            arg1_start_index = ntokens.index('<arg1_start>')
        except:
            arg1_start_index = 0
        try:
            arg2_start_index = ntokens.index('<arg2_start>')
        except:
            arg2_start_index = 0
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            input_mask.append(0)
            segment_ids.append(0)

        features.append(
            Features(input_ids, segment_ids=segment_ids, input_mask=input_mask,
                     label_id=label, t_span=arg1_start_index, e_span=arg2_start_index)
        )
    return features

class Features(object):
    def __init__(self, input_ids, input_mask, segment_ids,
                 label_id, t_span=None, e_span=None, valid_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.t_span = t_span
        self.e_span = e_span
        self.valid_id = valid_id

def create_dataloader(data_dir, set_type='train', batchsize=32, max_seq_length=128, tokenizer=None, num_duplicate=None, proc=None):

    data_examples = []
    data_examples.extend(proc.get_examples(data_dir))

    data_size = len(data_examples)
    print('Dataset set: %s, orig size: %s' % (set_type, data_size))

    data_size_total = data_size

    data_features = convert_examples_to_features(data_examples, proc.get_labels(),
                                           max_seq_length, tokenizer, num_duplicate)
    all_input_ids = torch.tensor([f.input_ids for f in data_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in data_features], dtype=torch.long, requires_grad=False)
    all_segment_ids = torch.tensor([f.segment_ids for f in data_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in data_features], dtype=torch.long)
    all_t_span = torch.tensor([f.t_span for f in data_features], dtype=torch.long)
    all_e_span = torch.tensor([f.e_span for f in data_features], dtype=torch.long)
    # all_valid_ids = torch.tensor([f.valid_id for f in data_features], dtype=torch.long)


    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_e_span, all_t_span)
    if set_type == 'train':
        data_sampler = RandomSampler(dataset)
    else:
        data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batchsize)

    return dataloader, data_size_total


class BertForSentClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSentClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e_span=None, t_span=None):

        # get bert output
        output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # last_hidden_states, pooler_output, hidden_states

        sequence_output = output[0]
        #cls_token = sequence_output[:, 0]
        batchsize, _, _ = sequence_output.size()
        batch_index = [i for i in range(batchsize)]
        ent_repr = sequence_output[batch_index, e_span]
        trig_repr = sequence_output[batch_index, t_span]

        cls_token = torch.cat((ent_repr, trig_repr), -1)
        repr = self.dropout(cls_token)
        logits = self.classifier(repr)
        avg_loss = F.cross_entropy(logits, labels)
        return avg_loss, logits

    def set_label_map(self, label_list):
        label_map = {i: label for i, label in enumerate(label_list)}

        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device

class XLMRobertaForSentClassification(RobertaForSequenceClassification):
    config_class = XLMRobertaConfig
    XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-pytorch_model.bin",
}
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    def __init__(self, config):
        super(XLMRobertaForSentClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, e_span=None, t_span=None):

        # get bert output
        output = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # last_hidden_states, pooler_output, hidden_states

        sequence_output = output[0]
        #cls_token = sequence_output[:, 0]
        batchsize, _, _ = sequence_output.size()
        batch_index = [i for i in range(batchsize)]
        ent_repr = sequence_output[batch_index, e_span]
        trig_repr = sequence_output[batch_index, t_span]

        cls_token = torch.cat((ent_repr, trig_repr), -1)
        repr = self.dropout(cls_token)
        logits = self.classifier(repr)
        avg_loss = F.cross_entropy(logits, labels)
        return avg_loss, logits

    def set_label_map(self, label_list):
        label_map = {i: label for i, label in enumerate(label_list)}

        self.label_map = label_map

    def get_label_map(self):
        return self.label_map

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device


def evaluate(model, eval_dataloader):
    model.eval()
    device = model.get_device()
    label_map = model.get_label_map()
    y_true = []
    y_pred = []

    dev_loss = 0

    for input_ids, input_mask, segment_ids, label_ids, e_span, t_span in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        # valid_id = valid_id.to(device)
        e_span = e_span.to(device)
        t_span = t_span.to(device)

        with torch.no_grad():
            loss, logits = model(input_ids, segment_ids, input_mask, labels=label_ids, e_span=e_span, t_span=t_span)

        dev_loss += loss.item()

        logits = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            y_true.append(label_map[label_ids[i]])
            y_pred.append(label_map[logits[i]])
    avg_dev_loss = dev_loss / len(eval_dataloader)
    n_correct = 0
    n_predicted = 0
    n_gold = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_t == 'O':
            y_t = set()
        else:
            y_t = set(y_t)
        if y_p == 'O':
            y_p = set()
        else:
            y_p = set(y_p)


        n_correct += len(y_t & y_p)
        n_gold += len(y_t)
        n_predicted += len(y_p)
    if n_correct == 0:
        p, r, f1 = 0, 0, 0
    else:
        p = 100.0 * n_correct / n_predicted
        r = 100.0 * n_correct / n_gold
        f1 = 2 * p * r / (p + r)

    return f1, avg_dev_loss


def train(model, train_dataloader, dev_size, dev_dataloader, optimizer, scheduler, max_epochs, save_ckpt, save_config):

    device = model.get_device()
    best_acc = 0

    for epoch in range(max_epochs):
        train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids, e_span, t_span = batch

            loss, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                            labels=label_ids, e_span=e_span, t_span=t_span)

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
        acc_cur, avg_dev_loss = evaluate(model, dev_dataloader)
        print("Epoch: %d, ACC: %.5f, average dev loss: %.5f" % (epoch, acc_cur, avg_dev_loss))

        # Save best model
        if acc_cur > best_acc:
            best_acc = acc_cur
            # save best chpt
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = 'best_' + save_ckpt
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = 'best_' + save_config
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

    return model
