import argparse
import random
import numpy as np
import torch
from pos_ud import *
from transformers import *
# take args
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--source_language", default='en', type=str,
                    help="The target language")
parser.add_argument("--target_language", default='en', type=str,
                    help="The target language")
parser.add_argument("--bert_model", default='', type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")

parser.add_argument("--output_dir", default='save', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--ckpt", default=None, type=str,
                    help="Checkpoint for previously saved mdoel")
parser.add_argument("--exp_name", default=None, type=str,
                    help="Checkpoint and config save prefix")
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument("--num_exp", default=None, type=int,
                    help="Number of additional examples from source language")
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_epoch", default=5, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpuid", default='0', type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--num_duplicate", default=20, type=int)
parser.add_argument("--warmup_proportion", default=0.4, type=float)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
args = parser.parse_args()


if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_ckpt = args.exp_name + '.ckpt'
    save_config = args.exp_name + '.cfg'

    # parse source domains
    print('F1 ================== EXP =====================')
    source_language = args.source_language
    target_language = args.target_language
    print('F1 Target language: %s' % target_language)

    print('batchsize: %d' % args.batchsize)
    print('learning rate: %.7f' % args.learning_rate)
    print('max epochs: %d' % args.max_epoch)
    print('max_seq_length: %d' % args.max_seq_length)
    print('num_depulicate: %d' % args.num_duplicate)
    print('warmup proportion: %.5f' % args.warmup_proportion)
    print('model ckpt will be saved at: %s' % save_ckpt)
    print('model config will be saved at: %s' % save_config)

    processor = UDProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    device = torch.device('cuda:' + args.gpuid)
    # build model
    if args.bert_model == 'bert-base-multilingual-cased':
        model = BertForPOS.from_pretrained(args.bert_model,
                                    cache_dir=args.output_dir,
                                    num_labels = num_labels,
                                    output_hidden_states=True) # if you want to get all layer hidden states
    #elif args.bert_model == 'xlm-roberta-base':
    #    model = XLMRobertaForPOS.from_pretrained(args.bert_model,
    #                                       cache_dir=args.output_dir,
    #                                       num_labels=num_labels,
    #                                       output_hidden_states=True)  # if you want to get all layer hidden states
    elif args.bert_model == 'xlm-mlm-xnli15-1024':
        model = XLMForPOS.from_pretrained(args.bert_model,
                                           cache_dir=args.output_dir,
                                           num_labels=num_labels,
                                           output_hidden_states=True)  # if you want to get all layer hidden states
    elif args.bert_model == 'xlm-mlm-tlm-xnli15-1024':
        model = XLMForPOS.from_pretrained(args.bert_model,
                                           cache_dir=args.output_dir,
                                           num_labels=num_labels,
                                           output_hidden_states=True)  # if you want to get all layer hidden states
    elif args.bert_model == 'xlm-roberta-large':
        model = XLMRobertaForPOS.from_pretrained(args.bert_model,
                                           cache_dir=args.output_dir,
                                           num_labels=num_labels,
                                           output_hidden_states=True)  # if you want to get all layer hidden states
    else:
        config = BertConfig.from_json_file('/home/wuwei-lan/Documents/huggingface/saved_model/'+args.bert_model+'/bert_config.json')
        config.num_labels = num_labels
        config.output_hidden_states = True
        model = BertForPOS(config=config)
        model.load_state_dict(torch.load('/home/wuwei-lan/Documents/huggingface/saved_model/'+args.bert_model+'/pytorch_model.bin', map_location=device),strict=False)
        print('load '+args.bert_model+' finished.')

    model.set_label_map(label_list)

    model.to(device)
    model.set_device('cuda:' + args.gpuid)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    lang2full = {'zh': 'Chinese',
                 'ar': 'Arabic',
                 'bg': 'Bulgarian',
                 'da': 'Danish',
                 'nl': 'Dutch',
                 'en': 'English',
                 'de': 'German',
                 'hu': 'Hungarian',
                 'it': 'Italian',
                 'fa': 'Persian',
                 'pl': 'Polish',
                 'pt': 'Portuguese',
                 'ro': 'Romanian',
                 'sk': 'Slovak',
                 'sl': 'Slovenian',
                 'es': 'Spanish',
                 'sv': 'Swedish'}

    training_data_path = 'UD_' + lang2full[source_language] + '/' + source_language+'-ud-train.conllu'

    dev_data_path = 'UD_' + lang2full[target_language] + '/' + target_language+'-ud-dev.conllu'
    test_data_path = 'UD_' + lang2full[target_language] + '/' + target_language+'-ud-test.conllu'

    train_examples = processor.get_examples(training_data_path)

    num_train_optimization_steps = int(
        len(train_examples) / args.batchsize / args.gradient_accumulation_steps) * args.max_epoch

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        args.warmup_proportion * num_train_optimization_steps), num_training_steps=num_train_optimization_steps)

    if args.bert_model == 'bert-base-multilingual-cased':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        tokenizer.bos_token = '[CLS]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.unk_token = '[UNK]'
        tokenizer.sep_token = '[SEP]'
        tokenizer.cls_token = '[CLS]'
        tokenizer.mask_token = '[MASK]'
        tokenizer.pad_token = '[PAD]'
    elif args.bert_model == 'xlm-roberta-base':
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    elif args.bert_model == 'xlm-roberta-large':
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    elif args.bert_model == 'xlm-mlm-xnli15-1024':
        tokenizer = XLMTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        tokenizer.bos_token = '<s>'
        tokenizer.eos_token = '</s>'
        tokenizer.unk_token = '<unk>'
        tokenizer.sep_token = '</s>'
        tokenizer.cls_token = '</s>'
        tokenizer.mask_token = '<special1>'
        tokenizer.pad_token = '<pad>'

    elif args.bert_model == 'xlm-mlm-tlm-xnli15-1024':
        tokenizer = XLMTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
        tokenizer.bos_token = '<s>'
        tokenizer.eos_token = '</s>'
        tokenizer.unk_token = '<unk>'
        tokenizer.sep_token = '</s>'
        tokenizer.cls_token = '</s>'
        tokenizer.mask_token = '<special1>'
        tokenizer.pad_token = '<pad>'
    else:
        if args.bert_model == 'bibert':
            lower_case_flag=True
        elif args.bert_model == 'bibert-124k':
            lower_case_flag=False
        print('lower_case_flag: ', lower_case_flag)
        tokenizer = BertTokenizer.from_pretrained('/home/wuwei-lan/Documents/huggingface/saved_model/'+args.bert_model+'/vocab.txt', do_lower_case=lower_case_flag)
        tokenizer.bos_token = '[CLS]'
        tokenizer.eos_token = '[SEP]'
        tokenizer.unk_token = '[UNK]'
        tokenizer.sep_token = '[SEP]'
        tokenizer.cls_token = '[CLS]'
        tokenizer.mask_token = '[MASK]'
        tokenizer.pad_token = '[PAD]'

    # make data loader for train/dev/test
    print('Loading training data...\n')
    train_dataloader, _ = create_dataloader(training_data_path, set_type='train', batchsize=args.batchsize,
                                            max_seq_length=args.max_seq_length, tokenizer=tokenizer,
                                            num_duplicate=args.num_duplicate)
    print('Loading development data...\n')
    dev_dataloader, dev_size = create_dataloader(dev_data_path, set_type='dev',
                                                 batchsize=args.batchsize,
                                                 max_seq_length=args.max_seq_length, tokenizer=tokenizer,
                                                 num_duplicate=args.num_duplicate)
    print('Loading testing data...\n')
    test_dataloader, test_size = create_dataloader(test_data_path, set_type='test',
                                                   batchsize=args.batchsize,
                                                   max_seq_length=args.max_seq_length, tokenizer=tokenizer,
                                                   num_duplicate=args.num_duplicate)

    # train
    print('Training started...')
    model = train(model, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader,
                  dev_size=dev_size, optimizer=optimizer, scheduler=scheduler, max_epochs=args.max_epoch,
                  save_ckpt=save_ckpt, save_config=save_config)

    # Load best checkpoint
    print('Loading best check point...')
    output_model_file = 'best_' + save_ckpt
    model.load_state_dict(torch.load(output_model_file, map_location=device))

    # test
    print('Evaluating on dev set...\n')
    acc, avg_loss = evaluate(model, dev_dataloader, dev_size)
    print('DEV ACC: %.5f, avg loss: %.5f' % (acc, avg_loss))
    print('Evaluating on test set...\n')
    acc, avg_loss = evaluate(model, test_dataloader, test_size)
    print('Test ACC: %.5f, avg loss: %.5f' % (acc, avg_loss))
