import json
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
        return ['Adjudicator', 'Agent', 'Artifact', 'Attacker',
                 'Beneficiary', 'Buyer',
                 'Crime',
                 'Defendant', 'Destination',
                 'Entity',
                 'Giver',
                 'Instrument',
                 'Money',
                 'Org',
                 'Origin',
                 'Person', 'Place', 'Plaintiff', 'Position', 'Price', 'Prosecutor',
                 'Recipient',
                 'Seller', 'Sentence',
                 'Target', 'Time-After', 'Time-At-Beginning', 'Time-At-End', 'Time-Before', 'Time-Ending',
                 'Time-Holds', 'Time-Starting', 'Time-Within',
                 'Vehicle', 'Victim',
                 'O']

    def get_trigger_label(self):
        return ['Business',
                 'Conflict', 'Contact',
                 'Justice',
                 'Life',
                 'Movement',
                 'Personnel',
                 'Transaction']
        # return ['Business:Declare-Bankruptcy', 'Business:End-Org', 'Business:Merge-Org', 'Business:Start-Org',
        #          'Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write',
        #          'Justice:Acquit', 'Justice:Appeal', 'Justice:Arrest-Jail', 'Justice:Charge-Indict', 'Justice:Convict',
        #          'Justice:Execute', 'Justice:Extradite', 'Justice:Fine', 'Justice:Pardon', 'Justice:Release-Parole',
        #          'Justice:Sentence', 'Justice:Sue', 'Justice:Trial-Hearing',
        #          'Life:Be-Born', 'Life:Die', 'Life:Divorce', 'Life:Injure', 'Life:Marry',
        #          'Movement:Transport',
        #          'Personnel:Elect', 'Personnel:End-Position', 'Personnel:Nominate', 'Personnel:Start-Position',
        #          'Transaction:Transfer-Money', 'Transaction:Transfer-Ownership']

    def get_entity_label(self):
        return ["PER", "ORG", "GPE", "LOC", "FAC", "VEH", "WEA"]

    def create_examples(self, lines):
        examples = []

        # get tokens
        # get argument class labels

        for i, sent in enumerate(lines):
            tokens = sent['tokens']
            label = sent['label']
            entity_span = sent['entity-span']
            trigger_span = sent['trigger-span']

            tokens= self.insert_entity_marker(tokens, entity_span, trigger_span)

            examples.append(
                Example(text=tokens, label=label))

        return examples

    def insert_entity_marker(self, tokens, entity_span, trigger_span):
 
        entity_start, entity_end = entity_span
        trigger_start, trigger_end = trigger_span

     
            

        entity_start_token = tokens[entity_start]
        entity_end_token = tokens[entity_end - 1]
        entity = tokens[entity_start:entity_end]
        trigger_start_token = tokens[trigger_start]
        trigger_end_token = tokens[trigger_end - 1]
        trigger = tokens[trigger_start:trigger_end]

        #print(entity, trigger, tokens)
        # entity s, trigger s, trigger e, entity e
        if entity_start <= trigger_start < trigger_end <= entity_end:
            tokens.insert(entity_start, '<entity_start>')
            tokens.insert(trigger_start + 1, '<trigger_start>')
            tokens.insert(trigger_end + 2, '<trigger_end>')
            tokens.insert(entity_end + 3, '<entity_end>')
            #print('1')
        #elif entity_start <= trigger_start and trigger_ned > entity_end:
        # entity s, trigger s, entity e, trigger e
        elif entity_start < trigger_start < entity_end < trigger_end:
            tokens.insert(entity_start, '<entity_start>')
            tokens.insert(trigger_start + 1, '<trigger_start>')
            tokens.insert(entity_end + 2, '<entity_end>')
            tokens.insert(trigger_end + 3, '<trigger_end>')
            #print('2')

        # trigger s, entity s, trigger e, entity e

        elif entity_end <= trigger_start:
            tokens.insert(entity_start, '<entity_start>')
            tokens.insert(entity_end + 1, '<entity_end>')
            tokens.insert(trigger_start + 2, '<trigger_start>')
            tokens.insert(trigger_end + 3, '<trigger_end>')
            #print('3')
        # entity s, entity e, trigger s, trigger e
        # trigger s, trigger e, entity s, entity e
        elif trigger_end <= entity_start:
            tokens.insert(trigger_start, '<trigger_start>')
            tokens.insert(trigger_end + 1, '<trigger_end>')
            tokens.insert(entity_start + 2, '<entity_start>')
            tokens.insert(entity_end + 3, '<entity_end>')
            #print('4')
        elif trigger_start < entity_start < trigger_end < entity_end:
            tokens.insert(trigger_start, '<trigger_start>')
            tokens.insert(entity_start + 1, '<entity_start>')
            tokens.insert(trigger_end + 2, '<trigger_end>')
            tokens.insert(entity_end + 3, '<entity_end>')
            #print('5')
        # trigger s, entity s, entity e, trigger e
        elif trigger_start <= entity_start < entity_end <= trigger_end:
            tokens.insert(trigger_start, '<trigger_start>')
            tokens.insert(entity_start + 1, '<entity_start>')
            tokens.insert(entity_end + 2, '<entity_end>')
            tokens.insert(trigger_end + 3, '<trigger_end>')

        # print(tokens)
        s1 = tokens.index('<entity_start>')
        e1 = tokens.index('<entity_end>')
        s2 = tokens.index('<trigger_start>')
        e2 = tokens.index('<trigger_end>')

        entity_new = tokens[s1 + 1: e1]

        for i in ['<trigger_start>', '<trigger_end>']:
            if i in entity_new:
                entity_new.remove(i)

        trigger_new = tokens[s2 + 1: e2]
        for i in ['<entity_start>', '<entity_end>']:
            if i in trigger_new:
                trigger_new.remove(i)
        
        assert ''.join(entity) == ''.join(entity_new)
        
        assert ''.join(trigger) == ''.join(trigger_new)

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

proc = ACEProcessor()
for lang in ['en', 'zh', 'ar']:
    for split in ['train', 'dev', 'test']:
        proc.get_examples('../%s_arl/%s.json' % (lang, split))