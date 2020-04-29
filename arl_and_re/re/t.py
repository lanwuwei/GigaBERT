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

proc = ACEProcessor()
proc.get_examples('../ar_re/test.json')