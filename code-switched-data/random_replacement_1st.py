from argparse import ArgumentParser
import random,os,sys

panlex=True
muse=False
wiki=False
if panlex:
	bilingual_file='bilingual-panlex-vocab.txt'
	bilingual_dict_panlex = {}
	for line in open(bilingual_file):
		line = line.split(' ||| ')
		if len(line[1:-1]) > 0:
			bilingual_dict_panlex[line[0]] = line[1:-1]
if muse:
	bilingual_file='bilingual-muse-vocab.txt'
	bilingual_dict_muse = {}
	for line in open(bilingual_file):
		line=line.split(' ||| ')
		if len(line[1:-1])>0:
			bilingual_dict_muse[line[0]]=line[1:-1]
if wiki:
	bilingual_file='bilingual-wiki-title.txt'
	bilingual_dict_wiki = {}
	for line in open(bilingual_file):
		line=line.strip().lower().split(' ||| ')
		bilingual_dict_wiki[line[0]]=line[1]
		bilingual_dict_wiki[line[1]]=line[0]

parser = ArgumentParser()
parser.add_argument("--file-path", required=True, metavar="<file_path>", type=str)
parser.add_argument("--output-dir", required=True, metavar="<output_dir>", type=str)
A = parser.parse_args()

sentence_replacement_ratio=1.0
token_replacement_ratio=0.3
max_span_size=5
output_path = os.path.join(A.output_dir, os.path.basename(A.file_path) + ".cs")
with open(output_path,'w') as f:
	for line in open(A.file_path):
		if random.random()>sentence_replacement_ratio:
			f.writelines(line)
			continue
		tok_list=line.strip().lower().split()
		tok_idx=[i for i in range(len(tok_list))]
		random.shuffle(tok_idx)
		counter=0
		ii=0
		while counter<len(tok_list)*token_replacement_ratio and ii<len(tok_list):
			hit_wiki = False
			if wiki:
				shuffled_ii=tok_idx[ii]
				for span_size in range(max_span_size, 0, -1):
					start_idx = max(0, shuffled_ii-span_size+1)
					for jj in range(start_idx, shuffled_ii+1):
						end_index=min(jj+span_size, len(tok_list))
						span=' '.join(tok_list[jj:end_index])
						if span in bilingual_dict_wiki:
							tok_list[jj]=bilingual_dict_wiki[span]
							for kk in range(jj+1, end_index):
								tok_list[kk]=''
							counter+=span_size
							hit_wiki=True
							break
					if hit_wiki:
						break
			if not hit_wiki:
				word=tok_list[tok_idx[ii]]
				if panlex:
					if word in bilingual_dict_panlex:
						translations=[' '.join(item.split()[0:-1]) for item in bilingual_dict_panlex[word]]
						weights=[int(item.split()[-1]) for item in bilingual_dict_panlex[word]]
						tok_list[tok_idx[ii]] = random.choices(translations, weights=weights, k=1)[0]
						counter += 1
					else:
						if muse:
							if word in bilingual_dict_muse:
								tok_list[tok_idx[ii]] = random.sample(bilingual_dict_muse[word], 1)[0]
								counter += 1
				elif muse:
					if word in bilingual_dict_muse:
						tok_list[tok_idx[ii]]=random.sample(bilingual_dict_muse[word],1)[0]
						counter+=1
			ii+=1
		f.writelines(' '.join(tok_list)+'\n')
