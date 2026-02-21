import csv
import sys

import torch
from transformers import *
import torch.nn.functional as F
import numpy as np
from scipy import spatial
from collections import defaultdict

#Roberta and Gpt2 use bype-level BPE
def init_model(model_name):
    if model_name == "kobert":
        pretrained_name = 'beomi/kcbert-large'
        tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        model = BertModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = BertForMaskedLM.from_pretrained(pretrained_name).eval()
    elif model_name == "koroberta":
        pretrained_name = 'klue/roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        model = AutoModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = AutoModelWithLMHead.from_pretrained(pretrained_name).eval()
    #elif model_name == "kogpt2":
    #    pretrained_name = 'skt/kogpt2-base-v2'
    #    #pretrained_name = 'kykim/gpt3-kor-small_based_on_gpt2'
    #    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_name)
    #    model = GPT2Model.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    #    model_lm = GPT2LMHeadModel.from_pretrained(pretrained_name).eval()
    elif model_name == "kogpt2":
        pretrained_name = 'EleutherAI/polyglot-ko-1.3b'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        model = AutoModelForCausalLM.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = AutoModelWithLMHead.from_pretrained(pretrained_name).eval()
    elif model_name == "kogpt2-3.8b":
        pretrained_name = 'EleutherAI/polyglot-ko-3.8b'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        model = AutoModelForCausalLM.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
        model_lm = AutoModelWithLMHead.from_pretrained(pretrained_name).eval()

    #elif model_name == "koalpaca":
    #    pretrained_name = 'beomi/KoAlpaca-Polyglot-12.8B'
    #    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    #    model = AutoModelForCausalLM.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    #    model_lm = AutoModelWithLMHead.from_pretrained(pretrained_name).eval()
    else:
        print('wrong model name')

    return tokenizer, model, model_lm

def match_piece_to_word(piece, word):
    mapping = defaultdict(list)
    word_index = 0
    piece_index = 0
    while (word_index < len(word.split()) and piece_index < len(piece)):
        if piece[piece_index] != '[UNK]':
            mid = piece[piece_index].strip('Ġ').strip('▁').strip('##')
            mid = mid.replace('</w>', '')
            t = len(mid)
        else:
            t = 1
        while (piece_index + 1 < len(piece) and t<len(word.split()[word_index])):
            mapping[word_index].append(piece_index)
            piece_index += 1
            if piece[piece_index] != '[UNK]':
                mid = piece[piece_index].strip('Ġ').strip('▁').strip('##')
                mid = mid.replace('</w>', '')
                t += len(mid)
            else:
                t += 1
        try:
            assert(t == len(word.split()[word_index]))
        except:
            print(word)
            print(piece)
            #import pdb
            #pdb.set_trace()
        mapping[word_index].append(piece_index)
        word_index += 1
        piece_index += 1
    return mapping

def convert_logits_to_probs(logits, input_ids):
    """"
    input:
        logits: (1, n_word, n_vocab), GPT2 outputed logits of each word
        input_inds: (1, n_word), the word id in vocab
    output: probs: (1, n_word), the softmax probability of each word
    """

    probs = F.softmax(logits[0], dim=1)
    n_word = input_ids.shape[1]
    res = []
    for i in range(n_word):
        res.append(probs[i, input_ids[0][i]].item())
    return np.array(res).reshape(1, n_word)


if __name__ == '__main__':
    '''
    parameters
    inputfile: sentences with cue and target word
    
    '''
    #inputfiles = ['kor_polite_1.txt', 'kor_polite_2.txt', 'kor_polite_3.txt', 'kor_polite_4.txt', 'kor_polite_5.txt', 'kor_polite_6.txt', 'kor_polite_7.txt', 'kor_polite_8.txt', 'kor_baseline.txt']
    inputfiles = ['long_kor_polite_1.txt', 'long_kor_polite_2.txt', 'long_kor_polite_3.txt', 'long_kor_polite_4.txt', 'long_kor_polite_5.txt', 'long_kor_polite_6.txt', 'long_kor_polite_7.txt', 'long_kor_polite_8.txt']
    inputfile = inputfiles[int(sys.argv[1])]

    #model_names = ["kobert", "koroberta", "kogpt2", "koalpaca"]
    #model_names = ["kogpt2", "koalpaca"]
    model_names = ["kobert", "koroberta", "kogpt2", "kogpt2-3.8b"]
    #model_names = ["kogpt2-3.8b"]
    for model_name in model_names:
        print(model_name)
        tokenizer, model, model_lm = init_model(model_name)

        outfile = open('results/out_' + model_name+ '_' + inputfile, 'w')
        for input in open(inputfile):
            input_sent = input.strip()
            if not len(input_sent):
                outfile.write('\n')
                continue

            input_ids = tokenizer.encode(input_sent, return_tensors = "pt")
            tok_input = tokenizer.convert_ids_to_tokens(input_ids[0])
            input_ids = input_ids[:,1:-1]
            tok_input = tok_input[1:-1]

            word_piece_mapping = match_piece_to_word(tok_input, input_sent)

            with torch.no_grad():
                outputs = model(input_ids)
                logits = model_lm(input_ids)[0]

            prob = convert_logits_to_probs(logits, input_ids)[0]
            # print(len(prob), prob)
            mid1 = []
            sent_pp = 1
            for i in range(len(word_piece_mapping)):
                pp = 1
                for j in word_piece_mapping[i]:
                    pp *= prob[j]
                    sent_pp *= prob[j]
                mid1.append(str(pp))
            outfile.write(input_sent + ' prob_sent: ' + str(sent_pp) + ' prob_target: ' + str(mid1[-2]) + '\n')



