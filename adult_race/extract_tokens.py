import argparse
from transformers import RobertaTokenizer, DebertaTokenizer, BertTokenizer
import sys
sys.path.append("../common")
from data_utils import Adult_extract_tokens_data_1

parser = argparse.ArgumentParser(description='Extracting tokens for training adult.')
parser.add_argument('--model', type=str, help='model_version for tokenizer', choices=['deberta-base', 'bert-base-uncased'])

args = parser.parse_args()

if args.model == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(args.model)
else:
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

print("Loading")
Adult_extract_tokens_data_1(tokenizer, args.model)
print("finish")