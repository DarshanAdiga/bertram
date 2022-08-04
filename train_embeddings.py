import os.path
import re
import sys
import log
import random
import argparse
import os
from os import listdir
from os.path import isfile, join

from bertram import BertramWrapper
from transformers import BertForMaskedLM, BertTokenizer

logger = log.get_logger('root')


def get_more_examples_from_preprocessed(idioms, examples_folder, n=10, lower=True):
    """
    A variation of get_more_examples() that takes already processed examples.
    The example files,
      - are named with the single-token idiom (e.g. 'ID*ID.txt')
      - contain sentences where idioms are already replaced with their single-tokens 'ID*ID'
    """
    examps = {}
    for formatted_idiom in idioms:
        with open(f'{examples_folder}/{formatted_idiom}.txt', 'r') as f:
            lines = f.readlines()
            if lower:
                lines = [li.lower().replace('/', ' ') for li in lines]
                # Don't lowercase the idiom itself
                lower_idiom = formatted_idiom.lower()
                lines = [li.replace(lower_idiom, formatted_idiom) for li in lines]
            lines = [' '.join(li.split()) for li in lines]
            if len(lines) < 2:
                logger.warning(f'{formatted_idiom} has less than 2 context lines')
                continue
            examps[formatted_idiom] = random.sample(lines, k=min(len(lines), n))
    return examps

def get_idioms(examples_folder):
    onlyfiles = [f for f in listdir(examples_folder) if isfile(join(examples_folder, f))]
    return [f.split('.')[0] for f in onlyfiles]


def train_embeddings(bert_model, bertram_model, output_dir, examples_folder, no_examples):
    if os.path.isdir(output_dir):
        logger.warning(f'Output directory {output_dir} already exists. Will not overwrite.')
        return
    else:
        os.makedirs(output_dir)
    model = BertForMaskedLM.from_pretrained(bert_model)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    bertram = BertramWrapper(bertram_model, device='cuda')
    idioms = get_idioms(examples_folder)
    logger.info(f'Found {len(idioms)} en-idioms in {examples_folder}')

    examples = get_more_examples_from_preprocessed(idioms, examples_folder, no_examples)
    logger.info('Fetched examples from files')

    print(list(examples.values())[0])

    bertram.add_word_vectors_to_model(examples, tokenizer, model)
    logger.info('Added en-idioms and embeddings to bert model')

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f'Saved model and tokenizer to {output_dir}')


def main(args):
    parser = argparse.ArgumentParser("Train embeddings from a bertram model")

    parser.add_argument('--bert_model', default='bert-base-uncased', help='The underlying bert model')
    parser.add_argument('--bertram_model', help='The bertram model that will generate the embeddings')
    parser.add_argument('--output_dir',
                        help='The directory where the final model will be saved ({output_dir}-tokenizer'
                             ' will also be used to store the tokenizer for the model)')
    parser.add_argument('--examples_folder', help='The folder containing the examples')
    parser.add_argument('--no_examples', help='Number of examples used to train each idiom embedding')
    parser.add_argument('--lower', type=bool, help='Whether to lower case all examples, for uncased model')

    args = parser.parse_args(args)
    train_embeddings(args.bert_model, args.bertram_model, args.output_dir, args.examples_folder, int(args.no_examples))


if __name__ == "__main__":
    main(sys.argv[1:])
