from unsupervisedqa.generate_clozes import generate_clozes_from_paragraph, named_entity_answer_generator, noun_phrase_answer_generator
import json
import argparse
from unsupervisedqa.configs import CLOZE_MASKS
from unsupervisedqa.data_classes import Paragraph
from tqdm import tqdm


def _tqdm(iterator, desc=None):
    return tqdm(iterator, desc=desc, ncols=80)


def _wh_heurisistic(cloze):
    cloze_mask = CLOZE_MASKS[cloze.answer_type]
    cloze_text = cloze_mask + ' ' + cloze.cloze_text.replace(cloze_mask, 'MASK')
    return cloze_text


def mine_clozes(input_file, output_file, use_named_entity_clozes=True, use_wh_heuristic=False):
    print('Loading json file:')
    with open(input_file) as f:
        ds = json.load(f)

    print('Parsing Paragraphs:')
    paragraphs = [Paragraph(
        paragraph_id=-1,
        text=p['context']) for a in _tqdm(ds['data'], desc='Parsing Paragraphs') for p in a['paragraphs']
    ]
    print(f'Parsed {len(paragraphs)} paragraphs')
    answer_generator = named_entity_answer_generator if use_named_entity_clozes else noun_phrase_answer_generator
    print('Building Clozes:')
    clozes = [c for p in _tqdm(paragraphs, desc='Building Clozes')
              for c in generate_clozes_from_paragraph(p, answer_generator)]
    print(f'Built {len(clozes)} clozes from {len(paragraphs)} paragraphs')
    print('Building Clozes:')

    with open(output_file, 'w') as f:
        for cloze in _tqdm(clozes, desc='Dumping Clozes'):
            cloze_text = _wh_heurisistic(cloze) if use_wh_heuristic else cloze.cloze_text
            f.write(cloze_text + '\n')

    return clozes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mine clozes from a squad-formatted file')
    parser.add_argument("--path_to_squad", type=str, help="path to squad file to calculate length distribution")
    parser.add_argument("--output_path", type=str, help="path to write question distribution file to")
    parser.add_argument("--use_noun_phrase_clozes", action='store_true', help="use noun chunks rather than named entities to build clozes")
    parser.add_argument("--use_wh_word_heuristic", action='store_true', help="use the wh* word heuristic to build clozes")

    args = parser.parse_args()
    mine_clozes(args.path_to_squad, args.output_path,
                use_named_entity_clozes=not args.use_noun_phrase_clozes,
                use_wh_heuristic=args.use_wh_word_heuristic)
