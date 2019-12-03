import json
import spacy
from unsupervisedqa.configs import SPACY_MODEL
import argparse

nlp = spacy.load(SPACY_MODEL)


def get_question_length_distribution(inpath, outpath):
    with open(inpath) as f:
        ds = json.load(f)

        counter = {}
    for a in ds['data']:
        for p in a['paragraphs']:
            for qa in p['qas']:
                nt = len(nlp(qa['question']))
                counter[nt] = counter.get(nt, 0) + 1
    t = sum(counter.values())
    distro = {c: 100 * v / t for c, v in counter.items()}

    with open(outpath, 'w') as f:
        json.dump(distro, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get a question distribution from a squad-formatted file')
    parser.add_argument("--path_to_squad", type=str, help="path to squad file to calculate length distribution")
    parser.add_argument("--output_path", type=str, help="path to write question distribution file to")
    args = parser.parse_args()
    get_question_length_distribution(args.path_to_squad, args.output_path)
