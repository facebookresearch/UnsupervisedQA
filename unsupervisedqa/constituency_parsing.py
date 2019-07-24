# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Functionality to do constituency parsing, used for shortening cloze questions. We use AllenNLP and the
Parsing model from Stern et. al, 2018 "A Minimal Span-Based Neural Constituency Parser" arXiv:1705.03919
"""
import attr
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from tqdm import tqdm
from nltk import Tree
from .configs import CONSTITUENCY_MODEL, CONSTITUENCY_BATCH_SIZE, CONSTITUENCY_CUDA, CLOZE_SYNTACTIC_TYPES
from .generate_clozes import mask_answer
from .data_classes import Cloze


def _load_constituency_parser():
    archive = load_archive(CONSTITUENCY_MODEL, cuda_device=CONSTITUENCY_CUDA)
    return Predictor.from_archive(archive, 'constituency-parser')


def get_constituency_parsed_clozes(clozes, predictor=None, verbose=True, desc='Running Constituency Parsing'):
    if predictor is None:
        predictor = _load_constituency_parser()
    jobs = range(0, len(clozes), CONSTITUENCY_BATCH_SIZE)
    for i in tqdm(jobs, desc=desc, ncols=80) if verbose else jobs:
        input_batch = clozes[i: i + CONSTITUENCY_BATCH_SIZE]
        output_batch = predictor.predict_batch_json([{'sentence': c.source_text} for c in input_batch])
        for c, t in zip(input_batch, output_batch):
            root = _get_root_type(t['trees'])
            if root in CLOZE_SYNTACTIC_TYPES:
                c_with_parse = attr.evolve(c, constituency_parse=t['trees'], root_label=root)
                yield c_with_parse


def _get_root_type(tree):
    try:
        t = Tree.fromstring(tree)
        label = t.label()
    except:
        label = 'FAIL'
    return label


def _get_sub_clauses(root, clause_labels):
    """Simplify a sentence by getting clauses"""
    subtexts = []
    for current in root.subtrees():
        if current.label() in clause_labels:
            subtexts.append(' '.join(current.leaves()))
    return subtexts


def _tokens2spans(sentence, tokens):
    off = 0
    spans = []
    for t in tokens:
        span_start = sentence[off:].index(t) + off
        spans.append((span_start, span_start + len(t)))
        off = spans[-1][-1]
    for t, (s, e) in zip(tokens, spans):
        assert sentence[s:e] == t
    return spans


def _subseq2sentence(sentence, tokens, token_spans, subsequence):
    subsequence_tokens = subsequence.split(' ')
    for ind in (i for i, t in enumerate(tokens) if t == subsequence_tokens[0]):
        if tokens[ind: ind + len(subsequence_tokens)] == subsequence_tokens:
            return sentence[token_spans[ind][0]:token_spans[ind + len(subsequence_tokens) - 1][1]]
    raise Exception('Failed to repair sentence from token list')


def get_sub_clauses(sentence, tree):
    clause_labels = CLOZE_SYNTACTIC_TYPES
    root = Tree.fromstring(tree)
    subs = _get_sub_clauses(root, clause_labels)
    tokens = root.leaves()
    token_spans = _tokens2spans(sentence, tokens)
    return [_subseq2sentence(sentence, tokens, token_spans, sub) for sub in subs]


def shorten_cloze(cloze):
    """Return a list of shortened cloze questions from the original cloze question"""
    simple_clozes = []
    try:
        subs = get_sub_clauses(cloze.source_text, cloze.constituency_parse)
        subs = sorted(subs)
        for sub in subs:
            if sub != cloze.source_text:
                sub_start_index = cloze.source_text.index(sub)
                sub_answer_start_index = cloze.answer_start - sub_start_index
                good_start = 0 <= sub_answer_start_index <= len(sub)
                good_end = 0 <= sub_answer_start_index + len(cloze.answer_text) <= len(sub)
                if good_start and good_end:
                    simple_clozes.append(
                        Cloze(
                            cloze_id=cloze.cloze_id + f'_{len(simple_clozes)}',
                            paragraph=cloze.paragraph,
                            source_text=sub,
                            source_start=cloze.source_start + sub_start_index,
                            cloze_text=mask_answer(sub, cloze.answer_text, sub_answer_start_index, cloze.answer_type),
                            answer_text=cloze.answer_text,
                            answer_start=sub_answer_start_index,
                            constituency_parse=None,
                            root_label=None,
                            answer_type=cloze.answer_type,
                            question_text=None
                        )
                    )
    except:
        print(f'Failed to parse cloze: ID {cloze.cloze_id} Text: {cloze.source_text}')
    return simple_clozes
