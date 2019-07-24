# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Functionality to implement baseline cloze question translators,
referred to as "identity" and "noisy cloze" baselien methods in the publication
"""
from .configs import HEURISTIC_CLOZE_TYPE_QUESTION_MAP
import random
import numpy as np
import nltk


class NoiserParams(object):
    word_shuffle = 3
    word_dropout = 0.1
    word_blank = 0.2
    blank_word = 'BLANKWORD'


def _tokenize(x):
    return nltk.word_tokenize(x)


def _word_shuffle(tokens, noiser_params):
    noise = np.random.uniform(0, noiser_params.word_shuffle, size=(len(tokens),))
    permutation = np.argsort(np.arange(len(tokens)) + noise)
    return list(np.array(tokens)[permutation])


def _word_dropout(tokens, noiser_params):
    keep = np.random.rand(len(tokens), ) >= noiser_params.word_dropout
    return [w for i, w in enumerate(tokens) if keep[i]]


def _word_blank(tokens, noiser_params):
    keep = np.random.rand(len(tokens), ) >= noiser_params.word_blank
    return [w if keep[i] else noiser_params.blank_word for i, w in enumerate(tokens)]


def _add_noise(words, noiser_params):
    words = _word_shuffle(words, noiser_params)
    words = _word_dropout(words, noiser_params)
    words = _word_blank(words, noiser_params)
    return words


def _get_wh_word(cloze, wh_heuristic):
    if wh_heuristic:
        repl = random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[cloze.answer_type])
    else:
        repl = random.choice(['Who', 'What', 'When', 'Where', 'How'])
    return repl


def _add_wh(tokens, cloze, wh_heuristic):
    wh = _get_wh_word(cloze, wh_heuristic)
    return [wh] + tokens


def _replace_mask(cloze, repl):
    return cloze.source_text[:cloze.answer_start] + repl + cloze.source_text[
                                                      cloze.answer_start + len(cloze.answer_text):]


def _add_q_mark_and_fix_spaces(q):
    return q.replace('  ', ' ').rstrip(' ,.') + '?'


def noisy_cloze_translation(cloze, wh_heuristic):
    cloze_no_mask = _replace_mask(cloze, ' ')
    cloze_no_mask_tokens = _tokenize(cloze_no_mask)
    noisy_cloze_tokens = _add_noise(cloze_no_mask_tokens, noiser_params=NoiserParams())
    return _add_q_mark_and_fix_spaces(' '.join(_add_wh(noisy_cloze_tokens, cloze, wh_heuristic)))


def identity_translation(cloze, wh_heuristic):
    repl = _get_wh_word(cloze, wh_heuristic)
    q = _replace_mask(cloze, repl)
    return _add_q_mark_and_fix_spaces(q)


