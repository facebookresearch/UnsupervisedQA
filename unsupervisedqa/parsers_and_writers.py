# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Module to handle reading, (de)seriaizing and dumping data
"""
import json
import attr
from .data_classes import Cloze, Paragraph
import hashlib


def clozes2squadformat(clozes, out_fobj):
    assert all([c.question_text is not None for c in clozes]), 'Translate these clozes firse, some dont have questions'
    data = {cloze.paragraph.paragraph_id: {'context': cloze.paragraph.text, 'qas': []} for cloze in clozes}
    for cloze in clozes:
        qas = data[cloze.paragraph.paragraph_id]
        qas['qas'].append({
            'question': cloze.question_text, 'id': cloze.cloze_id,
            'answers': [{'text': cloze.answer_text, 'answer_start': cloze.answer_start}]
        })
    squad_dataset = {
        'version': 1.1,
        'data': [{'title': para_id, 'paragraphs': [payload]} for para_id, payload in data.items()]
    }
    json.dump(squad_dataset, out_fobj)


def _parse_attr_obj(cls, serialized):
    return cls(**json.loads(serialized))


def dumps_attr_obj(obj):
    return json.dumps(attr.asdict(obj))


def parse_clozes(fobj):
    for serialized in fobj:
        if serialized.strip('\n') != '':
            yield _parse_attr_obj(Cloze, serialized)


def dump_clozes(clozes, fobj):
    for cloze in clozes:
        fobj.write(dumps_attr_obj(cloze))
        fobj.write('\n')


def _get_paragraph_id(text):
    return hashlib.sha1(text.encode()).hexdigest()


def parse_paragraphs_from_txt(fobj):
    for paragraph_text in fobj:
        para_text = paragraph_text.strip('\n')
        if para_text != '':
           yield Paragraph(
               paragraph_id=_get_paragraph_id(para_text),
               text=para_text
           )


def parse_paragraphs_from_jsonl(fobj):
    for serialized in fobj:
        if serialized.strip('\n') != '':
            yield _parse_attr_obj(Paragraph, serialized)
