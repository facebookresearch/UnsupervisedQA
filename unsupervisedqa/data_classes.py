# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Data classes used in UnsupervisedQA are defined here
"""
import attr


@attr.s(hash=True)
class Paragraph:
    paragraph_id = attr.ib()
    text = attr.ib()


@attr.s(hash=True)
class Cloze:
    cloze_id = attr.ib()
    paragraph = attr.ib()
    source_text = attr.ib()
    source_start = attr.ib()
    cloze_text = attr.ib()
    answer_text = attr.ib()
    answer_start = attr.ib()
    constituency_parse = attr.ib()
    root_label = attr.ib()
    answer_type = attr.ib()
    question_text = attr.ib()
