# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Wrapper for UnsupervisedNMT inference-time functionality
"""
import attr
import os
from .configs import UNMT_MODEL_SENTENCE_NE, \
    UNMT_MODEL_SENTENCE_NP, UNMT_MODEL_SUBCLAUSE_NE, \
    UNMT_MODEL_SUBCLAUSE_NE_WH_HEURISTIC, \
    PATH_TO_UNMT, MOSES_DIR, FASTBPE_DIR, N_THREADS_PREPRO, CLOZE_MASKS, UNMT_BATCH_SIZE, UNMT_BEAM_SIZE
from .baseline_translators import identity_translation, noisy_cloze_translation
import subprocess
import tempfile
import sys
sys.path.append(PATH_TO_UNMT) # simple hack on the path to import Unsupervised NMT functionality
from src.data.loader import check_all_data_params, load_data
from src.utils import restore_segmentation
from src.model import check_mt_model_params, build_mt_model
from src.trainer import TrainerMT
from src.evaluator import EvaluatorMT
import torch


def _tokenize_file(input_path, output_path):
    tokenizer = os.path.join(MOSES_DIR, 'scripts', 'tokenizer', 'tokenizer.perl')
    norm = os.path.join(MOSES_DIR, 'scripts', 'tokenizer', 'normalize-punctuation.perl')
    cmd = f'cat {input_path} | {norm} -l en | {tokenizer} -l en -no-escape -threads {N_THREADS_PREPRO} > {output_path}'
    subprocess.check_call(cmd, shell=True)


def _apply_bpe(input_path, output_path, bpe_codes_path, vocab_path):
    fast_bpe = os.path.join(FASTBPE_DIR, 'fast')
    cmd = f'{fast_bpe} applybpe {output_path} {input_path} {bpe_codes_path} {vocab_path}'
    subprocess.check_call(cmd, shell=True)


def _binarize_data(vocab_path, input_path):
    prepro = os.path.join(PATH_TO_UNMT, 'preprocess.py')
    cmd = f'{prepro} {vocab_path} {input_path}'
    subprocess.check_call(cmd, shell=True)


def _dump_clozes_for_translation(clozes, dump_path, wh_heuristic):

    def _wh_heurisistic(cloze):
        cloze_mask = CLOZE_MASKS[cloze.answer_type]
        cloze_text = cloze_mask + ' ' + cloze.cloze_text.replace(cloze_mask, 'MASK')
        return cloze_text

    with open(dump_path, 'w') as fobj:
        for c in clozes:
            cloze_text = _wh_heurisistic(c) if wh_heuristic else c.cloze_text
            fobj.write(cloze_text)
            fobj.write('\n')


def preprocessing(clozes, directory, vocab_path, bpe_codes_path, wh_heuristic):
    raw_cloze_file = os.path.join(directory, 'dev.cloze')
    tok_cloze_file = os.path.join(directory, 'dev.cloze.tok')
    bpe_cloze_file = os.path.join(directory, 'dev.cloze.tok.bpe')
    binarized_cloze_file = os.path.join(directory, 'dev.cloze.tok.bpe.pth')

    _dump_clozes_for_translation(clozes, raw_cloze_file, wh_heuristic)
    _tokenize_file(raw_cloze_file, tok_cloze_file)
    _apply_bpe(tok_cloze_file, bpe_cloze_file, bpe_codes_path, vocab_path)
    _binarize_data(vocab_path, bpe_cloze_file)
    return binarized_cloze_file


def _get_model_paths(subclause_clozes, ne_answers, wh_heuristic):
    if subclause_clozes and ne_answers and wh_heuristic:
        data_dir = UNMT_MODEL_SUBCLAUSE_NE_WH_HEURISTIC
        model_dir = UNMT_MODEL_SUBCLAUSE_NE_WH_HEURISTIC
    elif subclause_clozes and ne_answers and (not wh_heuristic):
        data_dir = UNMT_MODEL_SUBCLAUSE_NE
        model_dir = UNMT_MODEL_SUBCLAUSE_NE
    elif (not subclause_clozes) and ne_answers and (not wh_heuristic):
        data_dir = UNMT_MODEL_SENTENCE_NE
        model_dir = UNMT_MODEL_SENTENCE_NE
    elif (not subclause_clozes) and (not ne_answers) and (not wh_heuristic):
        data_dir = UNMT_MODEL_SENTENCE_NP
        model_dir = UNMT_MODEL_SENTENCE_NP
    else:
        raise Exception('This model configuration doesnt exist')
    checkpoint_path = os.path.join(model_dir, 'periodic-20.pth')
    vocab_path = os.path.join(data_dir, 'vocab.cloze-question.60000')
    bpe_codes_path = os.path.join(data_dir, 'bpe_codes')
    fasttext_vectors_path = os.path.join(data_dir, 'all.cloze-question.60000.vec')
    return checkpoint_path, vocab_path, bpe_codes_path, fasttext_vectors_path


def _associate_questions_to_clozes(clozes, translation_output_file, wh_heuristic):

    def _clean_wh_heurisistic(question_text):
        return ' '.join(question_text.split(' ')[1:])

    clozes_with_questions = []
    translations = []

    for line in open(translation_output_file):
        if line.strip() != '':
            inp, outp = line.strip().split('\t')
            translations.append(_clean_wh_heurisistic(outp) if wh_heuristic else outp)

    assert len(clozes) == len(translations), "mismatch between number of clozes and translations"
    for c, q in zip(clozes, translations):
        c_with_q = attr.evolve(c, question_text=q)
        clozes_with_questions.append(c_with_q)
    return clozes_with_questions


class Params:
    pass


def get_params(
        exp_name,
        dump_path,
        cloze_train_path,
        question_train_path,
        cloze_test_path,
        fasttext_vectors_path,
        checkpoint_path
):
    params = Params()
    params.exp_name = exp_name
    params.exp_id = ""
    params.dump_path = dump_path
    params.save_periodic = False
    params.seed = -1
    params.emb_dim = 512
    params.n_enc_layers = 4
    params.n_dec_layers = 4
    params.hidden_dim = 512
    params.lstm_proj = False
    params.dropout = 0
    params.label_smoothing = 0
    params.attention = True
    params.transformer = True
    params.transformer_ffn_emb_dim = 2048
    params.attention_dropout = 0
    params.relu_dropout = 0
    params.encoder_attention_heads = 8
    params.decoder_attention_heads = 8
    params.encoder_normalize_before = False
    params.decoder_normalize_before = False
    params.share_lang_emb = True
    params.share_encdec_emb = False
    params.share_decpro_emb = False
    params.share_output_emb = True
    params.share_lstm_proj = False
    params.share_enc = 3
    params.share_dec = 3
    params.word_shuffle = 0
    params.word_dropout = 0
    params.word_blank = 0
    params.dis_layers = 3
    params.dis_hidden_dim = 128
    params.dis_dropout = 0
    params.dis_clip = 0
    params.dis_smooth = 0
    params.dis_input_proj = False
    params.langs = "cloze,question"
    params.vocab = ""
    params.vocab_min_count = 0
    params.mono_dataset = f"cloze:{cloze_train_path},,{cloze_test_path};question:{question_train_path},,"
    params.para_dataset = ""
    params.back_dataset = ""
    params.n_mono = -1
    params.n_para = 0
    params.n_back = 0
    params.max_len = 175
    params.max_vocab = -1
    params.n_dis = 0
    params.mono_directions = 'cloze,question'
    params.para_directions = ""
    params.pivo_directions = "cloze-question-cloze,question-cloze-question"
    params.back_directions = ""
    params.otf_sample = -1
    params.otf_backprop_temperature = -1
    params.otf_sync_params_every = 1000
    params.otf_num_processes = 0
    params.otf_update_enc = True
    params.otf_update_dec = True
    params.lm_before = 0
    params.lm_after = 0
    params.lm_share_enc = 0
    params.lm_share_dec = 0
    params.lm_share_emb = False
    params.lm_share_proj = False
    params.batch_size = UNMT_BATCH_SIZE
    params.group_by_size = True
    params.lambda_xe_mono = "0:1,100000:0.1,300000:0"
    params.lambda_xe_para = "0"
    params.lambda_xe_back = "0"
    params.lambda_xe_otfd = "1"
    params.lambda_xe_otfa = "0"
    params.lambda_dis = "0"
    params.lambda_lm = "0"
    params.enc_optimizer = "adam,lr=0.0001"
    params.dec_optimizer = "enc_optimizer"
    params.dis_optimizer = "rmsprop,lr=0.0005"
    params.clip_grad_norm = 5
    params.epoch_size = 10
    params.max_epoch = 100000
    params.stopping_criterion = ""
    params.pretrained_emb = fasttext_vectors_path
    params.pretrained_out = True
    params.reload_model = checkpoint_path
    params.reload_enc = True
    params.reload_dec = True
    params.reload_dis = False
    params.freeze_enc_emb = False
    params.freeze_dec_emb = False
    params.eval_only = False
    params.beam_size = UNMT_BEAM_SIZE
    params.length_penalty = 1.0
    return params


def convert_to_text(batch, lengths, dico, lang_id, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()
    bos_index = params.bos_index[lang_id]

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == bos_index).sum() == bs
    assert (batch == params.eos_index).sum() == bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


def perform_translation(input_file_path,
                        translation_directory,
                        cloze_train_path,
                        question_train_path,
                        fasttext_vectors_path,
                        checkpoint_path
                        ):
    params = get_params(
        exp_name='translation',
        dump_path=translation_directory,
        cloze_train_path=cloze_train_path,
        question_train_path=question_train_path,
        cloze_test_path=input_file_path,
        fasttext_vectors_path=fasttext_vectors_path,
        checkpoint_path=checkpoint_path,
    )

    # check parameters
    assert params.exp_name
    check_all_data_params(params)
    check_mt_model_params(params)
    data = load_data(params, mono_only=True)
    encoder, decoder, discriminator, lm = build_mt_model(params, data)
    # initialize trainer / reload checkpoint / initialize evaluator
    trainer = TrainerMT(encoder, decoder, discriminator, lm, data, params)
    trainer.reload_checkpoint()
    trainer.test_sharing()  # check parameters sharing
    evaluator = EvaluatorMT(trainer, data, params)

    with torch.no_grad():
        lang1, lang2 = 'cloze', 'question'

        evaluator.encoder.eval()
        evaluator.decoder.eval()
        lang1_id = evaluator.params.lang2id[lang1]
        lang2_id = evaluator.params.lang2id[lang2]

        translations = []
        dataset = evaluator.data['mono'][lang1]['test']
        dataset.batch_size = params.batch_size

        for i, (sent1, len1) in enumerate(dataset.get_iterator(shuffle=False, group_by_size=False)()):
            encoded = evaluator.encoder(sent1.cuda(), len1, lang1_id)
            sent2_, len2_, _ = evaluator.decoder.generate(encoded, lang2_id)
            lang1_text = convert_to_text(sent1, len1, evaluator.dico[lang1], lang1_id, evaluator.params)
            lang2_text = convert_to_text(sent2_, len2_, evaluator.dico[lang2], lang2_id, evaluator.params)
            translations += zip(lang1_text, lang2_text)

        # export sentences to hypothesis file and restore BPE segmentation
        out_name = os.path.join(translation_directory, 'output_translations.txt')
        with open(out_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join(['\t'.join(st) for st in translations]) + '\n')
        restore_segmentation(out_name)

    return out_name


def get_unmt_questions_for_clozes(clozes,
                                  subclause_clozes,
                                  ne_answers,
                                  wh_heuristic,
                                  ):
    checkpoint_path, vocab_path, bpe_codes_path, fasttext_vectors_path = \
        _get_model_paths(subclause_clozes, ne_answers, wh_heuristic)

    with tempfile.TemporaryDirectory() as tempdir:
        translation_input_path = preprocessing(clozes, tempdir, vocab_path, bpe_codes_path, wh_heuristic)
        translation_output_path = perform_translation(
            translation_input_path,
            tempdir,
            translation_input_path,
            translation_input_path,
            fasttext_vectors_path,
            checkpoint_path
        )
        clozes_with_questions = _associate_questions_to_clozes(clozes, translation_output_path, wh_heuristic)

    return clozes_with_questions
