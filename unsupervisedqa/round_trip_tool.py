import attr
from unsupervisedqa.unmt_translation import get_params as get_unmt_params, convert_to_text, \
    check_all_data_params, check_mt_model_params, load_data, build_mt_model, TrainerMT, EvaluatorMT, _get_model_paths
import torch
import tempfile
from .configs import UNMT_MODEL_SENTENCE_NE, \
    UNMT_MODEL_SENTENCE_NP, UNMT_MODEL_SUBCLAUSE_NE, \
    UNMT_MODEL_SUBCLAUSE_NE_WH_HEURISTIC, \
    PATH_TO_UNMT, MOSES_DIR, FASTBPE_DIR, N_THREADS_PREPRO, CLOZE_MASKS, UNMT_BATCH_SIZE, UNMT_BEAM_SIZE
import os
import subprocess
import sys
sys.path.append(PATH_TO_UNMT) # simple hack on the path to import Unsupervised NMT functionality
from src.data.dictionary import SPECIAL_WORDS, Dictionary
from src.data.loader import load_vocab, check_dictionaries, create_word_masks, MonolingualDataset


@attr.s()
class EnvironmentParams(object):
    exp_name = attr.ib()
    dump_path = attr.ib()
    fasttext_vectors_path = attr.ib()
    checkpoint_path = attr.ib()
    bpe_codes_path = attr.ib()
    vocab_path = attr.ib()
    cloze_lang = attr.ib()
    question_lang = attr.ib()
    ne_answers = attr.ib()
    wh_heuristic = attr.ib()


def get_environment_params():
    subclause_clozes = False
    ne_answers = True
    wh_heuristic = False
    checkpoint_path, vocab_path, bpe_codes_path, fasttext_vectors_path = _get_model_paths(
        subclause_clozes, ne_answers, wh_heuristic)
    return EnvironmentParams(
        exp_name='environment',
        dump_path='dumped', #TODO
        checkpoint_path=checkpoint_path,
        bpe_codes_path=bpe_codes_path,
        fasttext_vectors_path=fasttext_vectors_path,
        vocab_path=vocab_path,
        cloze_lang='cloze',
        question_lang='question',
        ne_answers=ne_answers,
        wh_heuristic=wh_heuristic,
    )


def error_if_not_setup(func):
    def _err(*args, **kwargs):
        assert args[0].has_been_setup, 'Environment not setup, call environment.setup()'
        return func(*args, **kwargs)
    return _err


class Environment:

    def __init__(self, params):
        self.params = params
        self.unmt_params = None
        self.evaluator = None
        self.scratch_dir = None
        self.data = None
        self.dummy_data_path = None
        self.has_been_setup = False

    def set_up(self):
        # TODO: handle dummy data_path here
        self.scratch_dir = tempfile.TemporaryDirectory()
        self.dummy_data_path = os.path.join(self.scratch_dir.name, 'dummy.cloze.tok.bpe.pth')
        with open(self.dummy_data_path.replace('.pth', ''), 'w') as f:
            f.write('the\nthe\n')
        Dictionary.index_data(
            self.dummy_data_path.replace('.pth', ''),
            self.dummy_data_path,
            Dictionary.read_vocab(self.params.vocab_path)
        )
        self.unmt_params = get_unmt_params(
            self.params.exp_name,
            self.scratch_dir.name,
            self.dummy_data_path,
            self.dummy_data_path,
            self.dummy_data_path,
            self.params.fasttext_vectors_path,
            self.params.checkpoint_path
        )
        check_all_data_params(self.unmt_params)
        check_mt_model_params(self.unmt_params)
        self.data = load_data(self.unmt_params, mono_only=True)
        encoder, decoder, discriminator, lm = build_mt_model(self.unmt_params, self.data)
        trainer = TrainerMT(encoder, decoder, discriminator, lm, self.data, self.unmt_params)
        trainer.reload_checkpoint()
        trainer.test_sharing()  # check parameters sharing
        self.evaluator = EvaluatorMT(trainer, self.data, self.unmt_params)
        self.evaluator.encoder.eval()
        self.evaluator.decoder.eval()
        self.has_been_setup = True

    def teardown(self):
        self.scratch_dir.cleanup()

    @error_if_not_setup
    def _prepro(self, strings):

        tokenizer = os.path.join(MOSES_DIR, 'scripts', 'tokenizer', 'tokenizer.perl')
        norm = os.path.join(MOSES_DIR, 'scripts', 'tokenizer', 'normalize-punctuation.perl')
        fast_bpe = os.path.join(FASTBPE_DIR, 'fast')

        cmd = f'{norm} -l en | {tokenizer} -l en -no-escape -threads {N_THREADS_PREPRO} | ' \
            f'{fast_bpe} applybpe_stream {self.params.bpe_codes_path} {self.params.vocab_path}'
        p = subprocess.run(cmd, shell=True, input='\n'.join(strings).encode(), check=True, capture_output=True)
        bpe_strings= [s for s in p.stdout.decode().split('\n') if s!='']

        assert len(bpe_strings) == len(strings)
        dico = Dictionary.read_vocab(self.params.vocab_path)

        positions, sentences, unk_words = [], [], {}

        # index sentences
        for i, line in enumerate(bpe_strings):
            s = line.rstrip().split()
            # skip empty sentences
            if len(s) == 0:
                print("Empty sentence in line %i." % i)
            # index sentence words
            count_unk = 0
            indexed = []
            for w in s:
                word_id = dico.index(w, no_unk=False)
                if word_id < 4 + SPECIAL_WORDS and word_id != dico.unk_index:
                    # logger.warning('Found unexpected special word "%s" (%i)!!' % (w, word_id))
                    continue
                indexed.append(word_id)
                if word_id == dico.unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            # add sentence
            positions.append([len(sentences), len(sentences) + len(indexed)])
            sentences.extend(indexed)
            sentences.append(-1)

        return {
            'dico': dico,
            'positions': torch.LongTensor(positions).numpy(),
            'sentences': torch.LongTensor(sentences),
            'unk_words': unk_words,
        }

    @error_if_not_setup
    def strings_to_mono_dataset(self, input_strings, lang):
        mono_data = self._prepro(input_strings)
        return MonolingualDataset(
            mono_data['sentences'], mono_data['positions'],
            mono_data['dico'], self.unmt_params.lang2id[lang], self.unmt_params)

    @error_if_not_setup
    def translate(self, input_strings, lang1, lang2):
        with torch.no_grad():
            mono_dataset = self.strings_to_mono_dataset(input_strings, lang1)
            lang1_id, lang2_id = self.unmt_params.lang2id[lang1], self.unmt_params.lang2id[lang2]

            translations = []
            for i, (sent1, len1) in enumerate(mono_dataset.get_iterator(shuffle=False, group_by_size=False)()):
                encoded = self.evaluator.encoder(sent1.cuda(), len1, lang1_id)
                s2_, l2_, _ = self.evaluator.decoder.generate(encoded, lang2_id)
                translations += convert_to_text(s2_, l2_, self.evaluator.dico[lang2], lang2_id, self.unmt_params)
        translations = [t.replace('@@ ', '') for t in translations]
        return translations

    @error_if_not_setup
    def round_trip(self, input_cloze_strings):
        questions = self.translate(input_cloze_strings, self.params.cloze_lang, self.params.question_lang)
        reconstructions = self.translate(questions, self.params.question_lang, self.params.cloze_lang)
        return questions, reconstructions

    @error_if_not_setup
    def get_reward(self, input_cloze_strings, reward_func):
        question_strings, reconstruction_strings = self.round_trip(input_cloze_strings)
        return reward_func(input_cloze_strings, question_strings, reconstruction_strings)


if __name__ == '__main__':
    env_params = get_environment_params()
    env = Environment(env_params)

    def toy_reward_function(input_cloze_strings, question_strings, reconstruction_strings):
        return 1.0


    env.set_up()

    transl, reverse_transl = env.round_trip(['Barack Obama was born in PLACEMASK.', ])
    print('input:', 'Barack Obama was born in PLACEMASK.')
    print('translation:', transl[0])
    print('reconstruction:', reverse_transl[0])

    r = env.get_reward(['Barack Obama was born in PLACEMASK', ], toy_reward_function)
    print('reward:', r)
    env.teardown()
