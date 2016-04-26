# corpus test runner
import nltk
from nltk.util import flatten
import POS_tagger
import il3TT as pd



def evaluate(a, b):
	ile = sum([1 for tag in flatten(b)])
	ile_ja = 0
	for i in range(len(b)):
		if len(b[i]) > 0:
			for j in range(len(b[i])):
				if b[i][j] == a[i][j]:
					ile_ja += 1
	return ile_ja/ile

def evaluate_sents(a, b):
	ile = len(b)
	ile_ja = 0
	for i in range(ile):
		if b[i] == list(a[i]):
			ile_ja += 1
	return ile_ja/ile


def test1(s):
	korpusy = pd.podziel(nltk.corpus.brown.tagged_sents(tagset = "universal"))
	k_rozw = POS_tagger.Korpus(korpusy[0][2911:2914], 3)
	k_test = POS_tagger.Korpus(korpusy[1], 3)
	k_tren = POS_tagger.Korpus(korpusy[2], 3)
	tagger = POS_tagger.Tagger(k_tren, 3, s)
	# for tag in tagger.korpus.get_zbior_tagow():
	# 	print(tag, tagger.prob_word(tag, 'bien'))
	print(repr(tagger.backoff_dict.get(('PRON', 'NUM', 'NUM'), tagger.katz_backoff(2,( 'NUM', 'NUM')))))
	# result = tagger.taguj(k_rozw.get_slowa())
	# ev = evaluate(result, k_rozw.get_tagi())
	# tagger.viterbi(k_rozw.get_slowa()[2])
	# print(ev)
	# tagger.gt_asrt()
	return tagger


tagr = test1(nltk.corpus.brown.tagged_sents(tagset = "universal"))
# ss = bound_korp(s, 3)

