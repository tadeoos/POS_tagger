# corpus test runner
import nltk
from nltk.util import flatten
import POS_tagger
import il3TT as pd
from tag_map import map_dict



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
	korpusy = pd.podziel(s)
	k_rozw = POS_tagger.Korpus(korpusy[0][:100], 3)
	k_test = POS_tagger.Korpus(korpusy[1], 3)
	k_tren = POS_tagger.Korpus(korpusy[2], 3)
	tagger = POS_tagger.Tagger(k_tren, 3, s)
	# for tag in tagger.korpus.get_zbior_tagow():
	# 	print(tag, tagger.prob_word(tag, 'bien'))
	# print(repr(tagger.backoff_dict.get(('DET', 'ADJ', 'NOUN'), tagger.katz_backoff(2,('ADJ','NOUN')))))
	result = tagger.taguj(k_rozw.get_slowa())
	ev = evaluate(result, k_rozw.get_tagi())
	# tagger.viterbi(k_rozw.get_slowa()[2])
	print('Tagów: {}'.format(len(tagger.korpus.get_zbior_tagow())))
	print('Eval: ', ev)
	# tagger.gt_asrt()
	# return tagger

korp = [[(w, map_dict.get(t, 'UNK'))  for (w,t) in s] for s in nltk.corpus.brown.tagged_sents()]
uni = nltk.corpus.brown.tagged_sents(tagset = "universal")
test1(korp)

test1(uni)
# ss = bound_korp(s, 3)

