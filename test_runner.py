# corpus test runner
import nltk
from nltk.util import flatten
import POS_tagger
import il3TT as pd
from tag_map import map_dict
from time import time



def evaluate(a, b):
	ile = sum([1 for tag in flatten(b)])
	ile_ja = 0
	for i in range(len(b)):
		if len(a[i]) > 0:
			for j in range(len(b[i])):
				if b[i][j] == a[i][j]:
					ile_ja += 1
	return ile_ja/ile

def evaluate_sents(a, b):
	ile = len(b)
	ile_ja = 0
	for i in range(ile):
		if a[i] == []:
			continue
		if b[i] == list(a[i]):
			ile_ja += 1
	return ile_ja/ile


def test1(s, n = 3, zakres = 0):

	korpusy = pd.podziel(s)

	if zakres > 0:
		k_rozw = POS_tagger.Korpus(korpusy[0][:zakres], n)
		k_test = POS_tagger.Korpus(korpusy[1][:zakres], n)
	else:
		k_rozw = POS_tagger.Korpus(korpusy[0], n)
		k_test = POS_tagger.Korpus(korpusy[1], n)		
	k_tren = POS_tagger.Korpus(korpusy[2], n)
	tagger = POS_tagger.Tagger(k_tren, n, s)

	t1 = time()
	result = tagger.taguj(k_test.get_slowa())
	t2 = time()
	ev = evaluate(result, k_test.get_tagi())
	ev_sent = evaluate_sents(result, k_test.get_tagi())

	print('Tagów: {}'.format(len(tagger.korpus.get_zbior_tagow())))
	print('Eval: ', ev)
	print("Eval sents: ", ev_sent)
	print('Czas tagowania (min.): {}'.format((t2-t1)/60))

# pełny tagset Browna - 471 tagów!
full = nltk.corpus.brown.tagged_sents()
# korpus z okrojonym 64-tagowym tagsetem
korp = [[(w, map_dict.get(t, 'UNK'))  for (w,t) in s] for s in nltk.corpus.brown.tagged_sents()]
# korpus z tagsetem uniwersalnym
uni = nltk.corpus.brown.tagged_sents(tagset = "universal")



test1(uni, n = 3)

test1(korp, 3, 100)

# test1(full, 3)

