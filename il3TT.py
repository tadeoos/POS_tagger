import nltk
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents()

# # Proszą o podzielenie pełnego korpus Brown, tj. listy danej poleceniem
# „brown_tagged_sents = brown.tagged_sents()” na 3 podlisty:
# • korpus rozwojowy: zdania o indeksach 8, 18, 28, 38 itd.,
# • korpus testowy: zdania o indeksach 9, 19, 29, 39 itd.,
# • korpus treningowy: reszta korpusu Browna.
# Proszę o przesłanie do poniedziałku (21 marca) do północy 1) jako
# załącznika: programu w Pythonie służącego do dokonania takiego podziału, 2)
# następujących informacji liczbowych o każdym z podkorpusów:
# • liczba zdań
# • liczba słów


def podziel(tagged_sents):
	korpus_rozwojowy, korpus_testowy, korpus_treningowy = [],[],[]
	for i in range(len(tagged_sents)):
		if i%10==8:
			korpus_rozwojowy.append(tagged_sents[i])
		elif i%10==9:
			korpus_testowy.append(tagged_sents[i])
		else:
			korpus_treningowy.append(tagged_sents[i])
	return (korpus_rozwojowy, korpus_testowy, korpus_treningowy)

# nie można użyć slica.. ValueError: slices with steps are not supported by ConcatenatedCorpusView
# korpus_rozwojowy = brown_tagged_sents[8::10]
# korpus_testowy = brown_tagged_sents[9::10]
# del brown_tagged_sents[8::10]
# del brown_tagged_sents[9::10]
# korpus_treningowy = brown_tagged_sents

# korp = podziel(brown_tagged_sents)
# for k in korp:
# 	print('sentences: {} words: {}'.format(len(k), sum(len(w) for w in k)))