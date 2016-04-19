import nltk, re, pprint
from nltk.corpus import brown
from nltk.util import flatten
import il3TT as pd
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.probability import ConditionalProbDist, ELEProbDist, KneserNeyProbDist, LaplaceProbDist, HeldoutProbDist, CrossValidationProbDist
from collections import Counter
import math
from reglin import fit
from time import time

times = {'start':time()}
print('Zaczynam')
def bound_korp(korp, n):
	# assert('*' not in { t for k in k_tren for (word, t) in k })
	for s in korp:
		for i in range(n-1):
			s.insert(0, ('*', '*'))
		s.append(('STOP', 'STOP'))
	return korp



s =  brown.tagged_sents(tagset='universal')
# s =  brown.tagged_sents()
korpusy = pd.podziel(s)


k_rozw = bound_korp(korpusy[0], 3)
k_test = korpusy[1]
k_tren = bound_korp(korpusy[2], 3)

def korp_tags_only(korp):
	k = flatten(korp)
	return k[1::2]

def cfdTags(korp, n = 3):
	corp_tag = korp_tags_only(korp)
	return (nltk.ConditionalFreqDist(
           (g[:n-1], g[n-1])
           for g in nltk.ngrams(corp_tag,n)), 
			nltk.FreqDist(g for g in nltk.ngrams(corp_tag,n)))

set_of_tags = { t for k in k_tren for (word, t) in k if t not in ['*', 'STOP']}
print('Tagów: ', len(set_of_tags))

# tu decyzja o normalizacji do w.lower można w projekcie się tym pobawić.. na treningowym korpusie
# słownik emitowych słów
# keys = tagi
def cfdWords(korp):
	return nltk.ConditionalFreqDist(
           (tag, word.lower())
           for sent in korp
           for (word, tag) in sent)

n=3
tagi_all = [['nic', ['nic']]]
GT_est = ['nic']
for i in range(1, n+1):
	tagi_all.append(cfdTags(k_tren, i))

def good_turing(korp, n):
	tags = tagi_all[n]
	c = Counter(tags[1].values())
	# all_samples = sum([a*b for (a,b) in c.items()])
	# num_of_tags = len(set(korp_tags_only(korp)))
	dict_zR = nRtoZr(c)
	del dict_zR[0]
	d = sorted(dict_zR.items())
	x = [math.log10(key) for (key, value) in d]
	y = [math.log10(value) for (key, value) in d]
	linreg = fit(x, y)

	tur_est = {key : (key+1)*c[key+1]/c[key] for key in c.keys()}
	LGT = {k : (k+1)*10**linreg(math.log10(k+1))/10**linreg(math.log10(k)) for k in c.keys()}
	nowy = {}
	switch = 0
	# print(sorted(c.keys()))
	for key in sorted(c.keys()):
		if (abs(tur_est[key] - LGT[key]) > 1.65 * tur_sd(key, c)) and switch == 0:
			# print('turing estimate: ', key)
			nowy[key] = tur_est[key]
		else:
			switch = 1
			nowy[key] = LGT[key]
	return nowy

def tur_sd(r, c):
	return math.sqrt( ((r+1)**2) * (c[r+1]/(c[r]**2)) * (1 + c[r+1]/c[r]))
def nRtoZr(c):
	prob_dict={}
	# prob_dict[0] = c[1]/sum([a*b for (a,b) in c.items()])
	for key in c.keys():
		prob_dict[key] = c[key]
	prob_dict[0] = 0
	pom = sorted(prob_dict.keys())
	# print(pom)
	for n in range(len(pom)-2):
		q = pom[n]
		r = pom[n+1]
		t = pom[n+2]
		prob_dict[r] = c[r] / (0.5*(t-q))
	last = pom[-1]
	before = pom[-2]
	prob_dict[last] = c[last] / (0.5*((2*last - before) - before))
	# print(sorted(prob_dict.items()))
	return prob_dict


for i in range(1, n+1):
	GT_est.append(good_turing(k_tren, i))

times['po_tablicah'] = time()

def p_star(n, gram):
	mniejsze_gramy = tagi_all[n-1][1]
	if gram[:-1] not in mniejsze_gramy:
		# print('gram not in mniejsze! w p_star!')
		return 0
	dic = GT_est[n]
	tags = tagi_all[n][1]
	return dic[tags[gram]]/mniejsze_gramy[gram[:-1]]

def beta(n, gram):
	tagi = tagi_all[n]
	mniejsze_gramy = tagi_all[n-1][1]
	if gram not in mniejsze_gramy:
		# print('gram not in mniejsze! w becie!', gram)
		return 1
	# print('BETA', [p_star( n, gram+(a,)) for a in tagi[0][gram].keys()])
	return 1 - sum([p_star( n, gram+(a,)) for a in tagi[0][gram].keys()])
def alfa( n, gram):
	tagi = tagi_all[n]
	# print([(a, p_star( n-1, gram[1:]+(a,)), gram[1:]+(a,), n-1) for a in tagi[0][gram].keys()])
	return beta(n,gram)/ (1 - sum([p_star( n-1, gram[1:]+(a,)) for a in tagi[0][gram].keys()]))

def katz_backoff(n, gram):
	# print('jestem w katzu', n, gram)
	if n == 1:
		return p_star( 1, gram)
	mniejsze_gramy = tagi_all[n-1][1]
	dic = GT_est[n]
	tags = tagi_all[n][1]
	if tags[gram] in dic:
		# print('cos znalazlem nie cofam sie')
		# print(gram, tags[gram])
		return dic[tags[gram]]/mniejsze_gramy[gram[:-1]]
	else:
		# print('nie znalazlem nic, alfa: ', alfa( n, gram[:-1]))
		return alfa( n, gram[:-1]) * katz_backoff( n-1, gram[1:])

# print(katz_backoff(3,('OD', 'VBG', 'JJ')))
tren_zdania = [[w.lower()  for (w,t) in s] for s in k_tren]
rozw_zdania = [[w.lower()  for (w,t) in s] for s in k_rozw]
fl = set(flatten(rozw_zdania))
fl2 = set(flatten(tren_zdania))
cnt = 0
for word in fl:
	if word not in fl2:
		cnt +=1

# oszukujemy i uczymy sie słownictwa na całym korpusie
lex = cfdWords(s)
cosa = {k : len(lex[k].keys()) for k in lex.keys()}

times['po_cosach'] = time()
print('jestem po cosa')


def zero_prob(tag):
	# a = lex[tag]
	# b = [(num, word) for (word, num) in a.items()]
	# c = [1 for (n,w) in b if n==1]
	# return sum(c)
	return cosa[tag]/sum(cosa.values())

def prob_word(tag, word):
	# assert word.islower()
	# return lex[tag].get(word, zero_prob(tag))/sum(lex[tag].values())
	# if word.lower() not in fl2:
	# 	return zero_prob(tag)
	return lex[tag][word.lower()]/sum(lex[tag].values())

sent = ['*', '*', 'I', 'was', 'happy', 'today', '.']
def pi(k, u, v, s):
	# print(k, u, v)
	if (k,u,v) == (1, '*', '*'):
		print(k, u, v)
		return 1
	if k>2:
		s = set_of_tags
	else:
		s = ['*']



	# if k==2:
		# return max([prob_word(v, sent[k].lower()) for v in set_of_tags])
	return max([pi(k-1, w, u) * katz_backoff(3, (w,u,v)) * prob_word(v, sent[k].lower()) for w in s])

states = sorted(list(set_of_tags))

def backoff_dict():
	res= {}
	d = tagi_all[3][0]
	for key in d:
		for tag in d[key]:
			k = key + (tag,)
			res[k] = katz_backoff(3, k)
	return res

dfgt = time()
bd = backoff_dict()
times['bd'] = time() - dfgt



def viterbi(sen, bd):
	path_prob = [[] for b in range(len(sen)-2)]

	for z in range(len(states)):
		g = ('*','*',states[z])
		res = bd.get(g, katz_backoff(3,g)) * prob_word(states[z], sen[2].lower())
		# if res == 0:
			# continue
		path_prob[0].append((res, g[1:]))

	# print(path_prob[0])
	# print(max(path_prob[0]))
	# print(path_prob[0].index(max(path_prob[0])))
		# trace[0].append('S')
	# st.append(states[path_prob[0].index(max(path_prob[0]))])
	# print(path_prob[0])
	for i in range(1, len(sen)-3):
		for m in range(len(states)):
			# g = path_prob[i-1][n][1]+(states[m],)
			# check = sum([1 for x in path_prob[i-1] if x[1][-1] == states[m] and x[0]==0])
			# if check == 1:
			# 	print('check!', states[m]) 
			# 	continue 
			p = []
			for n in range(len(states)):
				print(i, m, states[m], path_prob[i-1][n][1]+(states[m],), sen[2+i], prob_word(states[m], sen[2+i]))
				print('slowo: {} tag m: {} tag n {}'.format(sen[2+i], states[m], states[n]))
				gram = path_prob[i-1][n][1][-2:]+(states[m],)
				t1 = time()
				pro = path_prob[i-1][n][0] * bd.get(gram, katz_backoff(3,gram)) * prob_word(states[m], sen[2+i].lower())
				t2 = time()
				print('pro. time: {} score= {}, (i, m , n ) = {} {} {}'.format(t2-t1, pro, i, m, n))
				# if pro == 0:
				# 	continue
				p.append(pro)

			# if len(p)==0:
			# 	continue
			index = p.index(max(p))
			print(p)
			print(index)
			gr = path_prob[i-1][index][1]+(states[m],)
			print(gr)
			# print(gr)
			path_prob[i].append((max(p), gr))
		# print('AAAAA', sen[2+i], path_prob[i])	
	try:
		res = ('*',)+sorted(path_prob[-2])[-1][1]+('STOP',)
	except IndexError:
		print('error')
		print (len(path_prob))
		print(path_prob[-20:])
		print(sen)
		res = []
	return res







def time_this(func, *args):
	t1 = time()
	res = func(*args)
	t2 = time()
	return(t2-t1, res)

s = ['*', '*', 'My', 'dad', 'cut', 'himself', '.', 'STOP']
s2 = ['*', '*', 'the', 'senior', 'policy', 'officer', 'may', 'be', 'moved', 'to', 'think', 'hard', 'about', 'a', 'problem', 'by', 'any', 'of', 'an', 'infinite', 'variety', 'of', 'stimuli', ':', 'an', 'idea', 'in', 'his', 'own', 'head', ',', 'the', 'suggestions', 'of', 'a', 'colleague', ',', 'a', 'question', 'from', 'the', 'secretary', 'or', 'the', 'president', ',', 'a', 'proposal', 'by', 'another', 'department', ',', 'a', 'communication', 'from', 'a', 'foreign', 'government', 'or', 'an', 'american', 'ambassador', 'abroad', ',', 'the', 'filing', 'of', 'an', 'item', 'for', 'the', 'agenda', 'of', 'the', 'united', 'nations', 'or', 'of', 'any', 'other', 'of', 'dozens', 'of', 'international', 'bodies', ',', 'a', 'news', 'item', 'read', 'at', 'the', 'breakfast', 'table', ',', 'a', 'question', 'to', 'the', 'president', 'or', 'the', 'secretary', 'at', 'a', 'news', 'conference', ',', 'a', 'speech', 'by', 'a', 'senator', 'or', 'congressman', ',', 'an', 'article', 'in', 'a', 'periodical', ',', 'a', 'resolution', 'from', 'a', 'national', 'organization', ',', 'a', 'request', 'for', 'assistance', 'from', 'some', 'private', 'american', 'interests', 'abroad', ',', 'et', 'cetera', ',', 'ad', 'infinitum', '.', 'stop']

s1 = s2

print(s1)
t1 = time()
v = viterbi(s1, bd)
t2 = time()
print('time of single viterbi len = {0}, time: {1}'.format(len(s1),  t2-t1))

# print(time_this(viterbi, s1, bd))

print(v)
# print(sorted(v[-2]))
# print(sorted(v[-2])[-1][1])
# tren_zdania = [[w  for (w,t) in s] for s in k_tren]
# rozw_zdania = [[w  for (w,t) in s] for s in k_rozw]

fl = set(flatten(rozw_zdania))
fl2 = set(flatten(lex.values()))
cnt = 0
for word in fl:
	if word not in fl2:
		cnt +=1
print('nowych slow: ', cnt)

rozw_tagi = [[t for (w,t) in s] for s in k_rozw]

print(rozw_tagi[2923])
assert len(rozw_tagi[2923])==147
assert v == rozw_tagi[2923], 'NIE...'
t_v1 = time()
# test = [viterbi(s, bd) for s in rozw_zdania[:100]]
t_v2 = time()
print('Corpus tagging time: ', t_v2-t_v1)
# print('nowe:')
# print(nowe)
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

# print(evaluate(test, rozw_zdania[:100]))
# print(evaluate_sents(test, rozw_zdania[:100]))
# print(rozw_tagi[:6])
# print(test[:6])

# t3 = time()
# bd = backoff_dict()
# t4 = time()

# print('time of bd: ', t4-t3)

# h = []
# for k in hmm2.lex.keys():
# 	h.append((k, len(hmm2.lex[k].keys())))
times['koniec'] = time()
	