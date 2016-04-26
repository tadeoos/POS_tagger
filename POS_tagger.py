# Part of speech tagger using Brown Corpus from NLTK package & HMM classifier.

import nltk
import math
from nltk.util import flatten
from nltk.probability import ConditionalFreqDist, FreqDist
from collections import Counter
from reglin import fit
from time import time
from decimal import Decimal
from tag_map import map_dict
from gt_help import *

class Korpus:
	def __init__(self, nltk_corpus, n):
		self.korpusik = self.bound(nltk_corpus, n)
		self.slowa = [[w  for (w,t) in s] for s in self.korpusik]
		self.tagi = [[t  for (w,t) in s] for s in self.korpusik]
		self.zbior_tagow = { t for t in flatten(self.tagi) if t not in ['*', 'STOP'] }

	def bound(self, korpus, n):
	# assert('*' not in { t for k in k_tren for (word, t) in k })
		return [[('*', '*')]*(n-1) + s + [('STOP', 'STOP')] for s in korpus]

	def get_slowa(self):
		return self.slowa
	def get_tagi(self):
		return self.tagi
	def get_zbior_tagow(self):
		return self.zbior_tagow

class Tagger:
	def __init__(self, korpusik, n, lex):
		self.korpus = korpusik
		self.n = n
		self.tagi_all = [ ['nic', ['nic']] if i == 0 else self.cfdTags(i) for i in range(self.n + 1) ]
		self.GT_est = [ ['nic'] if i ==0 else self.good_turing(i) for i in range(self.n + 1) ]
		self.states = sorted(list(self.korpus.get_zbior_tagow()))
		self.backoff_dict = { key : self.katz_backoff(self.n, key) for key in self.tagi_all[self.n][1].keys()}
		self.lex = self.cfdWords(lex)

	def cfdTags(self, n):
		# n = self.n
		corp_tag = flatten(self.korpus.tagi)
		return (nltk.ConditionalFreqDist(
        	   (g[:n-1], g[n-1])
        	   for g in nltk.ngrams(corp_tag,n)), 
				nltk.FreqDist(g for g in nltk.ngrams(corp_tag,n)))

	def cfdWords(self, korp):
		return nltk.ConditionalFreqDist(
    	       (tag, word.lower())
    	       for sent in korp
    	       for (word, tag) in sent)

	def gt_asrt(self):
		for i in range(1,self.n+1):
			if 0 in self.GT_est[i].values():
				print('ooooooohhh')

	def good_turing(self, n):
		tags = self.tagi_all[n]
		c = Counter(tags[1].values())
		wszystko = (len(self.korpus.get_zbior_tagow())+2) ** n - len(tags[1].keys())
		dict_zR = nRtoZr(c)
		# print('dzr: {}'.format(dict_zR))
		dict_zR[0] = wszystko
		print(wszystko)
		print('dzr: {}'.format(dict_zR))
		d = sorted(dict_zR.items())
		x = [math.log10(key) for (key, value) in d]
		y = [math.log10(value) for (key, value) in d]
		linreg = fit(x, y)
		tur_est = {key : (key+1)*c[key+1]/c[key] for key in c.keys()}
		LGT = {k : (k+1)*10**linreg(math.log10(k+1))/10**linreg(math.log10(k)) for k in c.keys()}
		nowy = {}
		switch = 0
		for key in sorted(c.keys()):
			if ((abs(tur_est[key] - LGT[key]) > 1.65 * tur_sd(key, c)) and switch == 0 and tur_est[key] > 0) or (tur_est[key] < LGT[key]):
				nowy[key] = tur_est[key]
			else:
				switch = 1
				nowy[key] = LGT[key]

#
		# print('nowy ', nowy)
		# if 0 in nowy.values():
			# print('nowy items:', nowy.items())
			# print('tur_est:', tur_est.items())
			# print('LGT:', LGT.items())
		return nowy

	def p_star(self, n, gram):
		if n==1:
			if self.tagi_all[1][1][gram] == 0:
				print('P_star nie znajduje countu dla tego tagu...')
			return Decimal.from_float(self.tagi_all[1][1][gram]/self.tagi_all[1][1].N())
		mniejsze_gramy = self.tagi_all[n-1][1]
		if gram[:-1] not in mniejsze_gramy:
			# print('gram not in mniejsze! w p_star!')
			return Decimal.from_float(0)
		dic = self.GT_est[n]
		tags = self.tagi_all[n][1]
		return Decimal.from_float(dic[tags[gram]]/mniejsze_gramy[gram[:-1]])

	def beta(self, n, gram):
		tagi = self.tagi_all[n]
		mniejsze_gramy = self.tagi_all[n-1][1]
		if gram not in mniejsze_gramy:
			return Decimal.from_float(1)
		return Decimal.from_float(1) - sum([self.p_star( n, gram + (a,) ) for a in tagi[0][gram].keys()])
	def alfa(self, n, gram):
		tagi = self.tagi_all[n]
		return self.beta(n,gram)/(Decimal.from_float(1) - sum([self.p_star(n-1, gram[1:]+(a,)) for a in tagi[0][gram].keys()]))

	def katz_backoff(self, n, gram):
		if n == 1:
			print('cofam do 1')
			return self.p_star( 1, gram)
		mniejsze_gramy = self.tagi_all[n-1][1]
		dic = self.GT_est[n]
		tags = self.tagi_all[n][1]
		if tags[gram] in dic:
			# print(gram)
			# print(dic[tags[gram]])
			# print(mniejsze_gramy[gram[:-1]])
			return Decimal.from_float( dic[tags[gram]] / mniejsze_gramy[gram[:-1]] )
		else:
			# print('alfa: ', self.alfa( n, gram[:-1]))
			return self.alfa( n, gram[:-1]) * self.katz_backoff( n-1, gram[1:])

	def prob_word(self, tag, word):
		print('PROBWORD: tag {} word {} self.lex[tag][word.lower()] {} sum(self.lex[tag].values()) {}'.format(tag,word,self.lex[tag][word.lower()],sum(self.lex[tag].values())))
		return Decimal.from_float(self.lex[tag][word.lower()]/sum(self.lex[tag].values()))

	def viterbi(self, sen):
		print(sen)
		path_prob = [[] for b in range(len(sen)-2)]
		for z in range(len(self.states)):
			g = ('*','*',self.states[z])
			a = self.backoff_dict.get(g, self.katz_backoff(3,g))
			res =  a * self.prob_word(self.states[z], sen[2].lower())
			if res == 0:
				continue
			path_prob[0].append( (res, g[1:]) )

		for i in range(1, len(sen)-3):
			for m in range(len(self.states)):
				# print(i, m, self.states[m], sen[2+i], prob_word(self.states[m], sen[2+i]))
				
				p = []
				for n in range(len(path_prob[i-1])):
					print('slowo: {} tag n: {} tag m {}'.format(sen[2+i], self.states[n], self.states[m]))
					gram = path_prob[i-1][n][1][-2:]+(self.states[m],)
					print('gram: {} prob dotychczasowe dla n: {} backoff: {} prob slowa: {}'.format(gram, path_prob[i-1][n][0] , self.backoff_dict.get(gram, self.katz_backoff(3,gram)) , self.prob_word(self.states[m], sen[2+i].lower()) ))
					pro = path_prob[i-1][n][0] * self.backoff_dict.get(gram, self.katz_backoff(3,gram)) * self.prob_word(self.states[m], sen[2+i].lower())
					print('Pro: {}, n: {}'.format(pro, n))
					if pro > 0:
						p.append(pro)
					else:
						continue
					

				print('p: ', p)
				if len(p)==0:
					continue
				index = p.index(max(p))
				gr = path_prob[i-1][index][1]+(self.states[m],)
				print('dodaje p! ', p)
				path_prob[i].append((max(p), gr))

		try:
			res = ('*',)+sorted(path_prob[-2])[-1][1]+('STOP',)
		except IndexError:
			print('error')
			print (len(path_prob))
			print(path_prob[-20:])
			print(sen)
			res = []
		return res

	def taguj(self, korp):
		return [self.viterbi(s) for s in korp]
