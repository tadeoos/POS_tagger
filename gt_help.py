# pomocncize funkcje Good-Turinga
import math 

def tur_sd(r, c):
	return math.sqrt( ((r+1)**2) * (c[r+1]/(c[r]**2)) * (1 + c[r+1]/c[r]))
def nRtoZr(c):
	prob_dict={}
	for key in c.keys():
		prob_dict[key] = c[key]
	prob_dict[0] = 0
	pom = sorted(prob_dict.keys())
	for n in range(len(pom)-2):
		q = pom[n]
		r = pom[n+1]
		t = pom[n+2]
		prob_dict[r] = c[r] / (0.5*(t-q))
	last = pom[-1]
	before = pom[-2]
	prob_dict[last] = c[last] / (0.5*((2*last - before) - before))
	return prob_dict