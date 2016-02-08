import numpy as np
import cPickle

whole = u""
for line in open("html.txt", "r"):
	t = line.decode('utf-8', 'ignore')
	if len(t) > 0: whole += t[:]
	if len(whole) > 5000000: break
	if len(whole) % 1000 == 0: print len(whole)

step = 200
segments = []
for i in range(len(whole) / step):
	segments.append(whole[i * step:(i + 1) * step])
	if i % 1000 == 0: print i

dictionary = []
dictionary_count = {}
for i in range(len(segments)):
	for j in range(len(segments[i])):
		x = segments[i][j]
		if not x in dictionary_count:
			dictionary_count[x] = 1
		else:
			dictionary_count[x] += 1	
	if i % 100 == 0: print i

dictionary = []
for i in dictionary_count:
	if dictionary_count[i] > 0: dictionary += i

dictionary.append(u"\u951f")

for i in dictionary:
	print i.encode("GBK", "ignore"),

matrix = []
for i in range(len(segments)):
	vector = []
	for j in range(len(segments[i])):
		x = segments[i][j]
		if x in dictionary:
			vector.append(dictionary.index(x))
		else:
			vector.append(len(dictionary) - 1)	
	matrix.append(vector)
	if i % 100 == 0: print i

npmatrix = np.asarray(matrix).T
print npmatrix.shape
npmatrix.dump("feature.npy")
print npmatrix

cPickle.dump(dictionary, open("dictionary.pkl", "w"))