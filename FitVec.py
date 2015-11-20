import random as rn, numpy as np
def evolveParams(costFunction, vecLength, params=(100,0.01,100), *args):
	initPop, mutRate, numGen = params
	solLen = vecLength
	numWin = int(0.10 * initPop)
	step = 0.01
	bounds = (initPop * solLen) * step * 2
	curPop = np.random.choice(np.arange(-1*bounds,bounds,step=0.01),size=(initPop, solLen),replace=False)
	nextPop = np.zeros((curPop.shape[0], curPop.shape[1]))
	fitVec = np.zeros((initPop, 2))
	for i in range(numGen):
		fitVec = np.array([np.array([x, np.sum(costFunction(*args, curPop[x].T))]) for x in range(initPop)])
		#print(np.sum(fitVec[:,1]))
		winners = np.zeros((numWin, solLen))
		for n in range(len(winners)):
			selected = np.random.choice(range(len(fitVec)), numWin/2, replace=False)
			wnr = np.argmin(fitVec[selected,1])
			winners[n] = curPop[int(fitVec[selected[wnr]][0])]
		nextPop[:len(winners)] = winners
		duplicWin = np.zeros((((initPop - len(winners))),winners.shape[1]))
		for x in range(winners.shape[1]):
			numDups = ((initPop - len(winners))/len(winners))
			duplicWin[:, x] = np.repeat(winners[:, x], numDups, axis=0)
			duplicWin[:, x] = np.random.permutation(duplicWin[:, x])
		nextPop[len(winners):] = np.matrix(duplicWin)
		num_mut_els = nextPop.size * mutRate
		mutated_elements = np.random.random_integers(0, num_mut_els, size=(num_mut_els,))
		for z in mutated_elements:
			nextPop.flat[z] = nextPop.flat[z] * np.float(np.random.normal(0,2,1))
		curPop = nextPop
	best_soln = curPop[np.argmin(fitVec[:,1])]
	#print("Best Sol'n:\n%s\nCost:%s" % (best_soln,np.sum(costFunction(*args, best_soln.T))))

	return best_soln
