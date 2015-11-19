import random as rn, numpy as np
# [Initial population size, mutation rate (=1%), num generations (30), solution length (13), # winners/per gen]
def evolveParams(costFunction, params):
	#initPop, mutRate, numGen, solLen, numWin = 100, 0.01, 500, 17, 20
	initPop, mutRate, numGen, solLen = params
	numWin = int(0.10 * initPop)
	curPop = np.random.choice(np.arange(-15,15,step=0.01),size=(initPop, solLen),replace=False)
	nextPop = np.zeros((curPop.shape[0], curPop.shape[1]))
	fitVec = np.zeros((initPop, 2))
	for i in range(numGen):
		fitVec = np.array([np.array([x, np.sum(costFunction(X, y, curPop[x].T))]) for x in range(initPop)])
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
		mutMatrix = [np.float(np.random.normal(0,2,1)) if rn.random() < mutRate else 1 for x in range(nextPop.size)]
		nextPop = np.multiply(nextPop, np.matrix(mutMatrix).reshape(nextPop.shape))
		curPop = nextPop
	best_soln = curPop[np.argmin(fitVec[:,1])]
	print("Best Sol'n:\n%s\nCost:%s" % (best_soln,np.sum(costFunction(X, y, best_soln.T))))

	return best_soln
