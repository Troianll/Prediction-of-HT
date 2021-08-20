import numpy as np

def mixpooling(x, pooling = 'MAX'):
	
	output = None
	
	if pooling == 'MAX':
		output = np.max(x, axis = 0)
	elif pooling == 'MEAN':
		output = np.mean(x, axis = 0)
	
	return output



