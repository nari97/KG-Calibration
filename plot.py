import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale

'''
Sigmoid vs minmax
'''
A = 0.5
B = 0.5
x = np.asarray(sorted([2, 2.2, 3, 3.4, 3.5,3.7, 18.3, 19.1, 19.2, 19.8, 19.99, 20]))
sigmoidone = A*x+B
minmaxone = A*minmax_scale(x)+B
tanhone = A*(2*minmax_scale(x)-1)+B

print(1/(1 + np.exp(-sigmoidone)))
print(1/(1 + np.exp(-minmaxone)))
print(1/(1 + np.exp(-tanhone)))

#plt.scatter(x, 1/(1 + np.exp(-sigmoidone)), marker='^', label="Direct scores z")
#plt.scatter(x, 1/(1 + np.exp(-minmaxone)),marker='o', label="Min-max of z with [0,1] range")
#plt.scatter(x, 1/(1 + np.exp(-tanhone)),marker='x', label="Min-max of z with [-1,1] range")
#plt.xlabel("Original triple scores")
#plt.legend()
#plt.ylabel("Triple scores after calibrationn")
#plt.title("Issues with sigmoid for preprocessing")
#plt.plot()

plt.ylim([0,1.1])
#plt.savefig("preprocessing.pdf", bbox_inches='tight')

'''

minmax norm - different values of a

'''
def minmax(array,a,b, minimum, maximum):
  return (((b-a)*(array-minimum))/(maximum-minimum))+a

scores = x#sorted(np.random.rand(20)*100)
scores2 = minmax(np.array(scores), -2, 2, min(scores), max(scores))
scores5 = minmax(np.array(scores), -5, 5, min(scores), max(scores))
scores1 = minmax(np.array(scores), -1, 1, min(scores), max(scores))
scores3 = minmax(np.array(scores), -3, 3, min(scores), max(scores))
scores4 = minmax(np.array(scores), -4, 4, min(scores), max(scores))
scores10 = minmax(np.array(scores), -10, 10, min(scores), max(scores))

plt.plot(scores, torch.sigmoid(torch.tensor(scores1)), marker='x',label="a=1")
plt.plot(scores, torch.sigmoid(torch.tensor(scores5)), marker='x',label="a=5")

plt.plot(scores, torch.sigmoid(torch.tensor(scores10)), marker='x',label="a=10")
plt.plot(scores, torch.sigmoid(torch.tensor(scores2)), marker='x',label="a=2")

plt.plot(scores, torch.sigmoid(torch.tensor(scores3)), marker='x',label="a=3")
plt.plot(scores, torch.sigmoid(torch.tensor(scores4)), marker='x',label="a=4")
plt.legend()
plt.xlabel("Original scores")
plt.ylabel("Platt scaling after minmax (-a,a) transformation")
#plt.plot()
plt.savefig("Comparision.pdf", bbox_inches='tight')

