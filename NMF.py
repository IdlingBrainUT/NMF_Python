# perform NMF (Non-negative Matrix Factorization) and save the result to csv files.
from sklearn.decomposition import NMF
from sklearn.externals.joblib import Parallel, delayed
import numpy as np
import math, time, sys
import pandas as pd
from matplotlib import pyplot as plt

start_time = time.time()

""" parameters ( Start ) """
N_order_start = 1   # -Ns: minimum number of patterns
N_order_end = 800    # -Ne: maximum number of patterns
N_attempt = 10      # -Na: NMF per pattern
N_iter = 1000       # -Ni: maximum iteration in each NMF
N_downsample = 4    # -Nd: The size of bin for down sampling
core = 40
""" parameters ( End ) """

in_file = sys.argv[1] # receive filename from arg. 

data = pd.read_csv(in_file, header=None).dropna(axis=1).values

# R: downsampled data
R = np.array([np.mean(data[i*N_downsample:(i+1)*N_downsample], axis=0) for i in range(math.ceil(data.shape[0]/N_downsample))])

""" verify parameters """
n = R.shape[0] * R.shape[1]
if N_order_start < 1:
    N_order_start = 1
if N_order_end >= (n - 1) / (R.shape[0] + R.shape[1]):
    N_order_end = int((n - 1) / (R.shape[0] + R.shape[1])) - 1
if N_order_start > N_order_end:
    N_order_start = N_order_end

""" scale the data """
sd = np.sqrt(np.sum(R*R)/n)
R /= sd

# perform NMF and returns AICc / 2.
# On about AICc, please refer to https://en.wikipedia.org/wiki/Akaike_information_criterion
def calc(R, N_order, n, k, i):
    model = NMF(n_components=N_order, init="random", random_state=i, max_iter=N_iter, alpha=0.5)
    model.fit(R)

    return n/2 * math.log(model.reconstruction_err_**2 / n) + k + k * (k + 1) / (n - k - 1)

# perform NMF N_attempt time in parallel
# To compare their AICc, we use max value of it
def for_calc(R, N_order, N_attempt):
    k = N_order * (R.shape[0] + R.shape[1])
    temp = Parallel(n_jobs=-1)(delayed(calc)(R, N_order, n, k, i) for i in range(N_attempt))
    return np.max(temp)

""" calc AICc of [N_order_start, N_order_end] and compare them """
batch = (N_order_end - N_order_start + 1)//core
AICs = []
min_AIC = np.inf
flag = False
for i in range(batch):
    AICs += Parallel(n_jobs=-1)(delayed(for_calc)(R, N_order, N_attempt) for N_order in range(N_order_start+core*i, N_order_start+core*(i+1)))
    if min(AICs) == min_AIC:
        flag = True
        N_order_end = N_order_start+core*(i+1)-1
        break # if AICc is increasing, finish the calculation 
    else:
        min_AIC = min(AICs)
# if needed, perform NMF for the rest
if flag == False and N_order_start+core*batch < N_order_end:
    AICs += Parallel(n_jobs=-1)(delayed(for_calc)(R, N_order, N_attempt) for N_order in range(N_order_start+core*batch, N_order_end+1))

""" judge the optimal number of patterns from AICc and perform NMF at that number """
order = AICs.index(min(AICs))
model = NMF(n_components=order+1, init="random", random_state=0)
P = model.fit_transform(R)  # occ
Q = model.components_       # basis

""" Output results """
print("******************************")
print(in_file)
print("order: {}".format(order+1))
print("R-PxQ^T: {}".format(model.reconstruction_err_))
print("time: {}".format(time.time() - start_time))
print("******************************")
print("")
print("order    AIC")
for i, aic in zip(range(N_order_start, N_order_end+1), AICs):
    print("{}:      {}".format(i, aic))

filename = in_file.split(".")[0]

""" plot occ to png """
fig, ax = plt.subplots()
heatmap = ax.pcolor(P.T, cmap=plt.cm.Reds)

ax.set_xticks(np.arange(0, P.shape[0])+0.5, 30)
ax.set_yticks(np.arange(P.shape[1])+0.5, minor=False)

ax.set_yticklabels(range(P.shape[1]), minor=False)

ax.invert_yaxis()
ax.xaxis.tick_top()
plt.xlabel("time")
plt.ylabel("pattern")
plt.savefig(filename + "_occ.png")
plt.clf()

""" plot basis to png """
fig, ax = plt.subplots()
heatmap = ax.pcolor(Q, cmap=plt.cm.Reds)

ax.set_xticks(np.arange(0, Q.shape[1])+0.5, 30)
ax.set_yticks(np.arange(Q.shape[0])+0.5, minor=False)

ax.set_yticklabels(range(Q.shape[0]), minor=False)

ax.invert_yaxis()
ax.xaxis.tick_top()
plt.xlabel("neuron")
plt.ylabel("pattern")
plt.savefig(filename + "_basis.png")
plt.clf()

""" plot AICc (half!) to png """
plt.plot(range(N_order_start, N_order_end+1), AICs)
plt.xlabel("order")
plt.ylabel("AIC")
plt.savefig(filename + "_AIC.png")
plt.clf()

""" save to csv """
dp = pd.DataFrame(P)
dp.to_csv(filename + "_occ.csv", index=False, header=False)
dq = pd.DataFrame(Q)
dq.to_csv(filename +"_basis.csv", index=False, header=False)