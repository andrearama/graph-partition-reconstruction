import pickle
import matplotlib.pyplot as plt

with open("results.pickle", "rb") as f:
    output = pickle.load(f)

names = ['random', 'smallworld', 'prefgrowh', 'rome', 'swuspg', 'jazz']
colors = ['red', 'blue', 'green', 'orange', 'brown', 'purple']

shapes = ['x', '', '*'] #One for each community detection algorithm
modes = ['louvain', 'mapeq', 'label_prop']

fig, ax = plt.subplots(2, 2)

#ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')

ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')

#ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')

#ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')

for name,color in zip(names,colors):

    d1 = output[name]

    for mode,shape in zip(modes,shapes):

        clc = d1["clc"]
        dnc = d1["dnc"]
        nnd = d1["nnd"]

        d2 = d1[mode]
        o2p = d2["o2p"]
        p2o = d2["p2o"]
        cvg = d2["cvg"]
        ncl = d2["ncl"]

        for i in range(len(o2p)):

            if mode=='mapeq':
                shape = (i+3, 0, 0) # for enumerating the different partitions
            ax[0,0].plot(o2p[i]/nnd, p2o[i]/nnd, marker=shape, mec=color, mfc='none') # p2o v o2p
            ax[0,1].plot(ncl[i]/nnd, p2o[i]/nnd, marker=shape, mec=color, mfc='none') # p2o v n_clusters
            ax[1,0].plot(cvg[i], p2o[i]/nnd, marker=shape, mec=color, mfc='none') # p2o v 'coverage'
            ax[1,1].plot(dnc, p2o[i]/nnd, marker=shape, mec=color, mfc='none') # p2o v 'density'

plt.savefig("results.png")
plt.show()
