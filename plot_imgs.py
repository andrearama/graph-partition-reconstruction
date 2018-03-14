import pickle
import matplotlib.pyplot as plt

with open("results.pickle", "rb") as f:
    output = pickle.load(f)

names = ['random', 'smallworld', 'prefgrowh', 'rome', 'swuspg', 'jazz']
colors = ['red', 'blue', 'green', 'orange', 'brown', 'purple']

shapes = ['x', '*', ''] #One for each community detection algorithm
modes = ['louvain', 'label_prop', 'mapeq']

fig, ax = plt.subplots(2, 2, figsize=(14,10))

#ax[0,0].set_xscale('log')
#ax[0,0].set_yscale('log')

ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')

#ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')

ax[1,1].set_xscale('log')
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
            ax[0,0].plot(o2p[i]/nnd if o2p[i]>0 else 0, p2o[i]/nnd, marker=shape, ms=10, mew=1.5, mec=color, mfc='none') # p2o v o2p
            ax[0,1].plot(ncl[i]/nnd, p2o[i]/nnd, marker=shape, ms=10, mew=1.5, mec=color, mfc='none') # p2o v n_clusters
            ax[1,0].plot(cvg[i], p2o[i]/nnd, marker=shape, ms=10, mew=1.5, mec=color, mfc='none') # p2o v 'coverage'
            ax[1,1].plot(dnc, p2o[i]/nnd, marker=shape, ms=10, mew=1.5, mec=color, mfc='none') # p2o v 'density'


fs = 15
ax[0,0].set_xlabel("K[P0;P*] / N", fontsize=fs)
ax[0,0].set_ylabel("K[P*;P0] / N", fontsize=fs)

ax[0,1].set_xlabel("Num clusters / N", fontsize=fs)
ax[0,1].set_ylabel("K[P*;P0] / N", fontsize=fs)

ax[1,0].set_xlabel("Coverage", fontsize=fs)
ax[1,0].set_ylabel("K[P*;P0] / N", fontsize=fs)

ax[1,1].set_xlabel("Density coeff.", fontsize=fs)
ax[1,1].set_ylabel("K[P*;P0] / N", fontsize=fs)

import matplotlib.lines as mlines

patches = []
labels = []
p = mlines.Line2D([0], [0], marker='x', mec='k', mfc='none', ms=10, ls='', label="Louvain")
l = "Louvain"
labels.append(l)
patches.append(p)
p = mlines.Line2D([0], [0], marker='*', mec='k', mfc='none', ms=10, ls='', label="Label Prop.")
l = "Label Prop."
labels.append(l)
patches.append(p)
for i in range(6):
    p = mlines.Line2D([0], [0], marker=(i+3, 0, 0), mec='k', mfc='none', ms=10, ls='', label="MapEq, Level {:d}".format(i))
    l = "MapEq, Level {:d}".format(i)
    labels.append(l)
    patches.append(p)

#pc = mcollns.PatchCollection(patches)
ax[0,0].legend(handles=patches)

import matplotlib.patches as mpatches

patches = []
for name,color in zip(names,colors):
    p = mpatches.Patch(color=color, label=name)
    patches.append(p)
ax[1,1].legend(handles=patches)

plt.tight_layout()
plt.savefig("results.png")
plt.show()
