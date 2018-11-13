#!/usr/bin/env python
import pathlib

import matplotlib.pyplot as plt
import numpy as np

p = pathlib.Path('./')
for comp in list(p.glob('component_*.raw')):
    D = np.fromfile(comp.name, dtype=np.float32)
    D.resize([28, 28])
    plt.title("%s" % comp)
    plt.imshow(D)
    plt.show()
