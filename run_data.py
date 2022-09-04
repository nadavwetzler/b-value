import numpy as np 
import pandas as pd
import matplotlib.pyplot as plb
from load_cat import *
from calc_gr_ks import *

# 1) Load catalog
cat = read_SCEDC('SCEDC.csv', lats=[32.0, 37.0], lons=[-120, -114])

# 2) Run the K-S test
b_data = calc_b_val(cat.M, 0.1, 2.0, 4.0)

# 3) Plot results
fig1 = plb.figure(1)
ax1 = fig1.add_subplot(1, 2, 1)
ax2 = fig1.add_subplot(1, 2, 2)
print_b_val(b_data, min(cat.M), max(cat.M), ax1, ax2, 'S. California', 'gray', 'k')

plb.show()