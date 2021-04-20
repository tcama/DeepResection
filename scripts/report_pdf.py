#%%
import pickle
import pandas as pd
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


with open("../flask/temp_reportdata.pickle", 'rb') as f:
# with open("../flask/temp_reportdata.pickle") as f:
    data = pickle.load(f)
    imgs = data['imgs']
    df = data['df']
# %%

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('report.pdf') as pdf:
    pdf.savefig(imgs)
# %%
for ind, ax in enumerate(imgs.get_axes()):
    fig = plt.figure()
    fig.axes.append(ax)
    plt.show()
    # %%
