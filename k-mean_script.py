import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from pyrsgis import raster
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from scipy.io import loadmat
from sklearn.metrics import classification_report, accuracy_score

import plotly.graph_objects as go
from pyrsgis.convert import changeDimension

# k-mean classification unsupervised
from sklearn.cluster import KMeans
from osgeo import gdal
import numpy as np

#read in data
data = 'bc_test.tif'

#check data
ds, features = raster.read(data, bands='all')
print(features.shape)
#ep.plot_bands(features, cmap="Greys")
#plt.show()
rgb = ep.plot_rgb(features, rgb=(3,2,1), stretch=True, figsize = (8,10),)
plt.show()

#data preprocessing
x = np.moveaxis(features,0,-1)
#data scaling
from sklearn.preprocessing import StandardScaler
x = x.reshape(-1,4)
print(x.shape)
scaler= StandardScaler().fit(x)
x_scaled = scaler.transform(x)

#k-means classification
km = KMeans(n_clusters=5)
km.fit(x_scaled)
km.predict(x_scaled)

out_data = km.labels_.reshape((1546,1652))

ep.plot_bands(out_data, cmap=ListedColormap(['black','red','yellow','blue','green']))
plt.show()



