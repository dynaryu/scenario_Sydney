# a scrip to make contour map and KML for google earth

import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.colors as colors 
import numpy as np
import simplekml
import time
import sys
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.mlab import griddata
from shapely.geometry import Polygon, Point, MultiPolygon
from eqrm_code.RSA2MMI import rsa2mmi_array

def read_lat_lon(eqrm_output_path):
    ''' read lat lon file '''

    site_tag = [x for x in os.listdir(eqrm_output_path) \
        if 'sites' in x][0].replace('_sites', '')

    lat = np.load(os.path.join(
        eqrm_output_path, site_tag + '_sites', 'sites', 'latitude.npy'))

    lon = np.load(os.path.join(
        eqrm_output_path, site_tag + '_sites', 'sites', 'longitude.npy'))

    return(lat, lon, site_tag)

def read_gm(eqrm_output_path, site_tag):
    ''' read ground motion, an output of EQRM run'''

    atten_periods = np.load(os.path.join(eqrm_output_path, site_tag + \
        '_motion/atten_periods.npy'))

    selected_periods = [0.0, 0.3, 1.0]

    idx_period = [(np.abs(atten_periods - period)).argmin() \
        for period in selected_periods]

    try:
        gmotion = np.load(os.path.join(eqrm_output_path, site_tag +\
            '_motion/soil_SA.npy')) # (1,1,1,sites,nsims,nperiods)
    except IOError:
        gmotion = np.load(os.path.join(eqrm_output_path, site_tag +\
            '_motion/bedrock_SA.npy'))

    pga = gmotion[0, 0, 0, :, :, idx_period[0]]
    sa03 = gmotion[0, 0, 0, :, :, idx_period[1]]
    sa10 = gmotion[0, 0, 0, :, :, idx_period[2]]
    mmi = rsa2mmi_array(sa10)

    return(pga, sa03, sa10, mmi)

def plot_gmfield(lat, lon, value, levs, output_file):

    #lat = xyz[:,0]
    #lon = xyz[:,1]
    #value = xyz[:,2]

    # Basemap
    llcrnlon = lon.min()
    llcrnlat = lat.min()
    urcrnlon = lon.max()
    urcrnlat = lat.max()

    #print '%s, %s, %s, %s' %(llcrnlon, urcrnlat, urcrnlon, llcrnlat)
    #print '%s, %s, %s, %s' %(lon.min(), lon.max(), lat.min(), lat.max())

    m = Basemap(llcrnrlon=llcrnlon,llcrnrlat=llcrnlat,urcrnrlon=urcrnlon,urcrnrlat=urcrnlat,resolution='f',projection='merc',lon_0=(urcrnlon+llcrnlon)/2,lat_0=(urcrnlat+llcrnlat)/2)

    # plot contour map
    fig = plt.figure()
    ax = plt.gca()

    m.drawcoastlines()
    #m.drawmapboundary()
    m.drawstates(linewidth=3)
    #m.drawlsmask()
    #m.fillcontinents(color='lightgrey',lake_color='white')
    m.drawcountries(linewidth=3)
    m.drawparallels(np.arange(-34,-33.2,0.2),labels=[1,0,0,0])
    m.drawmeridians(np.arange(150.0,151.4,0.2),labels=[0,0,0,1])

    #xi = np.linspace(llcrnlon,urcrnlon,1000)
    #yi = np.linspace(llcrnlat,urcrnlat,1000)
    #zi = griddata(lon,lat,value,xi,yi, interp='linear')

    #xi_, yi_ = m(151.153, -33.914)
    #m.scatter(xi, yi, marker="*", facecolor='black', s=30)

    #(xi,yi) = np.meshgrid(xi,yi)
    xi, yi = m(lon, lat)

    #levs = np.arange(3, 7.0, 0.5)
    cmap = plt.get_cmap('jet', len(levs)-1)
    #cmap = plt.cm.jet(len(levs)-1)

    m.scatter(xi, yi, c=value, vmin = levs[0], vmax= levs[-1], cmap=cmap, lw=0)


    #colorscale = plt.cm.ScalarMappable()
    #colorscale.set_array(value)
    #colorscale.set_cmap(cmap)

    #colors = colorscale.to_rgba(value)
    #m.scatter(lon,lat,c=colors,zorder=1000,cmap=cmap,s=10)
    #m.colorbar(colorscale, shrink=0.50, ax=m,extend='both')

    cbar = m.colorbar(location='bottom', pad="10%", ticks=levs)

    plt.savefig(output_file, dpi=300)

###############################################################################
# main program
eqrm_output_path = sys.argv[1]
output_file = sys.argv[2]

# read ground motion grid data 
(lat, lon, site_tag) = read_lat_lon(eqrm_output_path)
(pga, sa03, sa10, mmi) = read_gm(eqrm_output_path, site_tag)

plot_gmfield(lat, lon, mmi[:, 0], levs=np.arange(3, 7.0, 0.5), './mmi.png')
plot_gmfield(lat, lon, sa03[:, 0], levs=np.arange(0.0, 0.55, 0.5), './sa03.png')
plot_gmfield(lat, lon, sa10[:, 0], levs=np.arange(3, 7.0, 0.5), './sa10.png')
plot_gmfield(lat, lon, pga[:, 0], levs=np.arange(3, 7.0, 0.5), './pga.png')

   
