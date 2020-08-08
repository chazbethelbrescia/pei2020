# Library of Functions for Preprocessing Data

import glob
import xarray as xr
import pandas as pd
import numpy as np
import xgcm
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from numba import jit
from xhistogram.xarray import histogram
import seaborn as sns
import math



# -------------------------------------------------------------------------
# Take the annual mean of the DataSet
def annual_mean(ds: xr.Dataset, 
                variable: str, year: int) -> xr.DataArray:
    """Takes in a dataset ds, a variable, and year of interest, and returns 
    the annual mean for a given statistic. Use for files in the OCN 
    directory"""
    data = ds[variable].sel(TIME=str(year)).squeeze() 
    mean_da = data.groupby('TIME.year').mean(dim='TIME', skipna=True).squeeze()
    
    return mean_da

    
# -------------------------------------------------------------------------    
# Take the annual minimum of the DataSet
def annual_min(ds: xr.Dataset, 
                variable: str, year: int) -> xr.DataArray:
    """Takes in a dataset ds, a variable, and year of interest, and returns 
    the annual minimum for a given statistic. Use for files in the OCN 
    directory"""
    data = ds[variable].sel(TIME=str(year)).squeeze()
    min_da = data.groupby('TIME.year').min(dim='TIME', skipna=True).squeeze()
    
    return min_da

# -------------------------------------------------------------------------    
# Take the decadal mean of the DataArray
def decadal_mean(da_annual: xr.DataArray, startyear: int) -> xr.DataArray:
    """Takes in a dataarray da, and the first year of the decade of interest, 
    and returns the decadal mean for a given statistic. For incomplete
    decades, takes the mean of years through the end of the decade."""
    da_mean = da_annual.sel(year=startyear).copy()
    endyear = startyear+(10-(startyear%10))
    for yr in range(startyear+1,endyear):
        da_mean += da_annual.sel(year=yr)
    
    return da_mean/10


# -------------------------------------------------------------------------    
# Take a moving average of an annual DataArray
def moving_avg(da_annual: xr.DataArray, 
               startyear: int, endyear: int,
              interval: int, yearstep: int=1) -> list:
    """Takes in a dataarray da, the starting year, the interval 
    over which to take a moving average, and number of years to skip 
    for each calculation. Returns a list of DataArrays for the moving 
    average at each measured year."""
    list_avgs = []
    span = interval//2
    curr = startyear+span
    # years with moving averages spanning the entire interval
    while (curr <= endyear-span):
        # Take avg over interval
        da_mean = da_annual.sel(year=curr).copy()
        for yr in range(curr-span, curr):
            da_mean += da_annual.sel(year=yr)
        if interval%2==1:
            span+=1
        for yr in range(curr+1, curr+span):
            da_mean += da_annual.sel(year=yr)
        da_mean = da_mean/interval
        list_avgs.append(da_mean.squeeze())
        curr += yearstep
    
    return list_avgs

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Make contour plot of the Dataset in its Axes for annual Omega Aragonite
def contour_oa(ds: xr.core.dataset.Dataset, 
                       in_year: int, 
                       ax: mpl.axes.Axes,
                       title: str,
                   central_longitude: int=0) -> mpl.axes.Axes:
    """Takes in a Dataset for Omega Aragonite, and creates a contour plot 
    for the data for the given year. Adds a darkgrey landmask, grindlines, 
    coastlines, a title, and a colorbar to the plot."""
    
    clevs = np.array([0,1,2,3,4])
    # Can also choose to define colormap
    # colors = ['red', 'orange', 'yellow','green','blue']
    crs = ccrs.PlateCarree(central_longitude=central_longitude)
    
    # Specify variables
    X = ds['XT_OCEAN']
    Y = ds['YT_OCEAN']
    # Already grouped by year
    Z = ds['OMEGA_ARAG'].sel(year=in_year).squeeze() 
    Z, X = add_cyclic_point(Z, coord=X)
    
    im = ax.contourf(X,Y,Z, clevs, transform=crs)
    if ax == plt.gcf().get_axes()[0]:
        cbar = plt.colorbar(im,ax=plt.gcf().get_axes(),orientation='horizontal',fraction=0.05,pad=0.05)
        cbar.set_label('$\Omega$ Aragonite', fontsize=18)
    
    # Zoom in on a region
    # ax.set_extent([x0,x1,y0,y1])

    # Add land mask, gridlines, coastlines, and title
    ax.add_feature(cfeature.LAND,zorder=10,facecolor='darkgray')
    ax.gridlines()
    ax.coastlines()
    ax.set_title(title+" "+str(in_year), fontsize=18, loc='center')
    
    return ax

    

# -------------------------------------------------------------------------    
# Make contour plot of the Dataset in its Axes for annual Temperature
def contour_temp(ds: xr.core.dataset.Dataset,
                       in_year: int, 
                       ax: mpl.axes.Axes,
                       title: str,
                   central_longitude: int=0) -> mpl.axes.Axes:
    """Takes in a Dataset for Temperature, and creates a contour plot for
    the data for the given year. Adds a darkgrey landmask, grindlines, 
    coastlines, a title, and a colorbar to the plot."""
    
    clevs = np.arange(260, 320, 10)
    # Can also choose to define colormap
    colors = ['blue','green','yellow','orange','red']
    crs = ccrs.PlateCarree(central_longitude=central_longitude)
    
    # Specify variables
    X = ds['xt_ocean']
    Y = ds['yt_ocean']
    # Already grouped by year
    Z = ds['temp'].sel(year=in_year).squeeze() 
    Z, X = add_cyclic_point(Z,coord=X)
    
    # can also use cmap='plasma','inferno',etc...
    im = ax.contourf(X,Y,Z, levels=clevs, transform=crs)
    if ax == plt.gcf().get_axes()[0]:
        cbar = plt.colorbar(im,ax=plt.gcf().get_axes(),
                            orientation='horizontal',fraction=0.05,pad=0.05)
        cbar.set_label('Temeprature ($^\circ\,K$)', fontsize=18)
    
    # Zoom in on a region
    # ax.set_extent([x0,x1,y0,y1])

    # Add land mask, gridlines, coastlines, and title
    ax.add_feature(cfeature.LAND,zorder=10,facecolor='darkgray')
    ax.gridlines()
    ax.coastlines()
    ax.set_title(title+" "+str(in_year), fontsize=18, loc='center')
    
    return ax
 

# -------------------------------------------------------------------------
# Make contour plot of the Dataset in its Axes for Annual Depth
def contour_z(ds: xr.core.dataset.Dataset, 
                       in_year: int, 
                       ax: mpl.axes.Axes,
                       title: str,
                   central_longitude: int=0) -> mpl.axes.Axes:
    """Takes in a Dataset for the depth at 0.1 W/m^2 irradiance or Omeaga 
    Aragonite undersaturation threshold, and creates a contour plot 
    for the data for the given year. Adds a darkgrey landmask, grindlines, 
    coastlines, a title, and a colorbar to the plot."""
    
    
    # Can also choose to define colormap
    # colors = ['red', 'orange', 'yellow','green','blue']
    crs = ccrs.PlateCarree(central_longitude=central_longitude)
    
    # Specify variables
    X = ds['XT_OCEAN']
    Y = ds['YT_OCEAN']
    # Already grouped by year
    Z = ds['Z'].sel(year=in_year).squeeze() 
    Z, X = add_cyclic_point(Z, coord=X)
    
    im = ax.contourf(X,Y,Z, transform=crs)
    if ax == plt.gcf().get_axes()[0]:
        cbar = plt.colorbar(im,ax=plt.gcf().get_axes(),orientation='horizontal',fraction=0.05,pad=0.05)
        cbar.set_label('Depth (m)', fontsize=16)
    
    # Zoom in on a region
    # ax.set_extent([x0,x1,y0,y1])

    # Add land mask, gridlines, coastlines, and title
    ax.add_feature(cfeature.LAND,zorder=10,facecolor='darkgray')
    ax.gridlines()
    ax.coastlines()
    ax.set_title(title+" "+str(in_year), fontsize=16, loc='center')
    
    return ax

# ------------------------------------------------------------------------- 
# ------------------------------------------------------------------------- 
# -------------------------------------------------------------------------   

def calc_undersat(ds_oa: xr.Dataset) -> xr.DataArray:
    """Takes in a Dataset for Omega Aragonite, and returns a DataArray 
    with tallied undersaturated months by year"""
    
    omega_arag = ds_oa.omega_arag.squeeze()
    # Filter out nan points (above threshold) and replace with 0
    da_oa_under = omega_arag.where(omega_arag <= 1.0).fillna(0)
    # Replace Omega Arag points below threshold with value 1 
    # (tally undersaturated months)
    da_oa_under = da_oa_under.where(da_oa_under == 0).fillna(1)
    # Group by year and sum over each year to obtain undersaturated months per year
    da_undersat = da_oa_under.groupby('time.year').sum(dim='time')
    
    return da_undersat


# -------------------------------------------------------------------------   

def contour_emergence(da_emergence: xr.DataArray, m: int, 
                      ax: mpl.axes.Axes, rcp: str=None):
    """Takes in DataArray for time of emergence, axes for the plot, number of
    undersaturation months for recognized signal, and simulation name, and
    makes a contour plot."""
    crs = ccrs.Robinson(central_longitude=180)
    src = ccrs.PlateCarree()
    clevs = np.arange(1950, 2101, 10)
    lon = da_emergence.xt_ocean.data
    lat = da_emergence.yt_ocean.data
    
    im = ax.contourf(lon, lat, da_emergence, cmap='plasma',
                     levels=clevs, transform=src, robust=True)
    if ax == plt.gcf().get_axes()[0]:
        cbar = plt.colorbar(im,ax=plt.gcf().get_axes(),orientation='horizontal',
                            fraction=0.05,pad=0.05)
        cbar.set_label('Year of Emergence', fontsize=16)

    ax.add_feature(cfeature.LAND,zorder=10,facecolor='darkgray')
    ax.set_global()
    if m == 1: months = 'month'
    else: months = 'months'
    if rcp==None:
        ax.set_title(str(m)+' '+months+'/year', fontsize=16)
    else:
        ax.set_title(rcp+' - '+str(m)+' '+months+'/year', fontsize=16)
        
        
# -------------------------------------------------------------------------   
def contour_transition(da_transition: xr.DataArray, 
                      ax: mpl.axes.Axes, rcp: str):
    """Takes in DataArray for transition time from x to y months Omega Aragonite
    undersaturation, axes for the plot, and the simulation name, and makes a 
    contour plot."""
    crs = ccrs.Robinson(central_longitude=180)
    src = ccrs.PlateCarree()
    clevs = np.arange(0, 31, 5)
    lon = da_transition.xt_ocean.data
    lat = da_transition.yt_ocean.data
    
    im = ax.contourf(lon, lat, da_transition, cmap='plasma',
                     levels=clevs, transform=src, robust=True)
    if ax == plt.gcf().get_axes()[0]:
        cbar = plt.colorbar(im,ax=plt.gcf().get_axes(),orientation='horizontal',
                            fraction=0.05,pad=0.05)
        cbar.set_label('Transition Time from 1-->6 Months Undersaturated (years)', fontsize=16)

    ax.add_feature(cfeature.LAND,zorder=10,facecolor='darkgray')
    ax.set_global()
    ax.set_title(rcp)


# ------------------------------------------------------------------------- 
def round_half(x):
    """Rounds a floating point number to exclusively the nearest 0.5 mark, 
    rather than the nearest whole integer."""
    x += 0.5
    x = round(x)
    return x-0.5


# ------------------------------------------------------------------------- 
@jit(nopython=True)
def distance(x0, y0, x1, y1):
    """Return the coordinate distance between 2 points"""
    dist = math.hypot((x1-x0),(y1-y0))
    return dist

# ------------------------------------------------------------------------- 
@jit(nopython=True)
def haversine_dist(lat1,lat2,lon1,lon2,degrees=True):
    if degrees:
        fac = np.pi/180
        lat1 = lat1*fac
        lat2 = lat2*fac
        lon1 = lon1*fac
        lon2 = lon2*fac
    R = 6371E3
    return 2*R*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))


# ------------------------------------------------------------------------- 
@jit(forceobj=True)
def valid_gradient(x0, y0, x, y, value0, da_mean: np.ndarray) -> bool:
    """Takes in a start and endpoint, the value of the climate stressor 
    at the starting point, and a DataArray for decadal mean of a climate
    stressor. Returns True if the spatial gradient is positive and False
    otherwise."""
    # Define search radius and angle along path
    radius = 10
    theta = math.atan2((y-y0),(x-x0))
    # Interpolate gradient within search radius
    for r in range(1,radius+1):
        xstep = r * math.cos(theta)
        ystep = r * math.sin(theta)
        x = (x0+xstep-0.5)%360
        y = (y0+ystep+89.5)%180
        if x >= 360:
            x = 0
        value1 = da_mean[int(y),int(x)]
        # Nearest point is valid if change in climate stressor is negative
        if (value1-value0 <= 0) or (abs(y-y0)>50) or (not np.isfinite(value1)):
            return False
    
    return True

# ------------------------------------------------------------------------- 
@jit(forceobj=True)
def min_dist(x0, y0, xlist, ylist, da_mean) -> tuple:
    """From a point on a contour line, find the closest point on the
    adjacent contour line, and the distance between the points (x, y 
    components and magnitude)
    
    PARAMETERS
    ----------
    x0 : float, x coord of interest
    y0 : float, y coord of interest
    xlist : list of adjacent contour's x coords
    ylist : list of adjacent contour's y coords
    da_mean : DataArray of climate stressor

    RETURNS
    -------
    x_nearest, y_nearest, dx, dy, min_dist : tuple, closest point from list 
    of adjacent contour coordinates with a positive gradient, (if all paths 
    have a negative gradient, return closest point), and the distance to 
    that point."""
    x0 = round(x0,1)
    y0 = round(y0,1)
    # Get sorted list of nearest points (distance and x-y coordinates)
    nearest_points = []
    num_points = len(xlist)
    xlist = np.around(np.array(xlist),1)
    ylist = np.around(np.array(ylist),1)
    for i in range(num_points):
        point = []
        x = xlist[i]
        y = ylist[i]
        dist = haversine_dist(y0, y, x0, x)  # calc distance
        point.extend([dist,x,y])  # make point list
        nearest_points.append(point)  # add point to points list
        
    nearest_points.sort()  # Sort points from near to far
    
    val = 1  # value at starting point (1 on contour)
    x_nearest = np.nan
    y_nearest = np.nan
    min_dist = nearest_points[0][0]
    # Starting at nearest point, check if gradient is positive
    # and check distances as necessary
    num_points = len(nearest_points)
    for i in range(num_points):
        point = nearest_points[i]
        if valid_gradient(x0, y0, point[1], point[2], val, da_mean):
            min_dist = point[0]
            x_nearest = point[1]
            y_nearest = point[2]
            
    # Return closest points, components of distance and distance magnitude
    dx = np.nan
    dy = np.nan
    if y_nearest != np.nan:
#         y_avg = (y0+y_nearest)/2
#         dx = haversine_dist(y_avg, y_avg, x0, x_nearest)
#         dy = haversine_dist(y0, y_nearest, 0, 0)
        
        dist_cap=1500000 # in meters
        # flip directions accordingly
#         if x_nearest < x0:
#             dx = -dx
#         if y_nearest < y0:
#             dy = -dy
        if min_dist > dist_cap:
            min_dist = np.nan
#             N = np.sqrt(dx**2 + dy**2)
#             dx = dist_cap*dx/N
#             dy = dist_cap*dy/N
    return x_nearest, y_nearest, dx, dy, min_dist

# ------------------------------------------------------------------------- 
@jit(forceobj=True)
def min_dist_approx(x0, y0, xlist, ylist, da_mean) -> tuple:
    """From a point on a contour line, find the closest point on the
    adjacent contour line, and the distance between the points (x, y 
    components and magnitude)
    
    PARAMETERS
    ----------
    x0 : float, x coord of interest
    y0 : float, y coord of interest
    xlist : list of adjacent contour's x coords
    ylist : list of adjacent contour's y coords
    da_mean : DataArray of climate stressor

    RETURNS
    -------
    x_nearest, y_nearest, dx, dy, min_dist : tuple, closest point from list 
    of adjacent contour coordinates with a positive gradient, (if all paths 
    have a negative gradient, return closest point), and the distance to 
    that point."""
    
    # Get sorted list of nearest points (distance and x-y coordinates)
    nearest_points = []
    num_points = len(xlist)
    for i in range(num_points):
        point = []
        x = xlist[i]
        y = ylist[i]
        dist = haversine_dist(y0, y, x0, x)  # calc distance
        point.extend([dist,x,y])  # make point list
        nearest_points.append(point)  # add point to points list
        
    nearest_points.sort()  # Sort points from near to far
    
    val = da_mean[int(y0+89.5), int(x0-0.5)]  # value at starting point
    x_nearest = np.nan
    y_nearest = np.nan
    min_dist = nearest_points[0][0]
    # Starting at nearest point, check if gradient is positive
    # and check distances as necessary
    num_points = len(nearest_points)
    for i in range(num_points):
        point = nearest_points[i]
        if valid_gradient(x0, y0, point[1], point[2], val, da_mean):
            min_dist = point[0]
            x_nearest = point[1]
            y_nearest = point[2]
            break
            
    # Return closest points, components of distance and distance magnitude
    dx = np.nan
    dy = np.nan
    if y_nearest != np.nan:
        y_avg = (y0+y_nearest)/2
        dx = haversine_dist(y_avg, y_avg, x0, x_nearest)  # dx approximation
        dy = haversine_dist(y0, y_nearest, 0, 0)  # dy approximation
        
        dist_cap=1500000 # in meters
        # flip directions accordingly
        if x_nearest < x0:
            dx = -dx
        if y_nearest < y0:
            dy = -dy
        if min_dist > dist_cap:
            min_dist = np.nan
            N = np.sqrt(dx**2 + dy**2)
            dx = dist_cap*dx/N
            dy = dist_cap*dy/N
    return x_nearest, y_nearest, dx, dy, min_dist

# ------------------------------------------------------------------------- 
def hist_sum(da_escvel, levels, start, end) -> np.ndarray:
    """From a DataArray of escape velocities and levels array, create an 
    array for the sum of histogram bins over a series of decades."""
    bins = np.array(levels)
    hist_sum = histogram(da_escvel[start], bins=[bins]).data
    for i in range(start+1,end):
        h = histogram(da_escvel[i], bins=[bins]).data
        hist_sum += h
    return hist_sum


# ------------------------------------------------------------------------- 
def hist_mean(da_escvel: xr.DataArray, levels: np.ndarray, 
              start: int,  end: int, 
              name: str='Esc Vel Distribution') -> np.ndarray:
    """From a DataArray of escape velocities, levels array, start and
    end index, and name, create a DataArray for the mean of histogram 
    bins over a series of decades."""
    bins = np.array(levels)
    hist_sum = histogram(da_escvel[start], bins=[bins]).data
    for i in range(start+1,end):
        h = histogram(da_escvel[i], bins=[bins]).data
        hist_sum += h
    hist_mean = hist_sum / hist_sum.sum()
    hist_mean = xr.DataArray(hist_mean, dims=['bin_edges'],coords=[np.delete(bins,len(bins)-1)]).rename(name)
    
    return hist_mean



# ------------------------------------------------------------------------- 
# ------------------------------------------------------------------------- 
# -------------------------------------------------------------------------    
# Generate a Grid for a given Dataset with X, Y, and T axes
def get_grid(ds: xr.core.dataset.Dataset) -> xgcm.Grid:
    grid = xgcm.Grid(ds, periodic=['X'], 
                 coords={'X': {'center': 'xt_ocean', 'left': 'xt_ocean_left'},
                         'Y': {'center': 'yt_ocean', 'left': 'yt_ocean_left'},
                         'T': {'center': 'time'}})
    return grid

# -------------------------------------------------------------------------    
# Convert degrees to Cartesian distances on the globe
def dll_dist(dlon, dlat, lon, lat):
    """Converts lat/lon differentials into distances in meters
        
        PARAMETERS
        ----------
        dlon : xarray.DataArray longitude differentials
        dlat : xarray.DataArray latitude differentials
        lon  : xarray.DataArray longitude values
        lat  : xarray.DataArray latitude values

        RETURNS
        -------
        dx  : xarray.DataArray distance inferred from dlon
        dy  : xarray.DataArray distance inferred from dlat
        
    from: 
    https://xgcm.readthedocs.io/en/latest/autogenerate_examples.html
    """
    distance_1deg_equator = 111000.0
    dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(lat)) * distance_1deg_equator
    dy = dlat * distance_1deg_equator
    return dx, dy


# -------------------------------------------------------------------------       
# Calculate the spatial gradient of a Grid object
def grid_calculations(grid: xgcm.Grid, ds_full: xr.core.dataset.Dataset):
    
    # Compute difference (in degrees) along longitude and latitude for
    # both cell center and left
    dlong = grid.diff(ds_full.xt_ocean, 'X', boundary_discontinuity=360)
    dlonc = grid.diff(ds_full.xt_ocean_left, 'X', boundary_discontinuity=360)

    dlatg = grid.diff(ds_full.yt_ocean, 'Y', boundary='fill', fill_value=np.nan)
    dlatc = grid.diff(ds_full.yt_ocean_left, 'Y', boundary='fill', fill_value=np.nan)
    
    # Convert degrees to actual Cartesian distances on the Earth
    # add distances to coordinates in data
    ds_full.coords['dxg'], ds_full.coords['dyg'] = dll_dist(dlong, dlatg,
                                     ds_full.xt_ocean, ds_full.yt_ocean)
    ds_full.coords['dxc'], ds_full.coords['dyc'] = dll_dist(dlonc, dlatc, 
                                     ds_full.xt_ocean, ds_full.yt_ocean)
    
    # Calculate area of each gridcell
    ds_full.coords['area_c'] = ds_full.dxc * ds_full.dyc
    
    # Fill nan values
    dyg = ds_full.dyg.fillna(111000)
    dyc = ds_full.dyc.fillna(111000)
    ds_full.coords['dyg'] = dyg
    ds_full.coords['dyc'] = dyc

# ------------------------------------------------------------------------- 

@jit(forceobj=True)
def numba_leastsqr(x, y):
    """ Computes the least-squares solution to a linear matrix equation.
    
    from https://stackoverflow.com/questions/23550483/numba-and-cython-arent-improving-the-performance-compared-to-cpython-significan"""
    len_x = len(x)
    x_avg = sum(x)/len_x
    y_avg = sum(y)/len(y)
    var_x = 0
    cov_xy = 0
    for i in range(len_x):
        temp = (x[i] - x_avg)
        var_x += temp**2
        cov_xy += temp*(y[i] - y_avg)
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)


# ------------------------------------------------------------------------- 
# ------------------------------------------------------------------------- 
# ------------------------------------------------------------------------- 
# -------------------------------------------------------------------------
# Make contour plot of the data mpl.axes.Axes, 
def contour_default(X: xr.core.dataarray.DataArray,
                 Y: xr.core.dataarray.DataArray,
                 Z: xr.core.dataarray.DataArray,
                    plotwidth: int=10, plotheight: int=10,
                   central_longitude: int=0) -> mpl.figure.Figure:
    """Takes in three dataarray variables, and creates a standard contour plot 
    for the data with the central longitude at 180 degrees. Adds a darkgrey 
    landmask, grindlines and coastlines to the plot. Contour levels and plot 
    size can be adjusted. Returns the figure itself and its image."""
    
    # Define contour levels for Omega Aragonite
    clevs=[0,1,2,3,4]
    
    # Specify the map projection
    crs = ccrs.PlateCarree(central_longitude=central_longitude)

    # Create a figure and axes using matplotlib
    fig, ax = plt.subplots(figsize=(plotwidth,plotheight), subplot_kw={'projection':crs})

    Z,X = add_cyclic_point(Z, coord=X)
    
    # Create contour plot of SST
    ax.contourf(X,Y,Z, transform=crs)
    
    # Add features to the plot
    ax.add_feature(cfeature.LAND, zorder=10, facecolor='darkgrey')
    ax.gridlines(crs=crs)
    ax.coastlines()
    
    return fig
    

# -------------------------------------------------------------------------
# Ideas for Possible Functions
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------