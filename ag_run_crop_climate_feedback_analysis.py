import rasterio as rio
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
import numpy as np
import numpy.matlib
from scipy import interpolate
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import scipy
import os, sys, pickle, gzip
import datetime
import geopy.distance
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry
import shapely.ops
import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import itertools
import random
import metpy
from metpy.plots import USCOUNTIES
from pyts.decomposition import SingularSpectrumAnalysis

import warnings
warnings.filterwarnings('ignore')

dataDirDiscovery = '/dartfs-hpc/rc/lab/C/CMIG/ecoffel/data/projects/ag-land-climate'

# run('../util/setupConsole')

# low and high temps for gdd/kdd calcs, taken from Butler, et al, 2015, ERL
t_low = 9
t_high = 29

crop = 'Maize'
wxData = 'era5'

useTrendMethod = True

uncertaintyProp = True

useDeepak = False

includeKdd = False

n_samples = 100
n_bootstraps = 100

yearRange = [1981, 2019]

minCropYears = 30

if os.path.isfile('%s/us-county-yield-gdd-kdd-%s-%s'%(dataDirDiscovery, crop, wxData)):
    usCounties = pd.read_pickle('%s/us-county-yield-gdd-kdd-%s-%s'%(dataDirDiscovery, crop, wxData))

usCounties = usCounties.drop(columns=['CWA', 'TIME_ZONE', 'FE_AREA'])

# drop all counties with any nans
yieldNans = np.array(list(map(np.isnan, usCounties['maizeYield'])))
yieldTrendNans = np.array(list(map(np.isnan, usCounties['maizeYieldTrend'])))
gddNans = np.array(list(map(np.isnan, usCounties['gdd'])))
kddNans = np.array(list(map(np.isnan, usCounties['kdd'])))
# inds1 = np.where( (np.array([len(np.where((yieldNans[i]==False) & ((gddNans[i]==True) | (kddNans[i]==True)))[0]) for i in range(len(yieldNans))]) > 0))[0]
inds = np.where((yieldTrendNans == True))[0]
# inds = np.union1d(inds1, inds2)

usCounties = usCounties.drop(index=inds)

countyMaizeHaAc = np.array([a for a in np.array(list(usCounties['maizeHarvestedArea']))])
countyMaizeHaMeanAc = np.array([np.nanmean(a) for a in np.array(list(usCounties['maizeHarvestedArea']))])
countySoybeanHaAc = np.array([a for a in np.array(list(usCounties['soybeanHarvestedArea']))])
countySoybeanHaMeanAc = np.array([np.nanmean(a) for a in np.array(list(usCounties['soybeanHarvestedArea']))])

countyMaizeYieldDetrend = np.array(list(usCounties['maizeYieldDetrendPlusMean']))
countyMaizeYield = np.array(list(usCounties['maizeYield']))
countyMaizeYieldDetrendAnom = np.array(list(usCounties['maizeYieldDetrend']))

countySoybeanYieldDetrend = np.array(list(usCounties['soybeanYieldDetrendPlusMean']))
countySoybeanYield = np.array(list(usCounties['soybeanYield']))
countySoybeanYieldDetrendAnom = np.array(list(usCounties['soybeanYieldDetrend']))

countyTotalProd = np.array([y*a for y,a in zip(countyMaizeYield, countyMaizeHaAc)] + [y*a for y,a in zip(countySoybeanYield, countySoybeanHaAc)])

adjustForHarvestedArea = False

areaLimit = [30]
irrLimit = 10

useSsa = False

ssa = SingularSpectrumAnalysis(window_size=.5, groups=None)

def ssaDetrend(x):
    ssafit = ssa.fit_transform(x.reshape(1,-1))
    return x-ssafit[0,:]

def normWithNan(x):
    x1d = np.reshape(x, [x.size])
    nn = np.where(~np.isnan(x1d))[0]
    x1d = x1d[nn]
    x_norm = x.copy()/np.linalg.norm(x1d)
    return x_norm


haMeanAggAll = np.nanmean(np.array([a for a in usCounties['maizeHarvestedAreaFraction']]), axis=1)*100
irAggAll = np.array([a for a in usCounties['maizeCountyIrrigationFraction']])

countyAcAll = np.array([np.nanmean(a) for a in usCounties['maizeCountyArea']])
haMaizeAllAc = np.array([a for a in usCounties['maizeHarvestedArea']])
haSoybeanAllAc = np.array([a for a in usCounties['soybeanHarvestedArea']])
haMaizeMeanAllAc = np.array([np.nanmean(a) for a in usCounties['maizeHarvestedArea']])
haSoybeanMeanAllAc = np.array([np.nanmean(a) for a in usCounties['soybeanHarvestedArea']])
haTotalAllAc = haMaizeMeanAllAc + haSoybeanMeanAllAc
haTotalFracAll = haTotalAllAc/countyAcAll

countySeasonLenSec = np.array(list(usCounties['seasonalSeconds']))

if useDeepak:
    countyYieldDetrend = np.array(list(usCounties['maizeYieldDetrendPlusMeanDeepak'])) 
    countyYield = np.array(list(usCounties['maizeYieldDeepak']))
    countyYieldDetrendAnom = np.array(list(usCounties['maizeYieldDetrendDeepak']))
else:
    countyMaizeHaAc = np.array([a for a in np.array(list(usCounties['maizeHarvestedArea']))])
    countyMaizeHaMeanAc = np.array([np.nanmean(a) for a in np.array(list(usCounties['maizeHarvestedArea']))])
    countyMaizeHaFrac = np.array([a for a in np.array(list(usCounties['maizeHarvestedAreaFraction']))])*100
    countySoybeanHaAc = np.array([a for a in np.array(list(usCounties['soybeanHarvestedArea']))])
    countySoybeanHaMeanAc = np.array([np.nanmean(a) for a in np.array(list(usCounties['soybeanHarvestedArea']))])
    countySoybeanHaFrac = np.array([a for a in np.array(list(usCounties['soybeanHarvestedAreaFraction']))])*100

    countyMaizeYield = np.array(list(usCounties['maizeYield']))
    countyMaizeYieldDetrendAnom = np.array(list(usCounties['maizeYieldDetrend']))

    countySoybeanYield = np.array(list(usCounties['soybeanYield']))
    countySoybeanYieldDetrendAnom = np.array(list(usCounties['soybeanYieldDetrend']))

    countyTotalProd = np.full(countySoybeanYield.shape, np.nan)
    for c in range(countyTotalProd.shape[0]):
        p_maize = np.array([x[0]*x[1] for x in zip(countyMaizeYield[c,:], countyMaizeHaAc[c,:])])
        p_soybean = np.array([x[0]*x[1] for x in zip(countySoybeanYield[c,:], countySoybeanHaAc[c,:])])
        countyTotalProd[c,:] = p_maize + p_soybean


countyIr = np.array(list(usCounties['maizeCountyIrrigationFraction']))
countyState = np.array(list(usCounties['STATE']))
countyFips = np.array(list(usCounties['FIPS']))

countyPr = np.array(list(usCounties['seasonalPrecip']))  # mm
if wxData == 'era5':
    countyT = np.array(list(usCounties['seasonalT']))  # growing season mean monthly temperature
countyKdd = np.array(list(usCounties['kdd']))
countyGdd = np.array(list(usCounties['gdd']))

# now these are in J/growing season/m2
countySlhf = -np.array(list(usCounties['seasonalSlhf']))
countySshf = -np.array(list(usCounties['seasonalSshf']))
countySsr = np.array(list(usCounties['seasonalSsr']))
countyStr = np.array(list(usCounties['seasonalStr']))

if wxData == 'era5':
    # convert from J/growing season/m2 to W/m2
    countySlhf /= np.matlib.repmat(countySeasonLenSec, 39, 1).T
    countySshf /= np.matlib.repmat(countySeasonLenSec, 39, 1).T
    countySsr /= np.matlib.repmat(countySeasonLenSec, 39, 1).T
    countyStr /= np.matlib.repmat(countySeasonLenSec, 39, 1).T

countyNetRad = (countySsr+countyStr)
countyU10 = np.array(list(usCounties['seasonalU10']))
countyV10 = np.array(list(usCounties['seasonalV10']))

if wxData == 'era5':
    countyT_Norm = normWithNan(countyT)
countyPr_Norm = normWithNan(countyPr)
countyKdd_Norm = normWithNan(countyKdd)
countyGdd_Norm = normWithNan(countyGdd)
countySlhf_Norm = normWithNan(countySlhf)
countySshf_Norm = normWithNan(countySshf)
countyNetRad_Norm = normWithNan(countyNetRad)
countyU10_Norm = normWithNan(countyU10)
countyV10_Norm = normWithNan(countyV10)
countyMaizeYield_Norm = normWithNan(countyMaizeYield)
countySoybeanYield_Norm = normWithNan(countySoybeanYield)
countyTotalProd_Norm = normWithNan(countyTotalProd)

NCounties = countyKdd.shape[0]
NYears = countyKdd.shape[1]

maizeYieldFromFeedback_SensTest = np.full([NCounties, n_samples, n_bootstraps, len(areaLimit)], np.nan)
soybeanYieldFromFeedback_SensTest = np.full([NCounties, n_samples, n_bootstraps, len(areaLimit)], np.nan)

for a, curAreaLimit in enumerate(areaLimit):
    
    if len(areaLimit) > 1:
        print('area limit %d%%'%curAreaLimit)

    lhTrendFrac = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    shTrendFrac = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    lhFromFeedback = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    shFromFeedback = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    if wxData == 'era5':
        tFromFeedback = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    kddTrendFrac = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    kddFromFeedback = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    gddTrendFrac = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    gddFromFeedback = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    maizeYieldTrendFrac = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    maizeYieldFromFeedback = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    soybeanYieldTrendFrac = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    soybeanYieldFromFeedback = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    prodFromFeedback = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    maizeYieldChgFeedbackWithAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    maizeYieldChgFeedbackWithoutAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    soybeanYieldChgFeedbackWithAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    soybeanYieldChgFeedbackWithoutAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    lhChgFeedbackWithAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    lhChgFeedbackWithoutAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    shChgFeedbackWithAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    shChgFeedbackWithoutAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    kddChgFeedbackWithAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    kddChgFeedbackWithoutAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    gddChgFeedbackWithAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    gddChgFeedbackWithoutAgInt = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    lhObsTrend = np.full([NCounties], np.nan)
    shObsTrend = np.full([NCounties], np.nan)
    netRadObsTrend = np.full([NCounties], np.nan)

    lhMod_yieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    lhMod_noYieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    lhModTrend_yieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    lhModTrend_noYieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    shMod_yieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    shMod_noYieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    shModTrend_yieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    shModTrend_noYieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    if wxData == 'era5':
        tMod_yieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
        tMod_noYieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
        tModTrend_yieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)
        tModTrend_noYieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    
    gddMod_yieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    gddMod_noYieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    gddModTrend_yieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    gddModTrend_noYieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    kddMod_yieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    kddMod_noYieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    kddModTrend_yieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    kddModTrend_noYieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    maizeYieldMod_yieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    maizeYieldMod_noYieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    maizeYieldModTrend_yieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    maizeYieldModTrend_noYieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    soybeanYieldMod_yieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    soybeanYieldMod_noYieldGrowth = np.full([NCounties, NYears, n_samples, n_bootstraps], np.nan)
    soybeanYieldModTrend_yieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)
    soybeanYieldModTrend_noYieldGrowth = np.full([NCounties, n_samples, n_bootstraps], np.nan)

    mdl_Param_Corr = {'Prod-Pr':np.full([NCounties, n_bootstraps], np.nan), 
                        'Prod-NetRad':np.full([NCounties, n_bootstraps], np.nan), 
                      'Prod-Wind':np.full([NCounties, n_bootstraps], np.nan), 
                      'Pr-NetRad':np.full([NCounties, n_bootstraps], np.nan), 
                      'Pr-Wind':np.full([NCounties, n_bootstraps], np.nan), 
                      'NetRad-Wind':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_LH_Y_Coefs = {'MaizeYield_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                      'SoybeanYield_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                      'TotalProd_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                           'Pr_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan),
                           'NetRad_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan),
                           'Wind_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                         'R2':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_LH_Y_PValues = {'MaizeYield_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                        'SoybeanYield_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                        'TotalProd_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                           'Pr_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan),
                           'NetRad_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan),
                           'Wind_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan)}
    

    mdl_LH_Y_Norm_Coefs = {'MaizeYield_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan), 
                           'SoybeanYield_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan), 
                           'TotalProd_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan), 
                           'Pr_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan),
                           'NetRad_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan),
                           'Wind_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_LH_Y_Norm_PValues = {'MaizeYield_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan), 
                             'SoybeanYield_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan), 
                             'TotalProd_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan), 
                           'Pr_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan),
                           'NetRad_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan),
                           'Wind_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_LH_Y_Decomp_Coefs = {'TotalYield_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                              'TotalHA':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_LH_Y_Decomp_PValues = {'TotalYield_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                              'TotalHA':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_LH_SH_Coefs = {'SLHF_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                           'NetRad_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan),
                          'R2':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_LH_SH_PValues = {'SLHF_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                           'NetRad_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_LH_SH_Norm_Coefs = {'SLHF_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan), 
                           'NetRad_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_LH_SH_Norm_PValues = {'SLHF_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan), 
                           'NetRad_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_SH_KDD_Coefs = {'SSHF_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan),
                       'R2':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_SH_KDD_PValues = {'SSHF_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_SH_GDD_Coefs = {'SSHF_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan), 
                       'R2':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_SH_GDD_PValues = {'SSHF_DetrendAnom':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_SH_KDD_Norm_Coefs = {'SSHF_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_SH_KDD_Norm_PValues = {'SSHF_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_SH_GDD_Norm_Coefs = {'SSHF_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_SH_GDD_Norm_PValues = {'SSHF_DetrendAnom_Norm':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_KDD_GDD_MaizeYield_Coefs = {'KDD_Detrend':np.full([NCounties, n_bootstraps], np.nan),
                                    'GDD_Detrend':np.full([NCounties, n_bootstraps], np.nan), 
                                    'Pr_Detrend':np.full([NCounties, n_bootstraps], np.nan),
                                    'R2':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_KDD_GDD_MaizeYield_PValues = {'KDD_Detrend':np.full([NCounties, n_bootstraps], np.nan),
                                      'GDD_Detrend':np.full([NCounties, n_bootstraps], np.nan), 
                                      'Pr_Detrend':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_KDD_GDD_MaizeYield_Norm_Coefs = {'KDD_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan),
                                    'GDD_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan), 
                                    'Pr_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan),
                                    'R2':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_KDD_GDD_MaizeYield_Norm_PValues = {'KDD_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan),
                                      'GDD_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan), 
                                      'Pr_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_KDD_GDD_SoybeanYield_Coefs = {'KDD_Detrend':np.full([NCounties, n_bootstraps], np.nan),
                                    'GDD_Detrend':np.full([NCounties, n_bootstraps], np.nan), 
                                    'Pr_Detrend':np.full([NCounties, n_bootstraps], np.nan),
                                    'R2':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_KDD_GDD_SoybeanYield_PValues = {'KDD_Detrend':np.full([NCounties, n_bootstraps], np.nan),
                                      'GDD_Detrend':np.full([NCounties, n_bootstraps], np.nan), 
                                      'Pr_Detrend':np.full([NCounties, n_bootstraps], np.nan)}

    mdl_KDD_GDD_SoybeanYield_Norm_Coefs = {'KDD_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan),
                                    'GDD_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan), 
                                    'Pr_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan),
                                    'R2':np.full([NCounties, n_bootstraps], np.nan)}
    mdl_KDD_GDD_SoybeanYield_Norm_PValues = {'KDD_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan),
                                      'GDD_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan), 
                                      'Pr_DetrendNorm':np.full([NCounties, n_bootstraps], np.nan)}
    
    mdl_LH_Y_CondNum = np.full([NCounties, n_bootstraps], np.nan)
    mdl_LH_SH_CondNum = np.full([NCounties, n_bootstraps], np.nan)
    mdl_SH_KDD_CondNum = np.full([NCounties, n_bootstraps], np.nan)
    mdl_SH_GDD_CondNum = np.full([NCounties, n_bootstraps], np.nan)
    mdl_KDD_GDD_PR_MaizeYield_CondNum = np.full([NCounties, n_bootstraps], np.nan)
    mdl_KDD_GDD_PR_SoybeanYield_CondNum = np.full([NCounties, n_bootstraps], np.nan)
    
    fipsSel = []
    fipsAll = []
    irSel = []
    haExclude = []
    shortSeriesExclude = []
    nonSigExclude = []
    stateSel = []

    for i in range(NCounties):

        fipsAll.append(countyFips[i])

        if i % 200 == 0:
            print('%d of %d'%(i, NCounties))

        curMaizeYieldDetrendAnom = countyMaizeYieldDetrendAnom[i,:]
        curMaizeYield = countyMaizeYield[i,:]
        curMaizeYield_Norm = countyMaizeYield_Norm[i,:]

        curSoybeanYieldDetrendAnom = countySoybeanYieldDetrendAnom[i,:]
        curSoybeanYield = countySoybeanYield[i,:]
        curSoybeanYield_Norm = countySoybeanYield_Norm[i,:]

        curTotalProd = countyTotalProd[i,:]
        curTotalProd_Norm = countyTotalProd_Norm[i,:]

        curTotalHaFrac = (haMaizeAllAc[i,:] + haSoybeanAllAc[i,:])/countyAcAll[i]

        curKdd = countyKdd[i,:]
        curKdd_Norm = countyKdd_Norm[i,:]
        curGdd = countyGdd[i,:]
        curGdd_Norm = countyGdd_Norm[i,:]
        if wxData == 'era5':
            curT = countyT[i,:]
            curT_Norm = countyT_Norm[i,:]
        curSshf = countySshf[i,:]
        curSshf_Norm = countySshf_Norm[i,:]
        curSlhf = countySlhf[i,:]
        curSlhf_Norm = countySlhf_Norm[i,:]
        curNetRad = countyNetRad[i,:]
        curNetRad_Norm = countyNetRad_Norm[i,:]
        curU10 = countyU10[i,:]
        curU10_Norm = countyU10_Norm[i,:]
        curV10 = countyV10[i,:]
        curV10_Norm = countyV10_Norm[i,:]
        curPr = countyPr[i,:]
        curPr_Norm = countyPr_Norm[i,:]

        nnProd = np.where((~np.isnan(curKdd)) & (~np.isnan(curPr)) & \
                          (~np.isnan(curTotalProd)) & (~np.isnan(curSshf)) & (~np.isnan(curSlhf)))[0]
        nnMaize = np.where((~np.isnan(curKdd)) & (~np.isnan(curPr)) & \
                          (~np.isnan(curMaizeYield)) & (~np.isnan(curSshf)) & (~np.isnan(curSlhf)))[0]
        nnSoybean = np.where((~np.isnan(curKdd)) & (~np.isnan(curPr)) & \
                          (~np.isnan(curSoybeanYield)) & (~np.isnan(curSshf)) & (~np.isnan(curSlhf)))[0]

        stateSel.append(countyState[i])

        exclude = False
        # identify counties excluded bc they have a short production time series
        if np.nansum(curTotalProd) == 0 or len(nnProd) < minCropYears:
            shortSeriesExclude.append(countyFips[i])
            fipsSel.append(np.nan)
            exclude = True

        # identify counties excluded bc they have too little harvested area
        if np.nanmean(countyMaizeHaFrac[i,:]+countySoybeanHaFrac[i,:]) < curAreaLimit:
            haExclude.append(countyFips[i])
            if not exclude: fipsSel.append(np.nan)
            exclude = True

        # identify counties that are irrigated
        if countyIr[i] > (irrLimit/100)*np.nanmean(countyMaizeHaFrac[i,:]+countySoybeanHaFrac[i,:]):
            irSel.append(i)
        else:
            irSel.append(np.nan)

        if exclude:
            continue
        
        # BOOTSTRAP INDEX SELECTION
        for b in range(n_bootstraps):
            nnProd_boot_ind = np.random.choice(np.arange(len(nnProd)), len(nnProd))
            nnMaize_boot_ind = np.random.choice(np.arange(len(nnMaize)), len(nnMaize))
            nnSoybean_boot_ind = np.random.choice(np.arange(len(nnSoybean)), len(nnSoybean))

            # remove linear intercept from yield time series
            if len(nnMaize) >= minCropYears:
                X = sm.add_constant(range(len(curMaizeYield[nnMaize])))
                mdl = sm.OLS(curMaizeYield[nnMaize], X).fit()
                curMaizeYieldIntercept = mdl.params[0]

                X = sm.add_constant(range(len(curMaizeYield_Norm[nnMaize])))
                mdl = sm.OLS(curMaizeYield_Norm[nnMaize], X).fit()
                curMaizeYieldIntercept_Norm = mdl.params[0]

            if len(nnSoybean) >= minCropYears:
                X = sm.add_constant(range(len(curSoybeanYield[nnSoybean])))
                mdl = sm.OLS(curSoybeanYield[nnSoybean], X).fit()
                curSoybeanYieldIntercept = mdl.params[0]

                X = sm.add_constant(range(len(curSoybeanYield_Norm[nnSoybean])))
                mdl = sm.OLS(curSoybeanYield_Norm[nnSoybean], X).fit()
                curSoybeanYieldIntercept_Norm = mdl.params[0]

            X = sm.add_constant(range(len(curTotalProd[nnProd])))
            mdl = sm.OLS(curTotalProd[nnProd], X).fit()
            curTotalProdIntercept = mdl.params[0]

            X = sm.add_constant(range(len(curTotalProd_Norm[nnProd])))
            mdl = sm.OLS(curTotalProd_Norm[nnProd], X).fit()
            curTotalProdIntercept_Norm = mdl.params[0]

            X = sm.add_constant(range(len(curPr)))
            mdl = sm.OLS(curPr, X).fit()
            curPrIntercept = mdl.params[0]

            X = sm.add_constant(range(len((curNetRad))))
            mdl = sm.OLS((curNetRad), X).fit()
            curNetRadIntercept = mdl.params[0]

            X = sm.add_constant(range(len(curU10)))
            mdl = sm.OLS(curU10, X).fit()
            curU10Intercept = mdl.params[0]

            X = sm.add_constant(range(len(curV10)))
            mdl = sm.OLS(curV10, X).fit()
            curV10Intercept = mdl.params[0]

            curWindProd = (curU10[nnProd]**2 + curV10[nnProd]**2)**.5
            curWindProd_Norm = (curU10_Norm[nnProd]**2 + curV10_Norm[nnProd]**2)**.5
            curWindMaize = (curU10[nnMaize]**2 + curV10[nnMaize]**2)**.5
            curWindMaize_Norm = (curU10_Norm[nnMaize]**2 + curV10_Norm[nnMaize]**2)**.5
            curWindSoybean = (curU10[nnSoybean]**2 + curV10[nnSoybean]**2)**.5
            curWindSoybean_Norm = (curU10_Norm[nnSoybean]**2 + curV10_Norm[nnSoybean]**2)**.5

            dataProd = {'KDD_DetrendAnom_Norm':scipy.signal.detrend(curKdd_Norm[nnProd]), \
                    'KDD_DetrendAnom':scipy.signal.detrend(curKdd[nnProd]), \
                    'KDD_DetrendAnom_SSA':ssaDetrend(curKdd[nnProd]), \
                    'KDD_DetrendNorm':scipy.signal.detrend(curKdd_Norm[nnProd])+np.nanmean(curKdd_Norm[nnProd]), \
                    'KDD_Detrend':scipy.signal.detrend(curKdd[nnProd])+np.nanmean(curKdd[nnProd]), \
                    'KDD':curKdd[nnProd], \
                    'GDD_DetrendAnom_Norm':scipy.signal.detrend(curGdd_Norm[nnProd]), \
                    'GDD_DetrendAnom':scipy.signal.detrend(curGdd[nnProd]), \
                    'GDD_DetrendAnom_SSA':ssaDetrend(curGdd[nnProd]), \
                    'GDD_DetrendNorm':scipy.signal.detrend(curGdd_Norm[nnProd])+np.nanmean(curGdd_Norm[nnProd]), \
                    'GDD_Detrend':scipy.signal.detrend(curGdd[nnProd])+np.nanmean(curGdd[nnProd]), \
                    'GDD':curGdd[nnProd], \
                    'T':curT[nnProd], \
                    'T_DetrendAnom_Norm':scipy.signal.detrend(curT_Norm[nnProd]), \
                    'T_DetrendAnom':scipy.signal.detrend(curT[nnProd]), \
                    'T_DetrendAnom_SSA':ssaDetrend(curT[nnProd]), \
                    'SSHF':curSshf[nnProd], \
                    'SSHF_DetrendAnom_Norm':scipy.signal.detrend(curSshf_Norm[nnProd]), \
                    'SSHF_DetrendAnom':scipy.signal.detrend(curSshf[nnProd]), \
                    'SSHF_DetrendAnom_SSA':ssaDetrend(curSshf[nnProd]), \
                    'SLHF_DetrendAnom_Norm':scipy.signal.detrend(curSlhf_Norm[nnProd]), \
                    'SLHF_DetrendAnom':scipy.signal.detrend(curSlhf[nnProd]), \
                    'SLHF_DetrendAnom_SSA':ssaDetrend(curSlhf[nnProd]), \
                    'SLHF':curSlhf[nnProd], \
                    'Wind_DetrendAnom_Norm':scipy.signal.detrend(curWindProd_Norm), \
                    'Wind_DetrendAnom':scipy.signal.detrend(curWindProd), \
                    'Wind_DetrendAnom_SSA':ssaDetrend(curWindProd), \
                    'Wind':curWindProd, \
                    'Pr_DetrendAnom_Norm':scipy.signal.detrend(curPr_Norm[nnProd]), \
                    'Pr_DetrendAnom':scipy.signal.detrend(curPr[nnProd]), \
                    'Pr_DetrendAnom_SSA':ssaDetrend(curPr[nnProd]), \
                    'Pr_Detrend':(scipy.signal.detrend(curPr[nnProd])+np.nanmean(curPr[nnProd])), \
                    'Pr_DetrendNorm':scipy.signal.detrend(curPr_Norm[nnProd])+np.nanmean(curPr_Norm[nnProd]), \
                    'Pr':curPr[nnProd], \
                    'NetRad_DetrendAnom_Norm':scipy.signal.detrend(curNetRad_Norm[nnProd]), \
                    'NetRad_DetrendAnom':scipy.signal.detrend(curNetRad[nnProd]), \
                    'NetRad_DetrendAnom_SSA':ssaDetrend(curNetRad[nnProd]), \
                    'NetRad':curNetRad[nnProd], \
                    'TotalYield_DetrendAnom':scipy.signal.detrend(curMaizeYield[nnProd]+curSoybeanYield[nnProd]), \
                    'TotalHA':scipy.signal.detrend(curTotalHaFrac[nnProd]), \
                    'TotalProd_DetrendAnom_SSA':ssaDetrend(curTotalProd[nnProd])/1e6, \
                    'TotalProd_DetrendAnom':scipy.signal.detrend(curTotalProd[nnProd])/1e6, \
                    'TotalProd_DetrendAnom_Norm_SSA':ssaDetrend(curTotalProd_Norm[nnProd]), \
                    'TotalProd_DetrendAnom_Norm':scipy.signal.detrend(curTotalProd_Norm[nnProd]), \
                    'TotalProd':curTotalProd[nnProd]/1e6, \
                    'TotalProd_Detrend':(scipy.signal.detrend(curTotalProd[nnProd])+curTotalProdIntercept)/1e6, \
                    'TotalProd_DetrendNorm':scipy.signal.detrend(curTotalProd_Norm[nnProd])+curTotalProdIntercept_Norm}

            dfProd = pd.DataFrame(dataProd, \
                              columns=['KDD_DetrendAnom_Norm', 'KDD_DetrendAnom', 'KDD_DetrendAnom_SSA', 'KDD', 'KDD_Detrend', 'KDD_DetrendNorm', \
                                       'GDD_DetrendAnom_Norm', 'GDD_DetrendAnom', 'GDD_DetrendAnom_SSA', 'GDD', 'GDD_Detrend', 'GDD_DetrendNorm', \
                                       'T_DetrendAnom_Norm', 'T_DetrendAnom', 'T_DetrendAnom_SSA', 'T', \
                                       'SLHF_DetrendAnom_Norm', 'SLHF_DetrendAnom', 'SLHF_DetrendAnom_SSA', 'SLHF', \
                                       'SSHF_DetrendAnom_Norm', 'SSHF_DetrendAnom', 'SSHF_DetrendAnom_SSA', 'SSHF', \
                                       'NetRad_DetrendAnom_Norm', 'NetRad_DetrendAnom', 'NetRad_DetrendAnom_SSA', 'NetRad', \
                                       'Pr_DetrendAnom_Norm', 'Pr_DetrendAnom', 'Pr_DetrendAnom_SSA', 'Pr', 'Pr_Detrend', 'Pr_DetrendNorm', \
                                       'Wind_DetrendAnom_Norm', 'Wind_DetrendAnom', 'Wind_DetrendAnom_SSA', 'Wind', \
                                       'TotalYield_DetrendAnom', 'TotalHA', \
                                       'TotalProd', 'TotalProd_Detrend', 'TotalProd_DetrendNorm', 'TotalProd_DetrendAnom', 'TotalProd_DetrendAnom_Norm', \
                                       'TotalProd_DetrendAnom_SSA', 'TotalProd_DetrendAnom_Norm_SSA'])
            
            dfProd_bootstrap = {'KDD_DetrendAnom':dfProd['KDD_DetrendAnom'][nnProd_boot_ind], \
                    'KDD_Detrend':dfProd['KDD_Detrend'][nnProd_boot_ind], \
                    'GDD_DetrendAnom':dfProd['GDD_DetrendAnom'][nnProd_boot_ind], \
                    'GDD_Detrend':dfProd['GDD_Detrend'][nnProd_boot_ind], \
                    'SSHF_DetrendAnom':dfProd['SSHF_DetrendAnom'][nnProd_boot_ind], \
                    'SLHF_DetrendAnom':dfProd['SLHF_DetrendAnom'][nnProd_boot_ind], \
                    'Wind_DetrendAnom':dfProd['Wind_DetrendAnom'][nnProd_boot_ind], \
                    'Pr_DetrendAnom':dfProd['Pr_DetrendAnom'][nnProd_boot_ind], \
                    'NetRad_DetrendAnom':dfProd['NetRad_DetrendAnom'][nnProd_boot_ind], \
                    'TotalProd_DetrendAnom':dfProd['TotalProd_DetrendAnom'][nnProd_boot_ind]}

            dfProd_bootstrap = pd.DataFrame(dfProd_bootstrap, \
                              columns=['KDD_DetrendAnom', 'KDD_Detrend', \
                                       'GDD_DetrendAnom', 'GDD_Detrend', \
                                       'SLHF_DetrendAnom', \
                                       'SSHF_DetrendAnom', \
                                       'NetRad_DetrendAnom', \
                                       'Pr_DetrendAnom', 'Pr_Detrend', \
                                       'Wind_DetrendAnom', \
                                       'TotalProd_DetrendAnom'])
            
            
            
            if len(nnMaize) >= minCropYears:
                dataMaize = {'KDD_DetrendAnom_Norm':scipy.signal.detrend(curKdd_Norm[nnMaize]), \
                        'KDD_DetrendAnom':scipy.signal.detrend(curKdd[nnMaize]), \
                        'KDD_DetrendNorm':scipy.signal.detrend(curKdd_Norm[nnMaize])+np.nanmean(curKdd_Norm[nnMaize]), \
                        'KDD_Detrend':scipy.signal.detrend(curKdd[nnMaize])+np.nanmean(curKdd[nnMaize]), \
                        'KDD':curKdd[nnMaize], \
                        'GDD_DetrendAnom_Norm':scipy.signal.detrend(curGdd_Norm[nnMaize]), \
                        'GDD_DetrendAnom':scipy.signal.detrend(curGdd[nnMaize]), \
                        'GDD_DetrendNorm':scipy.signal.detrend(curGdd_Norm[nnMaize])+np.nanmean(curGdd_Norm[nnMaize]), \
                        'GDD_Detrend':scipy.signal.detrend(curGdd[nnMaize])+np.nanmean(curGdd[nnMaize]), \
                        'GDD':curGdd[nnMaize], \
                        'SSHF':curSshf[nnMaize], \
                        'SSHF_DetrendAnom_Norm':scipy.signal.detrend(curSshf_Norm[nnMaize]), \
                        'SSHF_DetrendAnom':scipy.signal.detrend(curSshf[nnMaize]), \
                        'SLHF_DetrendAnom_Norm':scipy.signal.detrend(curSlhf_Norm[nnMaize]), \
                        'SLHF_DetrendAnom':scipy.signal.detrend(curSlhf[nnMaize]), \
                        'SLHF':curSlhf[nnMaize], \
                        'Wind_DetrendAnom_Norm':scipy.signal.detrend(curWindMaize_Norm), \
                        'Wind_DetrendAnom':scipy.signal.detrend(curWindMaize), \
                        'Wind':curWindMaize, \
                        'Pr_DetrendAnom_Norm':scipy.signal.detrend(curPr_Norm[nnMaize]), \
                        'Pr_DetrendAnom':scipy.signal.detrend(curPr[nnMaize]), \
                        'Pr_Detrend':(scipy.signal.detrend(curPr[nnMaize])+np.nanmean(curPr[nnMaize]))**2, \
                        'Pr_DetrendNorm':scipy.signal.detrend(curPr_Norm[nnMaize])+np.nanmean(curPr_Norm[nnMaize]), \
                        'Pr':curPr[nnMaize], \
                        'NetRad_DetrendAnom_Norm':scipy.signal.detrend(curNetRad_Norm[nnMaize]), \
                        'NetRad_DetrendAnom':scipy.signal.detrend(curNetRad[nnMaize]), \
                        'NetRad':curNetRad[nnMaize], \
                        'MaizeYield_DetrendAnom':scipy.signal.detrend(curMaizeYield[nnMaize]), \
                        'MaizeYield_DetrendAnom_Norm':scipy.signal.detrend(curMaizeYield_Norm[nnMaize]), \
                        'MaizeYield':curMaizeYield[nnMaize], \
                        'MaizeYield_Detrend':scipy.signal.detrend(curMaizeYield[nnMaize])+curMaizeYieldIntercept, \
                        'MaizeYield_DetrendNorm':scipy.signal.detrend(curMaizeYield_Norm[nnMaize])+curMaizeYieldIntercept_Norm}

                dfMaize = pd.DataFrame(dataMaize, \
                              columns=['KDD_DetrendAnom_Norm', 'KDD_DetrendAnom', 'KDD', 'KDD_Detrend', 'KDD_DetrendNorm', \
                                       'GDD_DetrendAnom_Norm', 'GDD_DetrendAnom', 'GDD', 'GDD_Detrend', 'GDD_DetrendNorm', \
                                       'SLHF_DetrendAnom_Norm', 'SLHF_DetrendAnom', 'SLHF', \
                                       'SSHF_DetrendAnom_Norm', 'SSHF_DetrendAnom', 'SSHF', \
                                       'NetRad_DetrendAnom_Norm', 'NetRad_DetrendAnom', 'NetRad', \
                                       'Pr_DetrendAnom_Norm', 'Pr_DetrendAnom', 'Pr', 'Pr_Detrend', 'Pr_DetrendNorm', \
                                       'Wind_DetrendAnom_Norm', 'Wind_DetrendAnom', 'Wind', \
                                       'MaizeYield', 'MaizeYield_Detrend', 'MaizeYield_DetrendNorm', 'MaizeYield_DetrendAnom', 'MaizeYield_DetrendAnom_Norm'])
                
                dataMaize_bootstrap = {'KDD_Detrend':dfMaize['KDD_Detrend'][nnMaize_boot_ind], \
                        'GDD_Detrend':dfMaize['GDD_Detrend'][nnMaize_boot_ind], \
                        'Pr_Detrend':dfMaize['Pr_Detrend'][nnMaize_boot_ind], \
                        'MaizeYield_Detrend':dfMaize['MaizeYield_Detrend'][nnMaize_boot_ind]}

                dfMaize_bootstrap = pd.DataFrame(dataMaize_bootstrap, \
                              columns=['KDD_Detrend', \
                                       'GDD_Detrend', \
                                       'Pr_Detrend', \
                                       'MaizeYield_Detrend'])

            if len(nnSoybean) >= minCropYears:
                dataSoybean = {'KDD_DetrendAnom_Norm':scipy.signal.detrend(curKdd_Norm[nnSoybean]), \
                        'KDD_DetrendAnom':scipy.signal.detrend(curKdd[nnSoybean]), \
                        'KDD_DetrendNorm':scipy.signal.detrend(curKdd_Norm[nnSoybean])+np.nanmean(curKdd_Norm[nnSoybean]), \
                        'KDD_Detrend':scipy.signal.detrend(curKdd[nnSoybean])+np.nanmean(curKdd[nnSoybean]), \
                        'KDD':curKdd[nnSoybean], \
                        'GDD_DetrendAnom_Norm':scipy.signal.detrend(curGdd_Norm[nnSoybean]), \
                        'GDD_DetrendAnom':scipy.signal.detrend(curGdd[nnSoybean]), \
                        'GDD_DetrendNorm':scipy.signal.detrend(curGdd_Norm[nnSoybean])+np.nanmean(curGdd_Norm[nnSoybean]), \
                        'GDD_Detrend':scipy.signal.detrend(curGdd[nnSoybean])+np.nanmean(curGdd[nnSoybean]), \
                        'GDD':curGdd[nnSoybean], \
                        'SSHF':curSshf[nnSoybean], \
                        'SSHF_DetrendAnom_Norm':scipy.signal.detrend(curSshf_Norm[nnSoybean]), \
                        'SSHF_DetrendAnom':scipy.signal.detrend(curSshf[nnSoybean]), \
                        'SLHF_DetrendAnom_Norm':scipy.signal.detrend(curSlhf_Norm[nnSoybean]), \
                        'SLHF_DetrendAnom':scipy.signal.detrend(curSlhf[nnSoybean]), \
                        'SLHF':curSlhf[nnSoybean], \
                        'Wind_DetrendAnom_Norm':scipy.signal.detrend(curWindSoybean_Norm), \
                        'Wind_DetrendAnom':scipy.signal.detrend(curWindSoybean), \
                        'Wind':curWindSoybean, \
                        'Pr_DetrendAnom_Norm':scipy.signal.detrend(curPr_Norm[nnSoybean]), \
                        'Pr_DetrendAnom':scipy.signal.detrend(curPr[nnSoybean]), \
                        'Pr_Detrend':(scipy.signal.detrend(curPr[nnSoybean])+np.nanmean(curPr[nnSoybean]))**2, \
                        'Pr_DetrendNorm':scipy.signal.detrend(curPr_Norm[nnSoybean])+np.nanmean(curPr_Norm[nnSoybean]), \
                        'Pr':curPr[nnSoybean], \
                        'NetRad_DetrendAnom_Norm':scipy.signal.detrend(curNetRad_Norm[nnSoybean]), \
                        'NetRad_DetrendAnom':scipy.signal.detrend(curNetRad[nnSoybean]), \
                        'NetRad':curNetRad[nnSoybean], \
                        'SoybeanYield_DetrendAnom':scipy.signal.detrend(curSoybeanYield[nnSoybean]), \
                        'SoybeanYield_DetrendAnom_Norm':scipy.signal.detrend(curSoybeanYield_Norm[nnSoybean]), \
                        'SoybeanYield':curSoybeanYield[nnSoybean], \
                        'SoybeanYield_Detrend':scipy.signal.detrend(curSoybeanYield[nnSoybean])+curSoybeanYieldIntercept, \
                        'SoybeanYield_DetrendNorm':scipy.signal.detrend(curSoybeanYield_Norm[nnSoybean])+curSoybeanYieldIntercept_Norm}

                dfSoybean = pd.DataFrame(dataSoybean, \
                              columns=['KDD_DetrendAnom_Norm', 'KDD_DetrendAnom', 'KDD', 'KDD_Detrend', 'KDD_DetrendNorm', \
                                       'GDD_DetrendAnom_Norm', 'GDD_DetrendAnom', 'GDD', 'GDD_Detrend', 'GDD_DetrendNorm', \
                                       'SLHF_DetrendAnom_Norm', 'SLHF_DetrendAnom', 'SLHF', \
                                       'SSHF_DetrendAnom_Norm', 'SSHF_DetrendAnom', 'SSHF', \
                                       'NetRad_DetrendAnom_Norm', 'NetRad_DetrendAnom', 'NetRad', \
                                       'Pr_DetrendAnom_Norm', 'Pr_DetrendAnom', 'Pr', 'Pr_Detrend', 'Pr_DetrendNorm', \
                                       'Wind_DetrendAnom_Norm', 'Wind_DetrendAnom', 'Wind', \
                                       'SoybeanYield', 'SoybeanYield_Detrend', 'SoybeanYield_DetrendNorm', 'SoybeanYield_DetrendAnom', 'SoybeanYield_DetrendAnom_Norm'])
                
                dataSoybean_bootstrap = {'KDD_Detrend':dfSoybean['KDD_Detrend'][nnSoybean_boot_ind], \
                        'GDD_Detrend':dfSoybean['GDD_Detrend'][nnSoybean_boot_ind], \
                        'Pr_Detrend':dfSoybean['Pr_Detrend'][nnSoybean_boot_ind], \
                        'SoybeanYield_Detrend':dfSoybean['SoybeanYield_Detrend'][nnSoybean_boot_ind]}

                dfSoybean_bootstrap = pd.DataFrame(dataSoybean_bootstrap, \
                              columns=['KDD_Detrend', \
                                       'GDD_Detrend', \
                                       'Pr_Detrend', \
                                       'SoybeanYield_Detrend'])



            if useSsa:
                lhVar = 'SLHF_DetrendAnom_SSA'
                shVar = 'SSHF_DetrendAnom_SSA'
                prodVar = 'TotalProd_DetrendAnom_SSA'
                prVar = 'Pr_DetrendAnom_SSA'
                netRadVar = 'NetRad_DetrendAnom_SSA'
                windVar = 'Wind_DetrendAnom_SSA'
                kddVar = 'KDD_DetrendAnom_SSA'
                gddVar = 'GDD_DetrendAnom_SSA'
                tVar = 'T_DetrendAnom_SSA'

                prodVar_Norm = 'TotalProd_DetrendAnom_Norm_SSA'
            else:
                lhVar = 'SLHF_DetrendAnom'
                shVar = 'SSHF_DetrendAnom'
                prodVar = 'TotalProd_DetrendAnom'
                prVar = 'Pr_DetrendAnom'
                netRadVar = 'NetRad_DetrendAnom'
                windVar = 'Wind_DetrendAnom'
                kddVar = 'KDD_DetrendAnom'
                gddVar = 'GDD_DetrendAnom'
                tVar = 'T_DetrendAnom'

                prodVar_Norm = 'TotalProd_DetrendAnom_Norm'

            if uncertaintyProp:
                if includeKdd:
                    mdl_LH_Y = smf.ols(formula='%s ~ %s + %s + %s + %s + %s'%(lhVar, prodVar, prVar, netRadVar, windVar, kddVar), \
                                       data=dfProd_bootstrap).fit()
                else:
                    mdl_LH_Y = smf.ols(formula='%s ~ %s + %s + %s + %s'%(lhVar, prodVar, prVar, netRadVar, windVar), \
                                       data=dfProd_bootstrap).fit()
            else:
                mdl_LH_Y = smf.ols(formula='%s ~ %s + %s + %s + %s + %s'%(lhVar, prodVar, prVar, netRadVar, windVar, kddVar), \
                                   data=dfProd).fit()
            
            mdl_LH_Y_Norm = smf.ols(formula='SLHF_DetrendAnom_Norm ~ %s + Pr_DetrendAnom_Norm + NetRad_DetrendAnom_Norm + Wind_DetrendAnom_Norm'%prodVar_Norm, \
                                    data=dfProd).fit()

            mdl_LH_Y_Decomp = smf.ols(formula='SLHF_DetrendAnom ~ TotalHA + TotalYield_DetrendAnom + Pr_DetrendAnom + NetRad_DetrendAnom + Wind_DetrendAnom', \
                               data=dfProd).fit()

            
            mdl_Param_Corr['Prod-Pr'][i, b] = np.corrcoef(dfProd[prodVar], dfProd[prVar])[0,1]
            mdl_Param_Corr['Prod-NetRad'][i, b] = np.corrcoef(dfProd[prodVar], dfProd[netRadVar])[0,1]
            mdl_Param_Corr['Prod-Wind'][i, b] = np.corrcoef(dfProd[prodVar], dfProd[windVar])[0,1]
            mdl_Param_Corr['Pr-NetRad'][i, b] = np.corrcoef(dfProd[prVar], dfProd[netRadVar])[0,1]
            mdl_Param_Corr['Pr-Wind'][i, b] = np.corrcoef(dfProd[prVar], dfProd[windVar])[0,1]
            mdl_Param_Corr['NetRad-Wind'][i, b] = np.corrcoef(dfProd[netRadVar], dfProd[windVar])[0,1]

            mdl_LH_Y_Decomp_Coefs['TotalYield_DetrendAnom'][i, b] = mdl_LH_Y_Decomp.params['TotalYield_DetrendAnom']
            mdl_LH_Y_Decomp_Coefs['TotalHA'][i, b] = mdl_LH_Y_Decomp.params['TotalHA']
            mdl_LH_Y_Decomp_PValues['TotalYield_DetrendAnom'][i, b] = mdl_LH_Y_Decomp.pvalues['TotalYield_DetrendAnom']
            mdl_LH_Y_Decomp_PValues['TotalHA'][i, b] = mdl_LH_Y_Decomp.pvalues['TotalHA']

            if uncertaintyProp:
                mdl_LH_SH = smf.ols(formula='%s ~ %s + %s'%(shVar, lhVar, netRadVar), data=dfProd_bootstrap).fit()
                mdl_SH_KDD = smf.ols(formula='%s ~ %s'%(kddVar, shVar), data=dfProd_bootstrap).fit()
                mdl_SH_GDD = smf.ols(formula='%s ~ %s'%(gddVar, shVar), data=dfProd_bootstrap).fit()
            else:
                mdl_LH_SH = smf.ols(formula='%s ~ %s + %s'%(shVar, lhVar, netRadVar), data=dfProd).fit()
                mdl_SH_KDD = smf.ols(formula='%s ~ %s'%(kddVar, shVar), data=dfProd).fit()
                mdl_SH_GDD = smf.ols(formula='%s ~ %s'%(gddVar, shVar), data=dfProd).fit()
                
            mdl_LH_SH_Norm = smf.ols(formula='SSHF_DetrendAnom_Norm ~ SLHF_DetrendAnom_Norm + NetRad_DetrendAnom_Norm', data=dfProd).fit()
            mdl_SH_KDD_Norm = smf.ols(formula='KDD_DetrendAnom_Norm ~ SSHF_DetrendAnom_Norm', data=dfProd).fit()
            mdl_SH_GDD_Norm = smf.ols(formula='GDD_DetrendAnom_Norm ~ SSHF_DetrendAnom_Norm', data=dfProd).fit()

            if wxData == 'era5':
                mdl_SH_T = smf.ols(formula='%s ~ %s'%(tVar, shVar), data=dfProd).fit()

            if len(nnMaize) >= minCropYears:
                if uncertaintyProp:
                    mdl_KDD_GDD_PR_MaizeYield = smf.ols(formula='MaizeYield_Detrend ~ GDD_Detrend + KDD_Detrend + Pr_Detrend', data=dfMaize_bootstrap).fit()
                else:
                    mdl_KDD_GDD_PR_MaizeYield = smf.ols(formula='MaizeYield_Detrend ~ GDD_Detrend + KDD_Detrend + Pr_Detrend', data=dfMaize).fit()
                mdl_KDD_GDD_PR_MaizeYield_Norm = smf.ols(formula='MaizeYield_DetrendNorm ~ GDD_DetrendNorm + KDD_DetrendNorm + Pr_DetrendNorm', data=dfMaize).fit()

            if len(nnSoybean) >= minCropYears:
                if uncertaintyProp:
                    mdl_KDD_GDD_PR_SoybeanYield = smf.ols(formula='SoybeanYield_Detrend ~ GDD_Detrend + KDD_Detrend + Pr_Detrend', data=dfSoybean_bootstrap).fit()
                else:
                    mdl_KDD_GDD_PR_SoybeanYield = smf.ols(formula='SoybeanYield_Detrend ~ GDD_Detrend + KDD_Detrend + Pr_Detrend', data=dfSoybean).fit()
                mdl_KDD_GDD_PR_SoybeanYield_Norm = smf.ols(formula='SoybeanYield_DetrendNorm ~ GDD_DetrendNorm + KDD_DetrendNorm + Pr_DetrendNorm', data=dfSoybean).fit()
            
            mdl_LH_Y_CondNum[i, b] = mdl_LH_Y.condition_number
            mdl_LH_SH_CondNum[i, b] = mdl_LH_SH.condition_number
            mdl_SH_KDD_CondNum[i, b] = mdl_SH_KDD.condition_number
            mdl_SH_GDD_CondNum[i, b] = mdl_SH_GDD.condition_number
            mdl_KDD_GDD_PR_MaizeYield_CondNum[i, b] = mdl_KDD_GDD_PR_MaizeYield.condition_number
            mdl_KDD_GDD_PR_SoybeanYield_CondNum[i, b] = mdl_KDD_GDD_PR_SoybeanYield.condition_number

        #     mdl_LH_Y_Coefs['Yield_DetrendAnom'][i] = mdl_LH_Y.params['Yield_DetrendAnom']
            mdl_LH_Y_Coefs['TotalProd_DetrendAnom'][i, b] = mdl_LH_Y.params[prodVar]
            mdl_LH_Y_Coefs['Pr_DetrendAnom'][i, b] = mdl_LH_Y.params[prVar]
            mdl_LH_Y_Coefs['NetRad_DetrendAnom'][i, b] = mdl_LH_Y.params[netRadVar]
            mdl_LH_Y_Coefs['Wind_DetrendAnom'][i, b] = mdl_LH_Y.params[windVar]
            mdl_LH_Y_Coefs['R2'][i, b] = mdl_LH_Y.rsquared
        #     mdl_LH_Y_PValues['Yield_DetrendAnom'][i] = mdl_LH_Y.pvalues['Yield_DetrendAnom']
            mdl_LH_Y_PValues['TotalProd_DetrendAnom'][i, b] = mdl_LH_Y.pvalues[prodVar]
            mdl_LH_Y_PValues['Pr_DetrendAnom'][i, b] = mdl_LH_Y.pvalues[prVar]
            mdl_LH_Y_PValues['NetRad_DetrendAnom'][i, b] = mdl_LH_Y.pvalues[netRadVar]
            mdl_LH_Y_PValues['Wind_DetrendAnom'][i, b] = mdl_LH_Y.pvalues[windVar]

        #     mdl_LH_Y_Norm_Coefs['Yield_DetrendAnom_Norm'][i] = mdl_LH_Y_Norm.params['Yield_DetrendAnom_Norm']
            mdl_LH_Y_Norm_Coefs['TotalProd_DetrendAnom_Norm'][i, b] = mdl_LH_Y_Norm.params[prodVar_Norm]
            mdl_LH_Y_Norm_Coefs['Pr_DetrendAnom_Norm'][i, b] = mdl_LH_Y_Norm.params['Pr_DetrendAnom_Norm']
            mdl_LH_Y_Norm_Coefs['NetRad_DetrendAnom_Norm'][i, b] = mdl_LH_Y_Norm.params['NetRad_DetrendAnom_Norm']
            mdl_LH_Y_Norm_Coefs['Wind_DetrendAnom_Norm'][i, b] = mdl_LH_Y_Norm.params['Wind_DetrendAnom_Norm']
        #     mdl_LH_Y_Norm_PValues['Yield_DetrendAnom_Norm'][i] = mdl_LH_Y_Norm.pvalues['Yield_DetrendAnom_Norm']
            mdl_LH_Y_Norm_PValues['TotalProd_DetrendAnom_Norm'][i, b] = mdl_LH_Y_Norm.pvalues[prodVar_Norm]
            mdl_LH_Y_Norm_PValues['Pr_DetrendAnom_Norm'][i, b] = mdl_LH_Y_Norm.pvalues['Pr_DetrendAnom_Norm']
            mdl_LH_Y_Norm_PValues['NetRad_DetrendAnom_Norm'][i, b] = mdl_LH_Y_Norm.pvalues['NetRad_DetrendAnom_Norm']
            mdl_LH_Y_Norm_PValues['Wind_DetrendAnom_Norm'][i, b] = mdl_LH_Y_Norm.pvalues['Wind_DetrendAnom_Norm']

            mdl_LH_SH_Coefs['SLHF_DetrendAnom'][i, b] = mdl_LH_SH.params[lhVar]
            mdl_LH_SH_Coefs['NetRad_DetrendAnom'][i, b] = mdl_LH_SH.params[netRadVar]
            mdl_LH_SH_Coefs['R2'][i, b] = mdl_LH_SH.rsquared
            mdl_LH_SH_PValues['SLHF_DetrendAnom'][i, b] = mdl_LH_SH.pvalues[lhVar]
            mdl_LH_SH_PValues['NetRad_DetrendAnom'][i, b] = mdl_LH_SH.pvalues[netRadVar]

            mdl_LH_SH_Norm_Coefs['SLHF_DetrendAnom_Norm'][i, b] = mdl_LH_SH_Norm.params['SLHF_DetrendAnom_Norm']
            mdl_LH_SH_Norm_Coefs['NetRad_DetrendAnom_Norm'][i, b] = mdl_LH_SH_Norm.params['NetRad_DetrendAnom_Norm']
            mdl_LH_SH_Norm_PValues['SLHF_DetrendAnom_Norm'][i, b] = mdl_LH_SH_Norm.pvalues['SLHF_DetrendAnom_Norm']
            mdl_LH_SH_Norm_PValues['NetRad_DetrendAnom_Norm'][i, b] = mdl_LH_SH_Norm.pvalues['NetRad_DetrendAnom_Norm']

            mdl_SH_KDD_Coefs['SSHF_DetrendAnom'][i, b] = mdl_SH_KDD.params[shVar]
            mdl_SH_KDD_Coefs['R2'][i, b] = mdl_SH_KDD.rsquared
            mdl_SH_GDD_Coefs['SSHF_DetrendAnom'][i, b] = mdl_SH_GDD.params[shVar]
            mdl_SH_GDD_Coefs['R2'][i, b] = mdl_SH_GDD.rsquared
            mdl_SH_KDD_PValues['SSHF_DetrendAnom'][i, b] = mdl_SH_KDD.pvalues[shVar]
            mdl_SH_GDD_PValues['SSHF_DetrendAnom'][i, b] = mdl_SH_GDD.pvalues[shVar]

            mdl_SH_KDD_Norm_Coefs['SSHF_DetrendAnom_Norm'][i, b] = mdl_SH_KDD_Norm.params['SSHF_DetrendAnom_Norm']
            mdl_SH_GDD_Norm_Coefs['SSHF_DetrendAnom_Norm'][i, b] = mdl_SH_GDD_Norm.params['SSHF_DetrendAnom_Norm']
            mdl_SH_KDD_Norm_PValues['SSHF_DetrendAnom_Norm'][i, b] = mdl_SH_KDD_Norm.pvalues['SSHF_DetrendAnom_Norm']
            mdl_SH_GDD_Norm_PValues['SSHF_DetrendAnom_Norm'][i, b] = mdl_SH_GDD_Norm.pvalues['SSHF_DetrendAnom_Norm']

            if len(nnMaize) >= minCropYears:
                mdl_KDD_GDD_MaizeYield_Coefs['KDD_Detrend'][i, b] = mdl_KDD_GDD_PR_MaizeYield.params['KDD_Detrend']
                mdl_KDD_GDD_MaizeYield_Coefs['GDD_Detrend'][i, b] = mdl_KDD_GDD_PR_MaizeYield.params['GDD_Detrend']
                mdl_KDD_GDD_MaizeYield_Coefs['Pr_Detrend'][i, b] = mdl_KDD_GDD_PR_MaizeYield.params['Pr_Detrend']
                mdl_KDD_GDD_MaizeYield_Coefs['R2'][i, b] = mdl_KDD_GDD_PR_MaizeYield.rsquared
                mdl_KDD_GDD_MaizeYield_PValues['KDD_Detrend'][i, b] = mdl_KDD_GDD_PR_MaizeYield.pvalues['KDD_Detrend']
                mdl_KDD_GDD_MaizeYield_PValues['GDD_Detrend'][i, b] = mdl_KDD_GDD_PR_MaizeYield.pvalues['GDD_Detrend']
                mdl_KDD_GDD_MaizeYield_PValues['Pr_Detrend'][i, b] = mdl_KDD_GDD_PR_MaizeYield.pvalues['Pr_Detrend']

                mdl_KDD_GDD_MaizeYield_Norm_Coefs['KDD_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_MaizeYield_Norm.params['KDD_DetrendNorm']
                mdl_KDD_GDD_MaizeYield_Norm_Coefs['GDD_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_MaizeYield_Norm.params['GDD_DetrendNorm']
                mdl_KDD_GDD_MaizeYield_Norm_Coefs['Pr_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_MaizeYield_Norm.params['Pr_DetrendNorm']
                mdl_KDD_GDD_MaizeYield_Norm_Coefs['R2'][i, b] = mdl_KDD_GDD_PR_MaizeYield_Norm.rsquared
                mdl_KDD_GDD_MaizeYield_Norm_PValues['KDD_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_MaizeYield_Norm.pvalues['KDD_DetrendNorm']
                mdl_KDD_GDD_MaizeYield_Norm_PValues['GDD_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_MaizeYield_Norm.pvalues['GDD_DetrendNorm']
                mdl_KDD_GDD_MaizeYield_Norm_PValues['Pr_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_MaizeYield_Norm.pvalues['Pr_DetrendNorm']

            if len(nnSoybean) >= minCropYears:
                mdl_KDD_GDD_SoybeanYield_Coefs['KDD_Detrend'][i, b] = mdl_KDD_GDD_PR_SoybeanYield.params['KDD_Detrend']
                mdl_KDD_GDD_SoybeanYield_Coefs['GDD_Detrend'][i, b] = mdl_KDD_GDD_PR_SoybeanYield.params['GDD_Detrend']
                mdl_KDD_GDD_SoybeanYield_Coefs['Pr_Detrend'][i, b] = mdl_KDD_GDD_PR_SoybeanYield.params['Pr_Detrend']
                mdl_KDD_GDD_SoybeanYield_Coefs['R2'][i, b] = mdl_KDD_GDD_PR_SoybeanYield.rsquared
                mdl_KDD_GDD_SoybeanYield_PValues['KDD_Detrend'][i, b] = mdl_KDD_GDD_PR_SoybeanYield.pvalues['KDD_Detrend']
                mdl_KDD_GDD_SoybeanYield_PValues['GDD_Detrend'][i, b] = mdl_KDD_GDD_PR_SoybeanYield.pvalues['GDD_Detrend']
                mdl_KDD_GDD_SoybeanYield_PValues['Pr_Detrend'][i, b] = mdl_KDD_GDD_PR_SoybeanYield.pvalues['Pr_Detrend']

                mdl_KDD_GDD_SoybeanYield_Norm_Coefs['KDD_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_SoybeanYield_Norm.params['KDD_DetrendNorm']
                mdl_KDD_GDD_SoybeanYield_Norm_Coefs['GDD_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_SoybeanYield_Norm.params['GDD_DetrendNorm']
                mdl_KDD_GDD_SoybeanYield_Norm_Coefs['Pr_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_SoybeanYield_Norm.params['Pr_DetrendNorm']
                mdl_KDD_GDD_SoybeanYield_Norm_Coefs['R2'][i, b] = mdl_KDD_GDD_PR_SoybeanYield_Norm.rsquared
                mdl_KDD_GDD_SoybeanYield_Norm_PValues['KDD_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_SoybeanYield_Norm.pvalues['KDD_DetrendNorm']
                mdl_KDD_GDD_SoybeanYield_Norm_PValues['GDD_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_SoybeanYield_Norm.pvalues['GDD_DetrendNorm']
                mdl_KDD_GDD_SoybeanYield_Norm_PValues['Pr_DetrendNorm'][i, b] = mdl_KDD_GDD_PR_SoybeanYield_Norm.pvalues['Pr_DetrendNorm']

            sample_x = np.linspace(0.01, 0.99, n_samples)
                
            # with yield growth and warming - this is just the historical LH trend
            curLhMod_yieldGrowth = np.matlib.repmat(curSlhf[nnProd], n_samples, 1).T
            lhMod_yieldGrowth[i, 0:len(curLhMod_yieldGrowth), :, b] = curLhMod_yieldGrowth
            
            # now calculate how much yield variation contributed to latent heat using our regression (above)
            # trained with detrended anomalies. run this using yield variation (minus the 1981 intercept), so we
            # capture both variability and trend
        #     X = {'MaizeYield_DetrendAnom':data['MaizeYield']-curMaizeYieldIntercept,
        #          'Pr_DetrendAnom':[0]*len(data['MaizeYield']),
        #          'NetRad_DetrendAnom':[0]*len(data['MaizeYield']),
        #          'U10_DetrendAnom':[0]*len(data['MaizeYield']),
        #          'V10_DetrendAnom':[0]*len(data['MaizeYield'])}
            X = {prodVar:dataProd['TotalProd']-curTotalProdIntercept/1e6,
                 prVar:[0]*len(dataProd['TotalProd']),
                 netRadVar:[0]*len(dataProd['TotalProd']),
                 windVar:[0]*len(dataProd['TotalProd']),
                 kddVar:[0]*len(dataProd['TotalProd'])}

            mdl_LH_Y_TotalProd_sample = mdl_LH_Y.params['TotalProd_DetrendAnom'] + \
                                        st.t.ppf(sample_x, mdl_LH_Y.df_model) * mdl_LH_Y.bse['TotalProd_DetrendAnom']

            lhFromYieldGrowth = np.full([len(X['TotalProd_DetrendAnom']), n_samples], np.nan)

            # loop over sampled coordinates
            for k in range(n_samples):
                lhFromYieldGrowth[:, k] = mdl_LH_Y.params['Intercept'] + \
                            mdl_LH_Y_TotalProd_sample[k]*np.array(X['TotalProd_DetrendAnom']) + \
                            mdl_LH_Y.params['Pr_DetrendAnom']*np.array(X['Pr_DetrendAnom']) + \
                            mdl_LH_Y.params['NetRad_DetrendAnom']*np.array(X['NetRad_DetrendAnom']) + \
                            mdl_LH_Y.params['Wind_DetrendAnom']*np.array(X['Wind_DetrendAnom'])

            # the lh without yield growth is just the total lh minus this estimated lh that
            # resulted from yield growth
            curLhMod_noYieldGrowth = np.matlib.repmat(curSlhf[nnProd], n_samples, 1).T - lhFromYieldGrowth
            lhMod_noYieldGrowth[i, 0:len(curLhMod_noYieldGrowth), :, b] = curLhMod_noYieldGrowth
            

    #         lhFromYieldGrowth1 = mdl_LH_Y.predict(X).values


            # the lh without yield growth is just the total lh minus this estimated lh that
            # resulted from yield growth
    #         curLhMod_noYieldGrowth = curSlhf[nnProd] - lhFromYieldGrowth
    #         lhMod_noYieldGrowth[i, 0:len(curLhMod_noYieldGrowth)] = curLhMod_noYieldGrowth

            # now do the same for sh
            curShMod_yieldGrowth = np.matlib.repmat(curSshf[nnProd], n_samples, 1).T
            shMod_yieldGrowth[i, 0:len(curShMod_yieldGrowth), :, b] = curShMod_yieldGrowth

            # how much did the lh change (from yield) change sh - estimate using the lh produced by yield 
            # from above
            mdl_LH_SH_LH_sample = mdl_LH_SH.params[lhVar] + \
                                        st.t.ppf(sample_x, mdl_LH_SH.df_model) * mdl_LH_SH.bse[lhVar]
            lh_coef_sample = np.random.choice(n_samples, n_samples)
            shFromYieldGrowth = np.full([len(X['TotalProd_DetrendAnom']), n_samples], np.nan)
            for k in range(n_samples):
                X = {lhVar:lhFromYieldGrowth[:,lh_coef_sample[k]],
                     netRadVar:[0]*lhFromYieldGrowth.shape[0]}

                shFromYieldGrowth[:, k] = mdl_LH_SH.params['Intercept'] + \
                                            mdl_LH_SH_LH_sample[k]*np.array(X[lhVar]) + \
                                            mdl_LH_SH.params[netRadVar]*np.array(X[netRadVar])

            curShMod_noYieldGrowth = np.matlib.repmat(curSshf[nnProd], n_samples, 1).T - shFromYieldGrowth
            shMod_noYieldGrowth[i, 0:len(curShMod_noYieldGrowth), :, b] = curShMod_noYieldGrowth


    #             shFromYieldGrowth = mdl_LH_SH.predict(X).values
    #             curShMod_noYieldGrowth = curSshf[nnProd] - shFromYieldGrowth
    #             shMod_noYieldGrowth[i, 0:len(curShMod_noYieldGrowth)] = curShMod_noYieldGrowth

            # with yield growth - gdd/kdd are observed values
            curKddMod_yieldGrowth = np.matlib.repmat(curKdd[nnProd], n_samples, 1).T
            kddMod_yieldGrowth[i, 0:len(curKddMod_yieldGrowth), :, b] = curKddMod_yieldGrowth
            curGddMod_yieldGrowth = np.matlib.repmat(curGdd[nnProd], n_samples, 1).T
            gddMod_yieldGrowth[i, 0:len(curGddMod_yieldGrowth), :, b] = curGddMod_yieldGrowth


            # without yield growth - predict gdd/kdd change from sh change
            # resulting from yield, estimated above
            mdl_SH_KDD_SH_sample = mdl_SH_KDD.params[shVar] + \
                                        st.t.ppf(sample_x, mdl_SH_KDD.df_model) * mdl_SH_KDD.bse[shVar]
            mdl_SH_GDD_SH_sample = mdl_SH_GDD.params[shVar] + \
                                        st.t.ppf(sample_x, mdl_SH_GDD.df_model) * mdl_SH_GDD.bse[shVar]
            sh_coef_sample = np.random.choice(n_samples, n_samples)
            kddFromYieldGrowth = np.full([len(X[lhVar]), n_samples], np.nan)
            gddFromYieldGrowth = np.full([len(X[lhVar]), n_samples], np.nan)

            for k in range(n_samples):
                X = {shVar:shFromYieldGrowth[:, sh_coef_sample[k]]}

                kddFromYieldGrowth[:, k] = mdl_SH_KDD.params['Intercept'] + \
                                            mdl_SH_KDD_SH_sample[k]*np.array(X[shVar])
                gddFromYieldGrowth[:, k] = mdl_SH_GDD.params['Intercept'] + \
                                            mdl_SH_GDD_SH_sample[k]*np.array(X[shVar])

            curKddMod_noYieldGrowth = np.matlib.repmat(curKdd[nnProd], n_samples, 1).T - kddFromYieldGrowth
            kddMod_noYieldGrowth[i, 0:len(curKddMod_noYieldGrowth), :, b] = curKddMod_noYieldGrowth

            curGddMod_noYieldGrowth = np.matlib.repmat(curGdd[nnProd], n_samples, 1).T - gddFromYieldGrowth
            gddMod_noYieldGrowth[i, 0:len(curGddMod_noYieldGrowth), :, b] = curGddMod_noYieldGrowth


    #             kddFromYieldGrowth = mdl_SH_KDD.predict(X).values
    #             curKddMod_noYieldGrowth = curKdd[nnProd] - kddFromYieldGrowth
    #             kddMod_noYieldGrowth[i, 0:len(curKddMod_noYieldGrowth)] = curKddMod_noYieldGrowth

    #             gddFromYieldGrowth = mdl_SH_GDD.predict(X).values
    #             curGddMod_noYieldGrowth = curGdd[nnProd] - gddFromYieldGrowth
    #             gddMod_noYieldGrowth[i, 0:len(curGddMod_noYieldGrowth)] = curGddMod_noYieldGrowth

            if wxData == 'era5':
                tFromYieldGrowth = np.full([len(X[shVar]), n_samples], np.nan)
                curTMod_yieldGrowth = np.matlib.repmat(curT[nnProd], n_samples, 1).T
                tMod_yieldGrowth[i, 0:len(curTMod_yieldGrowth), :, b] = curTMod_yieldGrowth

                mdl_SH_T_SH_sample = mdl_SH_T.params[shVar] + \
                                        st.t.ppf(sample_x, mdl_SH_T.df_model) * mdl_SH_T.bse[shVar]

                for k in range(n_samples):
                    X = {shVar:shFromYieldGrowth[:, sh_coef_sample[k]]}
                    tFromYieldGrowth[:, k] = mdl_SH_T.params['Intercept'] + \
                                        mdl_SH_T_SH_sample[k]*np.array(X[shVar])

                curTMod_noYieldGrowth = np.matlib.repmat(curT[nnProd], n_samples, 1).T - tFromYieldGrowth


            # MAIZE ---------------------------------------------------
            if len(nnMaize) >= minCropYears:

                mdl_KDD_GDD_PR_MaizeYield_KDD_sample = mdl_KDD_GDD_PR_MaizeYield.params['KDD_Detrend'] + \
                                        st.t.ppf(sample_x, mdl_KDD_GDD_PR_MaizeYield.df_model) * mdl_KDD_GDD_PR_MaizeYield.bse['KDD_Detrend']
                mdl_KDD_GDD_PR_MaizeYield_GDD_sample = mdl_KDD_GDD_PR_MaizeYield.params['GDD_Detrend'] + \
                                        st.t.ppf(sample_x, mdl_KDD_GDD_PR_MaizeYield.df_model) * mdl_KDD_GDD_PR_MaizeYield.bse['GDD_Detrend']
                mdl_KDD_GDD_PR_MaizeYield_Pr_sample = mdl_KDD_GDD_PR_MaizeYield.params['Pr_Detrend'] + \
                                        st.t.ppf(sample_x, mdl_KDD_GDD_PR_MaizeYield.df_model) * mdl_KDD_GDD_PR_MaizeYield.bse['Pr_Detrend']

                gdd_coef_sample = np.random.choice(n_samples, n_samples)
                kdd_coef_sample = np.random.choice(n_samples, n_samples)
                pr_coef_sample = np.random.choice(n_samples, n_samples)

                gdd_data_sample = np.random.choice(n_samples, n_samples)
                kdd_data_sample = np.random.choice(n_samples, n_samples)

                curMaizeYieldMod_yieldGrowth = np.full([len(X[shVar]), n_samples], np.nan)
                curMaizeYieldMod_noYieldGrowth = np.full([len(X[shVar]), n_samples], np.nan)

                for k in range(n_samples):
                    # and now calculate yield using gdd/kdd/pr
                    X = {'GDD_Detrend':curGddMod_yieldGrowth[:, gdd_data_sample[k]],
                         'KDD_Detrend':curKddMod_yieldGrowth[:, kdd_data_sample[k]], 
                         'Pr_Detrend':dataProd['Pr']**2}
                    curMaizeYieldMod_yieldGrowth[:, k] = mdl_KDD_GDD_PR_MaizeYield.params['Intercept'] + \
                                                    mdl_KDD_GDD_PR_MaizeYield_KDD_sample[kdd_coef_sample[k]]*np.array(X['KDD_Detrend']) + \
                                                    mdl_KDD_GDD_PR_MaizeYield_GDD_sample[gdd_coef_sample[k]]*np.array(X['GDD_Detrend']) + \
                                                    mdl_KDD_GDD_PR_MaizeYield_Pr_sample[pr_coef_sample[k]]*np.array(X['Pr_Detrend']) + \
                                                    np.nanmean(curMaizeYield)

                    X = {'GDD_Detrend':curGddMod_noYieldGrowth[:, gdd_data_sample[k]],
                         'KDD_Detrend':curKddMod_noYieldGrowth[:, kdd_data_sample[k]], 
                         'Pr_Detrend':dataProd['Pr']**2}
                    curMaizeYieldMod_noYieldGrowth[:, k] = mdl_KDD_GDD_PR_MaizeYield.params['Intercept'] + \
                                                    mdl_KDD_GDD_PR_MaizeYield_KDD_sample[kdd_coef_sample[k]]*np.array(X['KDD_Detrend']) + \
                                                    mdl_KDD_GDD_PR_MaizeYield_GDD_sample[gdd_coef_sample[k]]*np.array(X['GDD_Detrend']) + \
                                                    mdl_KDD_GDD_PR_MaizeYield_Pr_sample[pr_coef_sample[k]]*np.array(X['Pr_Detrend']) + \
                                                    np.nanmean(curMaizeYield)

                maizeYieldMod_yieldGrowth[i, 0:len(curMaizeYieldMod_yieldGrowth), :, b] = curMaizeYieldMod_yieldGrowth
                maizeYieldMod_noYieldGrowth[i, 0:len(curMaizeYieldMod_noYieldGrowth), :, b] = curMaizeYieldMod_noYieldGrowth

    #                 curMaizeYieldMod_yieldGrowth = mdl_KDD_GDD_PR_MaizeYield.predict(X).values+np.nanmean(curMaizeYield)
    #                 maizeYieldMod_yieldGrowth[i, 0:len(curMaizeYieldMod_yieldGrowth)] = curMaizeYieldMod_yieldGrowth

    #             X = {'GDD_Detrend':curGddMod_noYieldGrowth, 
    #                  'KDD_Detrend':curKddMod_noYieldGrowth, 
    #                  'Pr_Detrend':dataProd['Pr']}
    #             curMaizeYieldMod_noYieldGrowth = mdl_KDD_GDD_PR_MaizeYield.predict(X).values+np.nanmean(curMaizeYield)
    #             maizeYieldMod_noYieldGrowth[i, 0:len(curMaizeYieldMod_noYieldGrowth)] = curMaizeYieldMod_noYieldGrowth

            # SOYBEAN ---------------------------------------------------
            if len(nnSoybean) >= minCropYears:

                mdl_KDD_GDD_PR_SoybeanYield_KDD_sample = mdl_KDD_GDD_PR_SoybeanYield.params['KDD_Detrend'] + \
                                        st.t.ppf(sample_x, mdl_KDD_GDD_PR_SoybeanYield.df_model) * mdl_KDD_GDD_PR_SoybeanYield.bse['KDD_Detrend']
                mdl_KDD_GDD_PR_SoybeanYield_GDD_sample = mdl_KDD_GDD_PR_SoybeanYield.params['GDD_Detrend'] + \
                                        st.t.ppf(sample_x, mdl_KDD_GDD_PR_SoybeanYield.df_model) * mdl_KDD_GDD_PR_SoybeanYield.bse['GDD_Detrend']
                mdl_KDD_GDD_PR_SoybeanYield_Pr_sample = mdl_KDD_GDD_PR_SoybeanYield.params['Pr_Detrend'] + \
                                        st.t.ppf(sample_x, mdl_KDD_GDD_PR_SoybeanYield.df_model) * mdl_KDD_GDD_PR_SoybeanYield.bse['Pr_Detrend']

                gdd_coef_sample = np.random.choice(n_samples, n_samples)
                kdd_coef_sample = np.random.choice(n_samples, n_samples)
                pr_coef_sample = np.random.choice(n_samples, n_samples)

                gdd_data_sample = np.random.choice(n_samples, n_samples)
                kdd_data_sample = np.random.choice(n_samples, n_samples)

                curSoybeanYieldMod_yieldGrowth = np.full([len(X['GDD_Detrend']), n_samples], np.nan)
                curSoybeanYieldMod_noYieldGrowth = np.full([len(X['GDD_Detrend']), n_samples], np.nan)

                for k in range(n_samples):
                    # and now calculate yield using gdd/kdd/pr
                    X = {'GDD_Detrend':curGddMod_yieldGrowth[:, gdd_data_sample[k]],
                         'KDD_Detrend':curKddMod_yieldGrowth[:, kdd_data_sample[k]], 
                         'Pr_Detrend':dataProd['Pr']**2}
                    curSoybeanYieldMod_yieldGrowth[:, k] = mdl_KDD_GDD_PR_SoybeanYield.params['Intercept'] + \
                                                    mdl_KDD_GDD_PR_SoybeanYield_KDD_sample[kdd_coef_sample[k]]*np.array(X['KDD_Detrend']) + \
                                                    mdl_KDD_GDD_PR_SoybeanYield_GDD_sample[gdd_coef_sample[k]]*np.array(X['GDD_Detrend']) + \
                                                    mdl_KDD_GDD_PR_SoybeanYield_Pr_sample[pr_coef_sample[k]]*np.array(X['Pr_Detrend']) + \
                                                    np.nanmean(curSoybeanYield)

                    X = {'GDD_Detrend':curGddMod_noYieldGrowth[:, gdd_data_sample[k]],
                         'KDD_Detrend':curKddMod_noYieldGrowth[:, kdd_data_sample[k]], 
                         'Pr_Detrend':dataProd['Pr']**2}
                    curSoybeanYieldMod_noYieldGrowth[:, k] = mdl_KDD_GDD_PR_SoybeanYield.params['Intercept'] + \
                                                    mdl_KDD_GDD_PR_SoybeanYield_KDD_sample[kdd_coef_sample[k]]*np.array(X['KDD_Detrend']) + \
                                                    mdl_KDD_GDD_PR_SoybeanYield_GDD_sample[gdd_coef_sample[k]]*np.array(X['GDD_Detrend']) + \
                                                    mdl_KDD_GDD_PR_SoybeanYield_Pr_sample[pr_coef_sample[k]]*np.array(X['Pr_Detrend']) + \
                                                    np.nanmean(curSoybeanYield)

                soybeanYieldMod_yieldGrowth[i, 0:len(curSoybeanYieldMod_yieldGrowth), :, b] = curSoybeanYieldMod_yieldGrowth
                soybeanYieldMod_noYieldGrowth[i, 0:len(curSoybeanYieldMod_noYieldGrowth), :, b] = curSoybeanYieldMod_noYieldGrowth

    #             # and now calculate yield using gdd/kdd/pr
    #             X = {'GDD_Detrend':curGddMod_yieldGrowth,
    #                  'KDD_Detrend':curKddMod_yieldGrowth, 
    #                  'Pr_Detrend':dataProd['Pr']}
    #             curSoybeanYieldMod_yieldGrowth = mdl_KDD_GDD_PR_SoybeanYield.predict(X).values+np.nanmean(curSoybeanYield)
    #             soybeanYieldMod_yieldGrowth[i, 0:len(curSoybeanYieldMod_yieldGrowth)] = curSoybeanYieldMod_yieldGrowth

    #             X = {'GDD_Detrend':curGddMod_noYieldGrowth, 
    #                  'KDD_Detrend':curKddMod_noYieldGrowth, 
    #                  'Pr_Detrend':dataProd['Pr']}
    #             curSoybeanYieldMod_noYieldGrowth = mdl_KDD_GDD_PR_SoybeanYield.predict(X).values+np.nanmean(curSoybeanYield)
    #             soybeanYieldMod_noYieldGrowth[i, 0:len(curSoybeanYieldMod_noYieldGrowth)] = curSoybeanYieldMod_noYieldGrowth

            # now take linear trends over predicted lh, sh, gdd, kdd, yield values to calculate 1981-2018 trends
            # this smooths out variability and ensures that values exist for all years (for counties with short yield time series)
            X = sm.add_constant(range(len((curNetRad))))
            mdl = sm.OLS((curNetRad), X).fit()
            netRadObsTrend[i] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

            # SH
            X = sm.add_constant(range(len(curSshf)))
            mdl = sm.OLS(curSshf, X).fit()
            curSshfTrend = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

            for k in range(n_samples):
                X = sm.add_constant(range(len(curShMod_yieldGrowth[:, k])))
                mdl = sm.OLS(curShMod_yieldGrowth[:, k], X).fit()
                shModTrend_yieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                X = sm.add_constant(range(len(curShMod_noYieldGrowth[:, k])))
                mdl = sm.OLS(curShMod_noYieldGrowth[:, k], X).fit()
                shModTrend_noYieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                shTrendFrac[i, k, b] = (shModTrend_yieldGrowth[i, k, b]-shModTrend_noYieldGrowth[i, k, b]) / curSshfTrend
                shFromFeedback[i, k, b] = shModTrend_yieldGrowth[i, k, b]-shModTrend_noYieldGrowth[i, k, b]

                shChgFeedbackWithAgInt[i, k, b] = shModTrend_yieldGrowth[i, k, b]
                shChgFeedbackWithoutAgInt[i, k, b] = shModTrend_noYieldGrowth[i, k, b]
            shObsTrend[i] = curSshfTrend


            # LH
            X = sm.add_constant(range(len(curSlhf)))
            mdl = sm.OLS(curSlhf, X).fit()
            curSlhfTrend = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

            for k in range(n_samples):
                X = sm.add_constant(range(len(curLhMod_yieldGrowth[:, k])))
                mdl = sm.OLS(curLhMod_yieldGrowth[:, k], X).fit()
                lhModTrend_yieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                X = sm.add_constant(range(len(curLhMod_noYieldGrowth[:, k])))
                mdl = sm.OLS(curLhMod_noYieldGrowth[:, k], X).fit()
                lhModTrend_noYieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                lhTrendFrac[i, k, b] = (lhModTrend_yieldGrowth[i, k, b]-lhModTrend_noYieldGrowth[i, k, b]) / curSlhfTrend
                lhFromFeedback[i, k, b] = lhModTrend_yieldGrowth[i, k, b]-lhModTrend_noYieldGrowth[i, k, b]

                lhChgFeedbackWithAgInt[i, k, b] = lhModTrend_yieldGrowth[i, k, b]
                lhChgFeedbackWithoutAgInt[i, k, b] = lhModTrend_noYieldGrowth[i, k, b]
            lhObsTrend[i] = curSlhfTrend

            
            
            # SEASONAL MEAN T
            if wxData == 'era5':
                X = sm.add_constant(range(len(curT)))
                mdl = sm.OLS(curT, X).fit()
                curTTrend = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                for k in range(n_samples):
                    X = sm.add_constant(range(len(curTMod_yieldGrowth[:, k])))
                    mdl = sm.OLS(curTMod_yieldGrowth[:, k], X).fit()
                    tModTrend_yieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                    X = sm.add_constant(range(len(curTMod_noYieldGrowth[:, k])))
                    mdl = sm.OLS(curTMod_noYieldGrowth[:, k], X).fit()
                    tModTrend_noYieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                    tFromFeedback[i, k, b] = tModTrend_yieldGrowth[i, k, b]-tModTrend_noYieldGrowth[i, k, b]



            # KDD
            X = sm.add_constant(range(len(curKdd)))
            mdl = sm.OLS(curKdd, X).fit()
            curKddTrend = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

            for k in range(n_samples):
                X = sm.add_constant(range(len(curKddMod_yieldGrowth[:, k])))
                mdl = sm.OLS(curKddMod_yieldGrowth[:, k], X).fit()
                kddModTrend_yieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                X = sm.add_constant(range(len(curKddMod_noYieldGrowth[:, k])))
                mdl = sm.OLS(curKddMod_noYieldGrowth[:, k], X).fit()
                kddModTrend_noYieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                kddTrendFrac[i, k, b] = (kddModTrend_yieldGrowth[i, k, b]-kddModTrend_noYieldGrowth[i, k, b]) / curKddTrend
                kddFromFeedback[i, k, b] = kddModTrend_yieldGrowth[i, k, b]-kddModTrend_noYieldGrowth[i, k, b]

                kddChgFeedbackWithAgInt[i, k, b] = kddModTrend_yieldGrowth[i, k, b]
                kddChgFeedbackWithoutAgInt[i, k, b] = kddModTrend_noYieldGrowth[i, k, b]


            # GDD
            X = sm.add_constant(range(len(curGdd)))
            mdl = sm.OLS(curGdd, X).fit()
            curGddTrend = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

            for k in range(n_samples):
                X = sm.add_constant(range(len(curGddMod_yieldGrowth[:, k])))
                mdl = sm.OLS(curGddMod_yieldGrowth[:, k], X).fit()
                gddModTrend_yieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                X = sm.add_constant(range(len(curGddMod_noYieldGrowth[:, k])))
                mdl = sm.OLS(curGddMod_noYieldGrowth[:, k], X).fit()
                gddModTrend_noYieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                gddTrendFrac[i, k, b] = (gddModTrend_yieldGrowth[i, k, b]-gddModTrend_noYieldGrowth[i, k, b]) / curGddTrend
                gddFromFeedback[i, k, b] = gddModTrend_yieldGrowth[i, k, b]-gddModTrend_noYieldGrowth[i, k, b]

                gddChgFeedbackWithAgInt[i, k, b] = gddModTrend_yieldGrowth[i, k, b]
                gddChgFeedbackWithoutAgInt[i, k, b] = gddModTrend_noYieldGrowth[i, k, b]



            # MAIZE Yield ----------------------------------------
            if len(nnMaize) >= 10:
                X = sm.add_constant(range(len(curMaizeYield[nnMaize])))
                mdl = sm.OLS(curMaizeYield[nnMaize], X).fit()
                curMaizeYieldTrend = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                for k in range(n_samples):
                    X = sm.add_constant(range(len(curMaizeYieldMod_yieldGrowth[:, k])))
                    mdl = sm.OLS(curMaizeYieldMod_yieldGrowth[:, k], X).fit()
                    maizeYieldModTrend_yieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                    X = sm.add_constant(range(len(curMaizeYieldMod_noYieldGrowth[:, k])))
                    mdl = sm.OLS(curMaizeYieldMod_noYieldGrowth[:, k], X).fit()
                    maizeYieldModTrend_noYieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                    maizeYieldTrendFrac[i, k, b] = (maizeYieldModTrend_yieldGrowth[i, k, b]-maizeYieldModTrend_noYieldGrowth[i, k, b]) / curMaizeYieldTrend
                    maizeYieldFromFeedback[i, k, b] = maizeYieldModTrend_yieldGrowth[i, k, b]-maizeYieldModTrend_noYieldGrowth[i, k, b]

                    maizeYieldChgFeedbackWithAgInt[i, k, b] = maizeYieldModTrend_yieldGrowth[i, k, b]
                    maizeYieldChgFeedbackWithoutAgInt[i, k, b] = maizeYieldModTrend_noYieldGrowth[i, k, b]


            # SOYBEAN Yield ----------------------------------------
            if len(nnSoybean) >= 10:
                X = sm.add_constant(range(len(curSoybeanYield[nnSoybean])))
                mdl = sm.OLS(curSoybeanYield[nnSoybean], X).fit()
                curSoybeanYieldTrend = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                for k in range(n_samples):
                    X = sm.add_constant(range(len(curSoybeanYieldMod_yieldGrowth[:, k])))
                    mdl = sm.OLS(curSoybeanYieldMod_yieldGrowth[:, k], X).fit()
                    soybeanYieldModTrend_yieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                    X = sm.add_constant(range(len(curSoybeanYieldMod_noYieldGrowth[:, k])))
                    mdl = sm.OLS(curSoybeanYieldMod_noYieldGrowth[:, k], X).fit()
                    soybeanYieldModTrend_noYieldGrowth[i, k, b] = mdl.params[1]*10#mdl.params[0]+np.arange(0, NYears)*mdl.params[1]

                    soybeanYieldTrendFrac[i, k, b] = (soybeanYieldModTrend_yieldGrowth[i, k, b]-soybeanYieldModTrend_noYieldGrowth[i, k, b]) / curSoybeanYieldTrend
                    soybeanYieldFromFeedback[i, k, b] = soybeanYieldModTrend_yieldGrowth[i, k, b]-soybeanYieldModTrend_noYieldGrowth[i, k, b]

                    soybeanYieldChgFeedbackWithAgInt[i, k, b] = soybeanYieldModTrend_yieldGrowth[i, k, b]
                    soybeanYieldChgFeedbackWithoutAgInt[i, k, b] = soybeanYieldModTrend_noYieldGrowth[i, k, b]

        
        fipsSel.append(countyFips[i])

    maizeYieldFromFeedback_SensTest[:,:,:,a] = maizeYieldFromFeedback
    soybeanYieldFromFeedback_SensTest[:,:,:,a] = soybeanYieldFromFeedback
    
    fipsSel = np.array(fipsSel)
    fipsAll = np.array(fipsAll)
    haExclude = np.array(haExclude)
    irSel = np.array(irSel)
    shortSeriesExclude = np.array(shortSeriesExclude)
    nonSigExclude = np.array(nonSigExclude)
    lhTrendFrac *= 100
    shTrendFrac *= 100
    kddTrendFrac *= 100
    gddTrendFrac *= 100
    maizeYieldTrendFrac *= 100
    soybeanYieldTrendFrac *= 100


ccFeedbackAnalysis = {

    'fipsAll':fipsAll,
    'fipsSel':fipsSel,
    'haExclude':haExclude,
    'irSel':irSel,
    'shortSeriesExclude':shortSeriesExclude,
    'nonSigExclude':nonSigExclude,
    
    'lhTrendFrac':lhTrendFrac,
    'lhFromFeedback':lhFromFeedback,
    'lhChgFeedbackWithAgInt':lhChgFeedbackWithAgInt,
    'lhChgFeedbackWithoutAgInt':lhChgFeedbackWithoutAgInt,
    
    'shTrendFrac':shTrendFrac,
    'shFromFeedback':shFromFeedback,
    'shChgFeedbackWithAgInt':shChgFeedbackWithAgInt,
    'shChgFeedbackWithoutAgInt':shChgFeedbackWithoutAgInt,
    
    'gddTrendFrac':gddTrendFrac,
    'gddFromFeedback':gddFromFeedback,
    'gddChgFeedbackWithAgInt':gddChgFeedbackWithAgInt,
    'gddChgFeedbackWithoutAgInt':gddChgFeedbackWithoutAgInt,
    
    'kddTrendFrac':kddTrendFrac,
    'kddFromFeedback':kddFromFeedback,
    'kddChgFeedbackWithAgInt':kddChgFeedbackWithAgInt,
    'kddChgFeedbackWithoutAgInt':kddChgFeedbackWithoutAgInt,
    
    'maizeYieldTrendFrac':maizeYieldTrendFrac,
    'maizeYieldFromFeedback':maizeYieldFromFeedback,
    'maizeYieldChgFeedbackWithAgInt':maizeYieldChgFeedbackWithAgInt,
    'maizeYieldChgFeedbackWithoutAgInt':maizeYieldChgFeedbackWithoutAgInt,
    
    'soybeanYieldTrendFrac':soybeanYieldTrendFrac,
    'soybeanYieldFromFeedback':soybeanYieldFromFeedback,
    'soybeanYieldChgFeedbackWithAgInt':soybeanYieldChgFeedbackWithAgInt,
    'soybeanYieldChgFeedbackWithoutAgInt':soybeanYieldChgFeedbackWithoutAgInt,
    
    'tFromFeedback':tFromFeedback,
    
    'mdl_LH_Y_Coefs':mdl_LH_Y_Coefs,
    'mdl_LH_Y_PValues':mdl_LH_Y_PValues,
    
    'mdl_LH_Y_Norm_Coefs':mdl_LH_Y_Norm_Coefs,
    'mdl_LH_Y_Norm_PValues':mdl_LH_Y_Norm_PValues,
    
    'mdl_LH_SH_Coefs':mdl_LH_SH_Coefs,
    'mdl_LH_SH_PValues':mdl_LH_SH_PValues,
    
    'mdl_LH_SH_Norm_Coefs':mdl_LH_SH_Norm_Coefs,
    'mdl_LH_SH_Norm_PValues':mdl_LH_SH_Norm_PValues,
    
    'mdl_SH_KDD_Coefs':mdl_SH_KDD_Coefs,
    'mdl_SH_KDD_PValues':mdl_SH_KDD_PValues,
    
    'mdl_SH_KDD_Norm_Coefs':mdl_SH_KDD_Norm_Coefs,
    'mdl_SH_KDD_Norm_PValues':mdl_SH_KDD_Norm_PValues,
    
    'mdl_SH_GDD_Coefs':mdl_SH_GDD_Coefs,
    'mdl_SH_GDD_PValues':mdl_SH_GDD_PValues,
    
    'mdl_SH_GDD_Norm_Coefs':mdl_SH_GDD_Norm_Coefs,
    'mdl_SH_GDD_Norm_PValues':mdl_SH_GDD_Norm_PValues,
    
    'mdl_KDD_GDD_MaizeYield_Coefs':mdl_KDD_GDD_MaizeYield_Coefs,
    'mdl_KDD_GDD_MaizeYield_PValues':mdl_KDD_GDD_MaizeYield_PValues,
    
    'mdl_KDD_GDD_MaizeYield_Norm_Coefs':mdl_KDD_GDD_MaizeYield_Norm_Coefs,
    'mdl_KDD_GDD_MaizeYield_Norm_PValues':mdl_KDD_GDD_MaizeYield_Norm_PValues,
    
    'mdl_KDD_GDD_SoybeanYield_Coefs':mdl_KDD_GDD_SoybeanYield_Coefs,
    'mdl_KDD_GDD_SoybeanYield_PValues':mdl_KDD_GDD_SoybeanYield_PValues,
    
    'mdl_KDD_GDD_SoybeanYield_Norm_Coefs':mdl_KDD_GDD_SoybeanYield_Norm_Coefs,
    'mdl_KDD_GDD_SoybeanYield_Norm_PValues':mdl_KDD_GDD_SoybeanYield_Norm_PValues,
    
    'mdl_LH_Y_CondNum':mdl_LH_Y_CondNum,
    'mdl_LH_SH_CondNum':mdl_LH_SH_CondNum,
    'mdl_SH_KDD_CondNum':mdl_SH_KDD_CondNum,
    'mdl_SH_GDD_CondNum':mdl_SH_GDD_CondNum,
    'mdl_KDD_GDD_PR_MaizeYield_CondNum':mdl_KDD_GDD_PR_MaizeYield_CondNum,
    'mdl_KDD_GDD_PR_SoybeanYield_CondNum':mdl_KDD_GDD_PR_SoybeanYield_CondNum,
    
}

if includeKdd:
    with gzip.open('cc-feedback-analysis-%d-%d-kdd.dat'%(n_bootstraps, n_samples), 'wb') as f:
        pickle.dump(ccFeedbackAnalysis, f)
else:
    with gzip.open('cc-feedback-analysis-%d-%d-new.dat'%(n_bootstraps, n_samples), 'wb') as f:
        pickle.dump(ccFeedbackAnalysis, f)