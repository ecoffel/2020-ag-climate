# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:01:10 2019

@author: Ethan
"""                                                                                                                                

import rasterio as rio
import matplotlib.pyplot as plt 
import numpy as np
from scipy import interpolate
import statsmodels.api as sm
import scipy.stats as st
import sys, pickle
import geopy.distance

dataDir = 'E:/data/ecoffel/data/projects/ag-land-climate/CroplandPastureArea2000_Geotiff/CroplandPastureArea2000_Geotiff'

tDataset = 'era'
tDatasetAnom = ''#'-anom-2'
tVar = 'tmax'

pDataset = 'era';
pVar = 'pmax'

def roundTo2(x):
    return 2*round(x/2)

pasture = rio.open('%s/Pasture2000_5m.tif'%dataDir)
pastureData = pasture.read(1)

crop = rio.open('%s/Cropland2000_5m.tif'%dataDir)
cropData = crop.read(1)

latOld = np.linspace(pasture.bounds.bottom, pasture.bounds.top, pasture.shape[0])
lonOld = np.linspace(pasture.bounds.left, pasture.bounds.right, pasture.shape[1])

pastureInterp = interpolate.interp2d(lonOld, latOld, pastureData, kind='linear')
cropInterp = interpolate.interp2d(lonOld, latOld, cropData, kind='linear')

latNew = np.linspace(90, -90, 90)
lonNew = np.linspace(-180, 180, 180)

pastureRegrid = pastureInterp(lonNew, latNew)
pastureRegrid[pastureRegrid < 0] = np.nan

cropRegrid = cropInterp(lonNew, latNew)
cropRegrid[cropRegrid < 0] = np.nan

with open('elevation-map.dat', 'rb') as f:
    elevationMap = pickle.load(f)

#koppen = np.genfromtxt('koppen-classifications.txt', delimiter=',')
#koppen = np.flipud(koppen)
#koppen = np.roll(koppen, 90)
#koppen[koppen == 0] = np.nan

koppen = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/koppen-classification/koppen_1901-2010.tsv', dtype=None, names=True, encoding='UTF-8')

koppenGroupsPCells = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}
koppenGroupsNoPCells = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}

koppenGroupsCCells = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}
koppenGroupsNoCCells = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[]}

koppenGroupsFullPCells = {}
koppenGroupsFullNoPCells = {}

koppenGroupsFullCCells = {}
koppenGroupsFullNoCCells = {}

koppenLat = np.linspace(90, -90, 90)
koppenLon = np.linspace(-180, 180, 180)

koppenMap = np.zeros([len(koppenLat), len(koppenLon)])

for x in range(len(koppenLat)):
    
    for y in range(len(koppenLon)):
        
        minDist = -1
        minDistInd = -1
        
        latInd = np.where((abs(koppen['latitude']-koppenLat[x]) <= 1))[0]
        
        for i in latInd:
            if abs(koppen['longitude'][i]-koppenLon[y]) > 1:
                continue
            
            ptKoppen = (koppen['latitude'][i], koppen['longitude'][i])
            ptAg = (koppenLat[x], koppenLon[y])
            
            dist = geopy.distance.great_circle(ptKoppen, ptAg).km
            if minDist == -1 or dist < minDist:
                minDist = dist
                minDistInd = i
        
        if minDistInd != -1:
            classification = koppen['p1901_2010'][minDistInd]
            classificationInt = 0
            if classification in ['As', 'Aw', 'Am', 'Af']: 
                classificationInt = 1
            elif classification in ['BSk', 'BWk', 'BSh']: 
                classificationInt = 2
            elif classification in ['Cfb', 'Csb', 'Cfa', 'Csa', 'Cwa']:
                classificationInt = 3
            elif classification in ['Dfb', 'Dfa', 'Dwc', 'Dwb']:
                classificationInt = 4
            koppenMap[x,y] = classificationInt
        else:
            koppenMap[x,y] = np.nan
        
        if np.isnan(pastureRegrid[x,y]):
            continue
        
        if pastureRegrid[x,y] > 0.05:
            if classification in koppenGroupsFullPCells.keys():
                koppenGroupsFullPCells[classification].append((x,y))
            else:
                koppenGroupsFullPCells[classification] = [(x,y)]
        else:
            if classification in koppenGroupsFullNoPCells.keys():
                koppenGroupsFullNoPCells[classification].append((x,y))
            else:
                koppenGroupsFullNoPCells[classification] = [(x,y)]
        
        if cropRegrid[x,y] > 0.05:
            if classification in koppenGroupsFullCCells.keys():
                koppenGroupsFullCCells[classification].append((x,y))
            else:
                koppenGroupsFullCCells[classification] = [(x,y)]
        else:
            if classification in koppenGroupsFullNoCCells.keys():
                koppenGroupsFullNoCCells[classification].append((x,y))
            else:
                koppenGroupsFullNoCCells[classification] = [(x,y)]
        
        
        if elevationMap[x,y] > 500:
            continue
        
        if not np.isnan(koppenMap[x,y]) and koppenMap[x,y] != 0:
            if classification[0] in koppenGroupsPCells.keys():
                if pastureRegrid[x,y] > 0.25:
                    koppenGroupsPCells[classification[0]].append((x,y))
                else:
                    koppenGroupsNoPCells[classification[0]].append((x,y))
                
                if cropRegrid[x,y] > 0.25:
                    koppenGroupsCCells[classification[0]].append((x,y))
                else:
                    koppenGroupsNoCCells[classification[0]].append((x,y))


totalC = 0
for k in koppenGroupsFullCCells.keys():
    totalC += len(koppenGroupsFullCCells[k])
        
for k in koppenGroupsCCells.keys():
    if k in koppenGroupsNoCCells.keys():
        filteredC = len(koppenGroupsCCells[k])+len(koppenGroupsNoCCells[k])
        if filteredC > 0:
            curPerc = len(koppenGroupsCCells[k])/filteredC
        else:
            curPerc = np.nan
        print('%s: %.2f, %.2f'%(k, len(koppenGroupsCCells[k])/totalC, curPerc))



# load sacks calendars
sacksCal = {}
sacksCal['start_maize'] = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/sacks/sacks-planting-end-Maize.txt', delimiter=',')
sacksCal['start_rice'] = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/sacks/sacks-planting-end-Rice.txt', delimiter=',')
sacksCal['start_soybeans'] = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/sacks/sacks-planting-end-Soybeans.txt', delimiter=',')
sacksCal['start_wheat'] = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/sacks/sacks-planting-end-Wheat.txt', delimiter=',')

sacksCal['end_maize'] = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/sacks/sacks-harvest-start-Maize.txt', delimiter=',')
sacksCal['end_rice'] = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/sacks/sacks-harvest-start-Rice.txt', delimiter=',')
sacksCal['end_soybeans'] = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/sacks/sacks-harvest-start-Soybeans.txt', delimiter=',')
sacksCal['end_wheat'] = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/sacks/sacks-harvest-start-Wheat.txt', delimiter=',')


# indexed by their lon
tData = {}
pData = {}

dayNumbers = []
tLats = []

for k in koppenGroupsPCells.keys():
    print('loading t/p time series for k = %s...'%k)
    for c in range(len(koppenGroupsPCells[k])):
        curLat = int(roundTo2(latNew[koppenGroupsPCells[k][c][0]]))
        curLon = int(roundTo2(lonNew[koppenGroupsPCells[k][c][1]]))
        
        if curLon not in tData.keys():
            ttmp = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/t-p-dist/%s-%s-%d%s.txt'%(tVar, tDataset, curLon, tDatasetAnom), delimiter=',')
            ptmp = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/t-p-dist/%s-%s-%d.txt'%(pVar, pDataset, curLon), delimiter=',')    
            tData[curLon] = ttmp[1:,1:].copy()
            pData[curLon] = ptmp[1:,1:].copy()
            
            # store the day numbers on the first loop
            if len(dayNumbers) == 0:
                dayNumbers = ttmp[1:,0].copy()
                tLats = ttmp[0,1:].copy()
    
    for c in range(len(koppenGroupsNoPCells[k])):
        curLat = int(roundTo2(latNew[koppenGroupsNoPCells[k][c][0]]))
        curLon = int(roundTo2(lonNew[koppenGroupsNoPCells[k][c][1]]))
        
        if curLon not in tData.keys():
            ttmp = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/t-p-dist/%s-%s-%d%s.txt'%(tVar, tDataset, curLon, tDatasetAnom), delimiter=',')
            ptmp = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/t-p-dist/%s-%s-%d.txt'%(pVar, pDataset, curLon), delimiter=',')    
            tData[curLon] = ttmp[1:,1:].copy()
            pData[curLon] = ptmp[1:,1:].copy()
    
    for c in range(len(koppenGroupsCCells[k])):
        curLat = int(roundTo2(latNew[koppenGroupsCCells[k][c][0]]))
        curLon = int(roundTo2(lonNew[koppenGroupsCCells[k][c][1]]))
        
        if curLon not in tData.keys():
            ttmp = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/t-p-dist/%s-%s-%d%s.txt'%(tVar, tDataset, curLon, tDatasetAnom), delimiter=',')
            ptmp = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/t-p-dist/%s-%s-%d.txt'%(pVar, pDataset, curLon), delimiter=',')    
            tData[curLon] = ttmp[1:,1:].copy()
            pData[curLon] = ptmp[1:,1:].copy()
    
    for c in range(len(koppenGroupsNoCCells[k])):
        curLat = int(roundTo2(latNew[koppenGroupsNoCCells[k][c][0]]))
        curLon = int(roundTo2(lonNew[koppenGroupsNoCCells[k][c][1]]))
        
        if curLon not in tData.keys():
            ttmp = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/t-p-dist/%s-%s-%d%s.txt'%(tVar, tDataset, curLon, tDatasetAnom), delimiter=',')
            ptmp = np.genfromtxt('E:/data/ecoffel/data/projects/ag-land-climate/t-p-dist/%s-%s-%d.txt'%(pVar, pDataset, curLon), delimiter=',')    
            tData[curLon] = ttmp[1:,1:].copy()
            pData[curLon] = ptmp[1:,1:].copy()

tMeans = {}
pMeans = {}

for k in koppenGroupsPCells.keys():
    
    print('processing t & p for k = %s...'%k)
    
    tMeans[k] = {}
    tMeans[k]['pLat'] = []
    tMeans[k]['noPLat'] = []
    tMeans[k]['pLon'] = []
    tMeans[k]['noPLon'] = []
    tMeans[k]['pCover'] = []
    tMeans[k]['P'] = []
    tMeans[k]['noP'] = []
    tMeans[k]['cLat'] = []
    tMeans[k]['noCLat'] = []
    tMeans[k]['cLon'] = []
    tMeans[k]['noCLon'] = []
    tMeans[k]['cCover'] = []
    tMeans[k]['C'] = []
    tMeans[k]['noC'] = []
    
    pMeans[k] = {}
    pMeans[k]['pCover'] = []
    pMeans[k]['P'] = []
    pMeans[k]['noP'] = []
    pMeans[k]['cCover'] = []
    pMeans[k]['C'] = []
    pMeans[k]['noC'] = []
    
    # the range of lat values with crops in current koppen zone
    pLatRange = []
    cLatRange = []
    
    pTempRange = []
    cTempRange = []
    
    pCells = koppenGroupsPCells[k]
    for p in range(len(pCells)):
        curLon = int(2*round(lonNew[pCells[p][1]]/2))
        curLatCoord = pCells[p][0]
        curLonCoord = pCells[p][1]
        
        tLatInd = np.where((abs(tLats - latNew[curLatCoord]) == np.nanmin(abs(tLats - latNew[curLatCoord]))))[0]
        
        curTSeries = tData[curLon][:,tLatInd]
        nn = np.where(~np.isnan(curTSeries))[0]
        if len(nn) > 5:
            X = sm.add_constant(range(len(curTSeries[nn])))
            mdlT = sm.OLS(np.array(curTSeries[nn]).reshape(-1,1), X).fit()
        curTMean = np.nanmean(tData[curLon][:,tLatInd])
        
        curPSeries = pData[curLon][:,tLatInd]
        nn = np.where(~np.isnan(curPSeries))[0]
        if len(nn) > 5:
            X = sm.add_constant(range(len(curPSeries[nn])))
            mdlP = sm.OLS(np.array(curPSeries[nn]).reshape(-1,1), X).fit()
        
        pLatRange.append(latNew[curLatCoord])
        pTempRange.append(curTMean)
        
        tMeans[k]['pCover'].append(pastureRegrid[curLatCoord, curLonCoord])
        pMeans[k]['pCover'].append(pastureRegrid[curLatCoord, curLonCoord])        
        tMeans[k]['pLat'].append(latNew[curLatCoord])
        tMeans[k]['pLon'].append(lonNew[curLonCoord])
        tMeans[k]['P'].append(mdlT.params[1])
        pMeans[k]['P'].append(mdlP.params[1]*365)
    
    tMeans[k]['pCover'] = np.array(tMeans[k]['pCover'])
    pMeans[k]['pCover'] = np.array(pMeans[k]['pCover'])
    tMeans[k]['pLat'] = np.array(tMeans[k]['pLat'])
    tMeans[k]['pLon'] = np.array(tMeans[k]['pLon'])
    tMeans[k]['P'] = np.array(tMeans[k]['P'])
    pMeans[k]['P'] = np.array(pMeans[k]['P'])    
    
    cCells = koppenGroupsCCells[k]
    for p in range(len(cCells)):
        curLon = int(2*round(lonNew[cCells[p][1]]/2))
        curLatCoord = cCells[p][0]
        curLonCoord = cCells[p][1]
        
        cLatRange.append(latNew[curLatCoord])
        
        tLatInd = np.where((abs(tLats - latNew[curLatCoord]) == np.nanmin(abs(tLats - latNew[curLatCoord]))))[0][0]
        
        curTSeries = tData[curLon][:,tLatInd]
        nn = np.where(~np.isnan(curTSeries))[0]
        if len(nn) > 5:
            X = sm.add_constant(range(len(curTSeries[nn])))
            mdlT = sm.OLS(np.array(curTSeries[nn]).reshape(-1,1), X).fit()
        curTMean = np.nanmean(tData[curLon][:,tLatInd])
        
        curPSeries = pData[curLon][:,tLatInd]
        nn = np.where(~np.isnan(curPSeries))[0]
        if len(nn) > 5:
            X = sm.add_constant(range(len(curPSeries[nn])))
            mdlP = sm.OLS(np.array(curPSeries[nn]).reshape(-1,1), X).fit()
        
        cTempRange.append(curTMean)
        
        tMeans[k]['cCover'].append(cropRegrid[cCells[p][0], cCells[p][1]])
        pMeans[k]['cCover'].append(cropRegrid[cCells[p][0], cCells[p][1]])
        tMeans[k]['cLat'].append(latNew[curLatCoord])
        tMeans[k]['cLon'].append(lonNew[curLonCoord])
        tMeans[k]['C'].append(mdlT.params[1])
        pMeans[k]['C'].append(mdlP.params[1]*365)
        
    tMeans[k]['cCover'] = np.array(tMeans[k]['cCover'])
    pMeans[k]['cCover'] = np.array(pMeans[k]['cCover'])
    tMeans[k]['cLat'] = np.array(tMeans[k]['cLat'])
    tMeans[k]['cLon'] = np.array(tMeans[k]['cLon'])
    tMeans[k]['C'] = np.array(tMeans[k]['C'])
    pMeans[k]['C'] = np.array(pMeans[k]['C'])
    
    noPCells = koppenGroupsNoPCells[k]
    for p in range(len(noPCells)):
        curLon = int(2*round(lonNew[noPCells[p][1]]/2))
        curLatCoord = noPCells[p][0]
        curLonCoord = noPCells[p][1]

        # only look at no-p cells in the same lat range where p cells occur
        if len(pLatRange) > 0:
            if curLatCoord >= np.nanmin(pLatRange) and curLatCoord <= np.nanmax(pLatRange):
                
                tLatInd = np.where((abs(tLats - latNew[curLatCoord]) == np.nanmin(abs(tLats - latNew[curLatCoord]))))[0][0]
                
                curTSeries = tData[curLon][:,tLatInd]
                nn = np.where(~np.isnan(curTSeries))[0]
                if len(nn) > 5:
                    X = sm.add_constant(range(len(curTSeries[nn])))
                    mdlT = sm.OLS(np.array(curTSeries[nn]).reshape(-1,1), X).fit()
                curTMean = np.nanmean(tData[curLon][:,tLatInd])
                
                curPSeries = pData[curLon][:,tLatInd]
                nn = np.where(~np.isnan(curPSeries))[0]
                if len(nn) > 5:
                    X = sm.add_constant(range(len(curPSeries[nn])))
                    mdlP = sm.OLS(np.array(curPSeries[nn]).reshape(-1,1), X).fit()
                                    
                tMeans[k]['noPLat'].append(latNew[curLatCoord])
                tMeans[k]['noPLon'].append(lonNew[curLonCoord])
                tMeans[k]['noP'].append(mdlT.params[1])
                pMeans[k]['noP'].append(mdlP.params[1]*365)
    
    tMeans[k]['noPLat'] = np.array(tMeans[k]['noPLat'])
    tMeans[k]['noPLon'] = np.array(tMeans[k]['noPLon'])
    tMeans[k]['noP'] = np.array(tMeans[k]['noP'])
    pMeans[k]['noP'] = np.array(pMeans[k]['noP'])
    
    noCCells = koppenGroupsNoCCells[k]
    for p in range(len(noCCells)):
        curLon = int(2*round(lonNew[noCCells[p][1]]/2))
        curLatCoord = noCCells[p][0]
        curLonCoord = noCCells[p][1]
        
        if len(cLatRange) > 0:
            if curLatCoord >= np.nanmin(cLatRange) and curLatCoord <= np.nanmax(cLatRange):
                
                tLatInd = np.where((abs(tLats - latNew[curLatCoord]) == np.nanmin(abs(tLats - latNew[curLatCoord]))))[0][0]
                
                curTSeries = tData[curLon][:,tLatInd]
                nn = np.where(~np.isnan(curTSeries))[0]
                if len(nn) > 5:
                    X = sm.add_constant(range(len(curTSeries[nn])))
                    mdlT = sm.OLS(np.array(curTSeries[nn]).reshape(-1,1), X).fit()
                curTMean = np.nanmean(tData[curLon][:,tLatInd])
                
                curPSeries = pData[curLon][:,tLatInd]
                nn = np.where(~np.isnan(curPSeries))[0]
                if len(nn) > 5:
                    X = sm.add_constant(range(len(curPSeries[nn])))
                    mdlP = sm.OLS(np.array(curPSeries[nn]).reshape(-1,1), X).fit()
                
                tMeans[k]['noCLat'].append(latNew[curLatCoord])
                tMeans[k]['noCLon'].append(lonNew[curLonCoord])
                tMeans[k]['noC'].append(mdlT.params[1])
                pMeans[k]['noC'].append(mdlP.params[1]*365)
    
    tMeans[k]['noCLat'] = np.array(tMeans[k]['noCLat'])
    tMeans[k]['noCLon'] = np.array(tMeans[k]['noCLon'])
    tMeans[k]['noC'] = np.array(tMeans[k]['noC'])
    pMeans[k]['noC'] = np.array(pMeans[k]['noC'])
    #noPCells = koppenGroupsNoPCells[k]
    
kModels = {'A':{}, 'B':{}, 'C':{}, 'D':{}}
    
totalPCells = len(tMeans['A']['P']) + len(tMeans['B']['P']) + len(tMeans['C']['P']) + \
              len(tMeans['D']['P'])
totalCCells = len(tMeans['A']['C']) + len(tMeans['B']['C']) + len(tMeans['C']['C']) + \
              len(tMeans['D']['C'])

allPCover = np.concatenate((tMeans['A']['pCover'], tMeans['B']['pCover'], tMeans['C']['pCover'], \
                            tMeans['D']['pCover']))
allCCover = np.concatenate((tMeans['A']['cCover'], tMeans['B']['cCover'], tMeans['C']['cCover'], \
                            tMeans['D']['cCover']))
allP = np.concatenate((tMeans['A']['P'], tMeans['B']['P'], tMeans['C']['P'], \
                            tMeans['D']['P']))
allC = np.concatenate((tMeans['A']['C'], tMeans['B']['C'], tMeans['C']['C'], \
                            tMeans['D']['C']))
    
nn = np.where((~np.isnan(allP)) & (~np.isnan(allPCover)))[0]
X = sm.add_constant(allPCover.reshape(-1,1))
mdl = sm.OLS(allP.reshape(-1,1), X).fit()
print('All: T/Pasture: interc = %.2f, coef = %.2f, p = %.2f'%(mdl.params[0], mdl.params[1], mdl.pvalues[1]))

nn = np.where((~np.isnan(allC)) & (~np.isnan(allCCover)))[0]
X = sm.add_constant(allCCover.reshape(-1,1))
mdl = sm.OLS(allC.reshape(-1,1), X).fit()
print('All: T/Crop: interc = %.2f, coef = %.2f, p = %.2f'%(mdl.params[0], mdl.params[1], mdl.pvalues[1]))

for k in koppenGroupsPCells.keys():
    print('%s: %.2f total P'%(k, len(tMeans[k]['P'])/totalPCells))
    print('%s: %.2f total C'%(k, len(tMeans[k]['C'])/totalCCells))
    
    nn = np.where((~np.isnan(tMeans[k]['P'])) & (~np.isnan(tMeans[k]['pCover'])))[0]
    if len(tMeans[k]['P'][nn]) > 5:
        X = sm.add_constant(np.array(tMeans[k]['pCover'][nn]).reshape(-1,1))
        mdl = sm.OLS(np.array(tMeans[k]['P'][nn]).reshape(-1,1), X).fit()
        kModels[k]['tP'] = mdl
        print('%s: T/Pasture: interc = %.2f, coef = %.2f, p = %.2f'%(k, mdl.params[0], mdl.params[1], mdl.pvalues[1]))
    
    nn = np.where((~np.isnan(pMeans[k]['P'])) & (~np.isnan(pMeans[k]['pCover'])))[0]
    if len(pMeans[k]['P'][nn]) > 5:
        X = sm.add_constant(np.array(pMeans[k]['pCover'][nn]).reshape(-1,1))
        mdl = sm.OLS(np.array(pMeans[k]['P'][nn]).reshape(-1,1), X).fit()
        kModels[k]['pP'] = mdl
        print('%s: P/Pasture: interc = %.2f, coef = %.2f, p = %.2f'%(k, mdl.params[0], mdl.params[1], mdl.pvalues[1]))
    
    nn = np.where((~np.isnan(tMeans[k]['C'])) & (~np.isnan(tMeans[k]['cCover'])))[0]
    if len(tMeans[k]['C'][nn]) > 5:
        X = sm.add_constant(np.array(tMeans[k]['cCover'][nn]).reshape(-1,1))
        mdl = sm.OLS(np.array(tMeans[k]['C'][nn]).reshape(-1,1), X).fit()
        kModels[k]['tC'] = mdl
        print('%s: T/Crop: interc = %.2f, coef = %.2f, p = %.2f'%(k, mdl.params[0], mdl.params[1], mdl.pvalues[1]))
    
    nn = np.where((~np.isnan(pMeans[k]['C'])) & (~np.isnan(pMeans[k]['cCover'])))[0]
    if len(pMeans[k]['C'][nn]) > 5:
        X = sm.add_constant(np.array(pMeans[k]['cCover'][nn]).reshape(-1,1))
        mdl = sm.OLS(np.array(pMeans[k]['C'][nn]).reshape(-1,1), X).fit()
        kModels[k]['pC'] = mdl
        print('%s: P/Crop: interc = %.2f, coef = %.2f, p = %.2f'%(k, mdl.params[0], mdl.params[1], mdl.pvalues[1]))
    print()

sys.exit()

dist = st.norm
for i, k in enumerate(koppenGroupsPCells.keys()):
    
    pPerc = len(tMeans[k]['P'])/totalPCells*100
    cPerc = len(tMeans[k]['C'])/totalCCells*100
    
    paramsDistC = dist.fit(tMeans[k]['C'])
    argDistC = paramsDistC[:-2]
    locDistC = paramsDistC[-2]
    scaleDistC = paramsDistC[-1]
    
    paramsDistNoC = dist.fit(tMeans[k]['noC'])
    argDistNoC = paramsDistNoC[:-2]
    locDistNoC = paramsDistNoC[-2]
    scaleDistNoC = paramsDistNoC[-1]
    
    if len(tMeans[k]['noC']) == 0 or len(tMeans[k]['C']) == 0:
        continue
    
    x = np.linspace(np.nanmin(np.concatenate((tMeans[k]['noC'], tMeans[k]['C']))), \
                    np.nanmax(np.concatenate((tMeans[k]['noC'], tMeans[k]['C']))), 50)
    
    pdfDistNoC = dist.pdf(x, loc=locDistNoC, scale=scaleDistNoC, *argDistNoC)
    pdfDistC = dist.pdf(x, loc=locDistC, scale=scaleDistC, *argDistC)
    
    
    xMaxC = np.where((pdfDistC == np.nanmax(pdfDistC)))[0]
    xMaxNoC = np.where((pdfDistNoC == np.nanmax(pdfDistNoC)))[0]
    
    nnC = np.where(~np.isnan(pdfDistC))[0]
    nnNoC = np.where(~np.isnan(pdfDistNoC))[0]
    
    if len(nnC) > 5 and len(nnNoC) > 5:
        plt.figure(figsize=(4, 4))
        plt.grid(True, color=[.9, .9, .9])
        
        plt.plot(x, pdfDistC, 'g', lw=2, label='Cropland')
        plt.plot(x, pdfDistNoC, 'm', lw=2, label='No crop')
        
        plt.plot([x[xMaxC], x[xMaxC]], [0, np.nanmax(pdfDistC)], '--g', lw=2)
        plt.plot([x[xMaxNoC], x[xMaxNoC]], [0, np.nanmax(pdfDistNoC)], '--m', lw=2)
        
        plt.title('Classification: %s (%.1f%% total)'%(k, cPerc), fontname = 'Helvetica', fontsize=16)
        plt.legend(markerscale=2, prop = {'size':10, 'family':'Helvetica'}, frameon=False)
        
        plt.xlabel('Temperature ($\degree$C)', fontname = 'Helvetica', fontsize=16)
        plt.ylabel('Density', fontname = 'Helvetica', fontsize=16)
        
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontname('Helvetica')
            tick.label.set_fontsize(14)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontname('Helvetica')    
            tick.label.set_fontsize(14)

        
#        plt.savefig('koppen-t-crop-%s.eps'%k, format='eps', dpi=500, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
sys.exit()
    
#for i, k in enumerate(koppenGroupsPCells.keys()):
#    
#    paramsDistP = dist.fit(tMeans[k]['P'])
#    argDistP = paramsDistP[:-2]
#    locDistP = paramsDistP[-2]
#    scaleDistP = paramsDistP[-1]
#    
#    paramsDistNoP = dist.fit(tMeans[k]['noP'])
#    argDistNoP = paramsDistNoP[:-2]
#    locDistNoP = paramsDistNoP[-2]
#    scaleDistNoP = paramsDistNoP[-1]
#    
#    if len(tMeans[k]['noP']) == 0 or len(tMeans[k]['P']) == 0:
#        continue
#    
#    x = np.linspace(np.nanmin(np.concatenate((tMeans[k]['noP'], tMeans[k]['P']))), \
#                    np.nanmax(np.concatenate((tMeans[k]['noP'], tMeans[k]['P']))), 50)
#    
#    pdfDistNoP = dist.pdf(x, loc=locDistNoP, scale=scaleDistNoP, *argDistNoP)
#    pdfDistP = dist.pdf(x, loc=locDistP, scale=scaleDistP, *argDistP)
#    
#    
#    xMaxP = np.where((pdfDistP == np.nanmax(pdfDistP)))[0]
#    xMaxNoP = np.where((pdfDistNoP == np.nanmax(pdfDistNoP)))[0]
#    
#    nnP = np.where(~np.isnan(pdfDistP))[0]
#    nnNoP = np.where(~np.isnan(pdfDistNoP))[0]
#    
#    if len(nnP) > 5 and len(nnNoP) > 5:
#        plt.figure(figsize=(4, 4))
#        plt.grid(True, color=[.9, .9, .9])
#        
#        plt.plot(x, pdfDistP, 'g', lw=2, label='Pasture')
#        plt.plot(x, pdfDistNoP, 'm', lw=2, label='Not pasture')
#        
#        plt.plot([x[xMaxP], x[xMaxP]], [0, np.nanmax(pdfDistP)], '--g', lw=2)
#        plt.plot([x[xMaxNoP], x[xMaxNoP]], [0, np.nanmax(pdfDistNoP)], '--m', lw=2)
#        
#        plt.title('Classification: %s (%.1f%% total)'%(k, cPerc), fontname = 'Helvetica', fontsize=16)
#        plt.legend(markerscale=2, prop = {'size':10, 'family':'Helvetica'}, frameon=False)
#        
#        plt.xlabel('Temperature ($\degree$C)', fontname = 'Helvetica', fontsize=16)
#        plt.ylabel('Density', fontname = 'Helvetica', fontsize=16)
#        
#        for tick in plt.gca().xaxis.get_major_ticks():
#            tick.label.set_fontname('Helvetica')
#            tick.label.set_fontsize(14)
#        for tick in plt.gca().yaxis.get_major_ticks():
#            tick.label.set_fontname('Helvetica')    
#            tick.label.set_fontsize(14)
#
#        
#        plt.savefig('koppen-t-pasture-%s.eps'%k, format='eps', dpi=500, bbox_inches = 'tight', pad_inches = 0)
#    plt.show()
    
    

plt.figure(figsize=(4, 4))
plt.grid(True, color=[.9, .9, .9])

xs = np.linspace(0,1,10)

colors = ['r', 'm', 'g', 'b']
xpos = [.95, .98, 1.01, 1.04]
for i, k in enumerate(['A', 'B', 'C', 'D']):
    if i == 0:
        plt.plot(xpos[i], np.nanmean(tMeans[k]['noC']), 'ok', ms=6, label='No crops', mfc=colors[i])
    else:
        plt.plot(xpos[i], np.nanmean(tMeans[k]['noC']), 'ok', ms=6, mfc=colors[i])
    plt.errorbar(xpos[i], np.nanmean(tMeans[k]['noC']), yerr = np.nanstd(tMeans[k]['noC']), lw=2, color=colors[i], \
                 elinewidth = 1, capsize = 3, fmt = 'none')
    
    
    if len(tMeans[k]['C']) > 5:
        ys = []
        for x in xs:
            ys.append(kModels[k]['tC'].predict([1, x]))
        
        plt.plot(xpos[i]+xs, ys, '--', label='Region %s'%k, color=colors[i])

plt.title('T vs. Cropland Fraction', fontname = 'Helvetica', fontsize=16)
plt.legend(markerscale=2, prop = {'size':12, 'family':'Helvetica'}, frameon=False, \
           bbox_to_anchor=(1, .65))

plt.xticks([1, 1.2, 1.4, 1.6, 1.8, 2])
plt.gca().set_xticklabels([0, .2, .4, .6, .8, 1])

plt.xlabel('Cropland fraction', fontname = 'Helvetica', fontsize=16)
plt.ylabel('Temperature ($\degree$C)', fontname = 'Helvetica', fontsize=16)

for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontname('Helvetica')
    tick.label.set_fontsize(14)
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontname('Helvetica')    
    tick.label.set_fontsize(14)







plt.figure(figsize=(4, 4))
plt.grid(True, color=[.9, .9, .9])

xs = np.linspace(0,1,10)

colors = ['r', 'm', 'g', 'b']
xpos = [.95, .98, 1.01, 1.04]
for i, k in enumerate(['A', 'B', 'C', 'D']):
    if i == 0:
        plt.plot(xpos[i], np.nanmean(tMeans[k]['noP']), 'ok', ms=6, label='No pasture', mfc=colors[i])
    else:
        plt.plot(xpos[i], np.nanmean(tMeans[k]['noP']), 'ok', ms=6, mfc=colors[i])
    plt.errorbar(xpos[i], np.nanmean(tMeans[k]['noP']), yerr = np.nanstd(tMeans[k]['noP']), lw=2, color=colors[i], \
                 elinewidth = 1, capsize = 3, fmt = 'none')
    
    
    if len(tMeans[k]['P']) > 5:
        ys = []
        for x in xs:
            ys.append(kModels[k]['tP'].predict([1, x]))
        
        plt.plot(xpos[i]+xs, ys, '--', label='Region %s'%k, color=colors[i])

plt.title('T vs. Pasture Fraction', fontname = 'Helvetica', fontsize=16)
plt.legend(markerscale=2, prop = {'size':12, 'family':'Helvetica'}, frameon=False, \
           bbox_to_anchor=(1, .65))

plt.xticks([1, 1.2, 1.4, 1.6, 1.8, 2])
plt.gca().set_xticklabels([0, .2, .4, .6, .8, 1])

plt.xlabel('Pasture fraction', fontname = 'Helvetica', fontsize=16)
plt.ylabel('Temperature ($\degree$C)', fontname = 'Helvetica', fontsize=16)

for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontname('Helvetica')
    tick.label.set_fontsize(14)
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontname('Helvetica')    
    tick.label.set_fontsize(14)



sys.exit()

plt.savefig('t-cropland-regression.eps', format='eps', dpi=500, bbox_inches = 'tight', pad_inches = 0)

plt.imshow(koppen)
plt.title('Koppen classification', fontname = 'Helvetica', fontsize=16)
cb = plt.colorbar(orientation='horizontal')
cb.set_ticks([1,2,3,4,5])
cb.set_ticklabels(['A', 'B', 'C', 'D', 'E'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
plt.savefig('koppen-map.png', format='png', dpi=500, bbox_inches = 'tight', pad_inches = 0)


plt.imshow(cropRegrid)
plt.title('Cropland fraction', fontname = 'Helvetica', fontsize=16)
cb = plt.colorbar(orientation='horizontal')
cb.set_ticks([0, .25, .5, .75, 1])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
plt.savefig('crop-frac-map.png', format='png', dpi=500, bbox_inches = 'tight', pad_inches = 0)

plt.imshow(pastureRegrid)
plt.title('Pasture fraction', fontname = 'Helvetica', fontsize=16)
cb = plt.colorbar(orientation='horizontal')
cb.set_ticks([0, .25, .5, .75, 1])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off


























#        sacksInds = []
#        growingMaize = False
#        growingSoybeans = False
#        growingRice = False
#        growingWheat = False
#        for i in dayNumbers:
#            if np.isnan(i): continue
#        
#            if i == sacksCal['start_maize'][cCells[p][0]*4, cCells[p][1]*4]:
#                growingMaize = True
#            elif growingMaize and i == sacksCal['end_maize'][cCells[p][0]*4, cCells[p][1]*4]:
#                growingMaize = False
#                
#            if i == sacksCal['start_soybeans'][cCells[p][0]*4, cCells[p][1]*4]:
#                growingSoybeans = True
#            elif growingSoybeans and i == sacksCal['end_soybeans'][cCells[p][0]*4, cCells[p][1]*4]:
#                growingSoybeans = False
#                
#            if i == sacksCal['start_rice'][cCells[p][0]*4, cCells[p][1]*4]:
#                growingRice = True
#            elif growingRice and i == sacksCal['end_rice'][cCells[p][0]*4, cCells[p][1]*4]:
#                growingRice = False
#                
#            if i == sacksCal['start_wheat'][cCells[p][0]*4, cCells[p][1]*4]:
#                growingWheat = True
#            elif growingWheat and i == sacksCal['end_wheat'][cCells[p][0]*4, cCells[p][1]*4]:
#                growingWheat = False
#            if growingMaize or growingSoybeans or growingRice or growingWheat: sacksInds.append(int(i))
#        sacksInds = np.array(sacksInds)
#        
#        if sacksInds.shape[0] == 0: continue