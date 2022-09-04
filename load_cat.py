import pandas as pd
import numpy as np


class CCats():
    pass

def read_SCEDC(file, lats=[-90, 90], lons=[-180, 180]):
    A = pd.read_csv(file)

    CAT = CCats()
    Ilats = (A.LAT > min(lats)) & ((A.LAT < max(lats)))
    Ilons = (A.LON > min(lons)) & ((A.LON < max(lons)))
    
    I = [Ilats & Ilons]
    
    CAT.Depth = np.asarray(A.DEPTH)[I]
    CAT.Lon = np.asarray(A.LON)[I]
    CAT.Lat = np.asarray(A.LAT)[I]
    CAT.M = np.asarray(A.MAG)[I]
    CAT.datenum = np.asarray(A.datenum)[I]
    
    return CAT