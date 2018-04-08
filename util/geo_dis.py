'''
Distance calculation by latitude and longitude
'''

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r * 1000

#coding=utf-8

from math import *

def calcDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    ra = 6378.140 
    rb = 6356.755
    flatten = (ra - rb) / ra 
    rad_lat_A = radians(Lat_A)
    rad_lng_A = radians(Lng_A)
    rad_lat_B = radians(Lat_B)
    rad_lng_B = radians(Lng_B)
    pA = atan(rb / ra * tan(rad_lat_A))
    pB = atan(rb / ra * tan(rad_lat_B))
    xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
    c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
    c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (xx + dr)
    return distance

if __name__ == '__main__':
	Lat_A=32.060255; Lng_A=118.796877
	Lat_B=39.904211; Lng_B=116.407395
	distance=calcDistance(Lat_A,Lng_A,Lat_B,Lng_B)
	print('(Lat_A, Lng_A)=({0:10.3f},{1:10.3f})'.format(Lat_A,Lng_A))
	print('(Lat_B, Lng_B)=({0:10.3f},{1:10.3f})'.format(Lat_B,Lng_B))
	print('Distance={0:10.3f} km'.format(distance))

	print (haversine(Lng_A, Lat_A, Lng_B, Lat_B))