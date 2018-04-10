#coding=utf-8
import sys
from math import ceil
from myTime import time_transfer
from myTime import  time_duration
from direction import direction_index

base = (121.31, 31.08)
# lon 96000
# lat 111000
size = int(sys.argv[2])

def grid_index(base, lon, lat, size):
	x, y = lon - base[0], lat - base[1]
	x *= 96000
	y *= 111000
	return (ceil(x / size), ceil(y / size))


def data_format(fname):
	i = 0
	fr = open(fname, 'r')
	fw = open('input' + repr(size) + '.txt', 'a')
	for line in fr:
		# string
		line=line.rstrip('\n')
		tid,s_lon,s_lat,s_time,e_lon,e_lat,e_time = line.split(',')
		s_x, s_y = grid_index(base, float(s_lon), float(s_lat), size)
		e_x, e_y = grid_index(base, float(e_lon), float(e_lat), size)
		angle = direction_index(s_x, s_y, e_x, e_y)
		des=e_x*101+e_y-1
		angle=chr(angle+97)
		e_x-=s_x
		e_y-=e_y

		print (i)
		print (e_time)
		i=i+1
		if len(s_time)<=2:
			s_time="000"+s_time
		if len(e_time) <= 2:
			e_time="000"+e_time

		s_t = time_transfer(s_time)
		e_time=e_time.rstrip()
		last_t=time_duration(s_time,e_time)
		#筛除潜在异常数据点
		if last_t<'60':
			continue
		#tmp = ','.join([repr(s_x), repr(s_y), repr(s_t), repr(e_x), repr(e_y), repr(e_t)])
		tmp = ','.join([tid, repr(s_x)+'-'+repr(s_y), angle, repr(s_t), last_t,repr(e_x)+'x'+repr(e_y)+'y',repr(des)])
		fw.write(tmp + '\n')

if __name__ == '__main__':
	fname = sys.argv[1]
	data_format(fname)
