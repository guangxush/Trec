'''
Transfer time stamp to neural network input
'''

from datetime import datetime as dt

def time_transfer(time):
	hour, minute = time[:2], time[2:4]
	minute = round(float(minute) / 60, 1)
	if 0 <= minute < 0.2:
		minute = 0.0
	elif 0.2 <= minute < 0.4:
		minute = 0.2
	elif 0.4 <= minute < 0.6:
		minute = 0.4
	elif 0.6 <= minute < 0.8:
		minute = 0.6
	else:
		minute = 0.8
	return int(hour) + minute

def time_duration(t1, t2):
	print (t1, t2)
	a, b = dt.strptime(t1, '%H%M%S'), dt.strptime(t2, '%H%M%S')
	return repr((b - a).seconds)

if __name__ == '__main__':
	a = '010203'
	b = '010312'
	c = '134520'

	#time = [a,b,c]
	#print map(time_transfer, time)
	print (time_duration(a, b))
