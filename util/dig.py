'''
处理切割后的单个文件
处理后得到的文件内容格式为：
taxiID,s_x,s_y,s_time,e_x,e_y，e_time
'''
import sys
import os
import csv


def excavate(infile):
    query = []
    result = {}
    order = 1
    mark = True
    with open(infile,'r')as csv_file:
        all_lines=csv.reader(csv_file)
        num=0
        for sp in all_lines:

            if num==0:
                num+=1
                continue
            sp[4] = round(float(sp[4]), 6)
            sp[5] = round(float(sp[5]), 6)


            if  sp[10] == '0':  # 数据可用性
                continue;
                # 载客状态0 False 1 True
            if sp[9] == '1.0':
                if mark is False:
                    # 乘客上车
                    query.append(repr(sp[4]))
                    query.append(repr(sp[5]))
                    query.append(sp[1] + '\n')
                    result[order] = query
                    order += 1
                    query = []

                mark = True
            else:
                if mark is True:  # 乘客下车
                    query = [sp[3], repr(sp[4]), repr(sp[5]), sp[1]]

                mark = False
    #print(result)
    return result

'''
对切割后的数据进行处理
raw_gps为存放taxiID数据单元csv的目录
对该目录中所有csv文件遍历处理
处理结果存至pre.txt中
'''
def record(folder):
    dir = os.listdir(folder)
    flink = open('pre.txt', 'a+')
    for infile in dir:
        data=excavate(folder+'/'+infile)

        for i in data:
            flink.writelines(','.join(data[i]))
    flink.close()



def main():
	folder = 'raw_gps'
	record(folder)

if __name__ == '__main__':
	main()
