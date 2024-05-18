import csv
import getopt
import sys
import uuid
import random
import numpy as np
import datetime
import pandas as pd
import math


# help message
def show_help_message():
    help_message = {'command': 'python generateData.py -t <type> -n <dataNumber> -d <deviceNumber>',
                    'type': 'Type: Uniform, Normal. Data distribution type, default value = Normal',
                    'dataNumber': 'Number of traffic spatiotemporal datasets, default value = 100,000',
                    'deviceNumber': 'Number of traffic devices, default value = 10,000'
                    }
    help_message_key = ['command', 'type', 'dataNumber', 'deviceNumber']
    for k in help_message_key:
        print(help_message[k])


# command line
def main(argv):
    type = "Normal"
    dataNumber = 100000
    deviceNumber = 10000
    print(argv)
    try:
        opts, args = getopt.getopt(argv, "t:n:d:")
    except getopt.GetoptError:
        show_help_message()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-t':
            type = arg
        elif opt == '-n':
            dataNumber = int(arg)
        elif opt == '-d':
            deviceNumber = int(arg)
    print(type,dataNumber,deviceNumber)

    # 1.生成监控设备信息
    devices = []
    for i in range(deviceNumber):
        devices.append(uuid.uuid4().hex[:8].upper())
    # 北京经纬度范围
    bounds = [115.416827, 39.442078, 117.508251, 41.058964]

    lngMeanValue = (bounds[2] + bounds[0]) / 2
    lngStandard = (bounds[2] - lngMeanValue) / 3
    lngList = np.random.normal(lngMeanValue, lngStandard, deviceNumber)

    latMeanValue = (bounds[3] + bounds[1]) / 2
    lngStandard = (bounds[3] - latMeanValue) / 3
    latList = np.random.normal(latMeanValue, lngStandard, deviceNumber)
    index = 0
    with open("./trafficData/device.csv", 'w', encoding="UTF-8",
              newline='') as csvFile:
        csv_writer = csv.writer(csvFile)
        if type == "Uniform":
            for i in devices:
                # 这些生成的随机数都是从一个均匀分布中独立抽取的，即每个数生成的概率相等。
                csv_writer.writerow([i, np.random.uniform(bounds[0],bounds[2]),  np.random.uniform(bounds[1],bounds[3])])
                index += 1
        else:
            for i in devices:
                # 经纬度都是正态分布的数据
                csv_writer.writerow([i, lngList[index], latList[index]])
                index += 1

    # 2.生成交通时空数据
    temporalData = pd.read_csv('./trafficData/device.csv', header=None)
    temporalData.columns = ['device', 'Lng', 'Lat']

    start_time = "2023-03-05 06:00:00"

    # 每秒钟多少条数据 设备数*（3/5）
    dataNumberPerSecond = int(deviceNumber*0.6)

    # 向上取整
    timeNum = math.ceil(dataNumber/dataNumberPerSecond)

    with open("./trafficData/trafficData.csv", 'w', encoding="UTF-8",
              newline='') as csvFile:
        csv_writer = csv.writer(csvFile)
        for i in range(timeNum):
            unique_random_numbers_np = np.random.choice(range(0, deviceNumber), size=dataNumberPerSecond, replace=False)
            for j in unique_random_numbers_np:
                csv_writer.writerow([temporalData.iloc[j]["device"], random.randint(1, 10),
                                     datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(
                                         minutes=i)])


if __name__ == "__main__":
    main(sys.argv[1:])