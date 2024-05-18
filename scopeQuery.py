import json
import math
import pandas as pd
import warnings
import time
import numpy as np
import multiprocessing
from pandas.core.frame import DataFrame
import pickle
from Trained_NN import AbstractNN


# Number of cells divided in the x and y directions
cx = 8
cy = 8

# query scope
queryBounds = [115.80346, 39.546709, '2023-03-05 06:00:00', 116.505747, 39.927429, '2023-03-05 06:05:00']

mergeOrderData = pd.read_csv("./mergeData/mergeOrderDataX"+str(cx)+"Y"+str(cy)+".csv", header=None)
mergeOrderData.columns = ['Lng','Lat','platenoNum','cellIndex','recv_date','index']

with open('./modelParameter/X'+str(cx)+"Y"+str(cy)+'/data.json', 'r') as load_f:
    data = json.load(load_f)

with open('./modelParameter/X'+str(cx)+"Y"+str(cy)+'/basicIndex.json', 'r') as load_f:
    basicIndex = json.load(load_f)

with open('./modelParameter/X'+str(cx)+"Y"+str(cy)+'/modelNum.json', 'r') as load_f:
    modelNum = json.load(load_f)

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

lngRbf = grabTree("./model/X"+str(cx)+"Y"+str(cy)+"/lngRbf")
latRbf = grabTree("./model/X"+str(cx)+"Y"+str(cy)+"/latRbf")

# Convert longitude values to grid x-axis numbering
def findCellX(lng):
    length = len(lng)
    lngX = lng.values.flatten().reshape(-1, 1)
    cells = []
    lngPercentage = lngRbf.predict(lngX)
    for o in range(length):
        cells.append(math.ceil(lngPercentage[o] * cx))
    return cells

# Convert latitude values to grid y-axis numbering
def findCellY(lat):
    length = len(lat)
    cells = []
    latX = lat.values.flatten().reshape(-1, 1)
    latPercentage = latRbf.predict(latX)
    for o in range(length):
        cells.append(math.ceil(latPercentage[o] * cy))
    return cells

# Decode the encoded values into grid positions
def toXY(cellIndex):
    if cellIndex % cx == 0:
        xi = int(cellIndex / cx) - 1
        yi = cy
        return xi, yi
    else:
        xi = int(cellIndex / cx)
        yi = cellIndex % cx
        return xi, yi
    return 0,0

# Simplify time values
def timeToIndex(dateList):
    length = len(dateList)
    index = []
    start_time = mergeOrderData.iloc[0]["recv_date"]
    start_time_stamp = time.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    start_time_timestamp = int(time.mktime(start_time_stamp))
    for o in range(length):
        query_time_stamp = time.strptime(dateList[o], "%Y-%m-%d %H:%M:%S")
        query_time_timestamp = int(time.mktime(query_time_stamp))
        index.append(int((query_time_timestamp - start_time_timestamp) / 60))
    return index

# Determine if the point is within the latitude and longitude range
def withRange(location,queryBounds):
    startLng = queryBounds[0]
    startLat = queryBounds[1]
    endLng = queryBounds[3]
    endLat = queryBounds[4]
    if location[0] >= startLng and location[0] <= endLng:
        if location[1] >= startLat and location[1] <= endLat:
            return True
    return False

# Calculate the number of vehicles within the query time range in each grid
def calCarNum(i,j,carNumList,queryBounds,queryCellTimeArray):
    basicValue = basicIndex[i][j]
    carNum = 0
    predictIndex = []
    fileName = "dataX" + str(i) + "Y" + str(j)
    file = open('./model/X'+str(cx)+"Y"+str(cy)+'/NN/' + fileName + '.json', 'r')
    content = file.read()
    modelInfo = json.loads(content)
    file.close()
    core_num = [1, 1]
    index = []
    weights = modelInfo[0]["parameters"]["0"]["weights"]
    bias = modelInfo[0]["parameters"]["0"]["bias"]
    indexOne = AbstractNN(weights, bias, core_num, 1)

    for k in range(modelNum[i][j]):
        weights = modelInfo[1]["parameters"][str(k)]["weights"]
        bias = modelInfo[1]["parameters"][str(k)]["bias"]
        index.append(AbstractNN(weights, bias, core_num, 1))

    dataMinMax = pd.read_csv("./model/X"+str(cx)+"Y"+str(cy)+"/minMax/" + fileName + "MinMax.csv", header=None)
    dataMinMax.columns = ['min', 'max']

    for queryTime in queryCellTimeArray:
        pre1 = round(indexOne.predict(queryTime[0]))
        if pre1 < 0:
            pre1 = 0
        if pre1 > modelNum[i][j]-1:
            pre1 = modelNum[i][j]-1
        predictY = index[pre1].predict(queryTime[0])
        minx = dataMinMax.iloc[pre1]["min"]
        maxx = dataMinMax.iloc[pre1]["max"]
        pre2 = int(predictY * (maxx - minx) + minx)
        if pre2 < 0:
            pre2 = 0
        pre2+=basicValue
        predictIndex.append(pre2)
    queryTimeIndex = predictIndex
    queryStartIndex = queryTimeIndex[0]
    queryEndIndex = queryTimeIndex[1]

    #Find the starting physical index location
    queryStartTime = queryBounds[2]
    start_dataframe = mergeOrderData.iloc[queryStartIndex]
    queryStartIndexTime = start_dataframe["recv_date"]
    cellIndex = start_dataframe["cellIndex"]
    xi,yi = toXY(cellIndex)
    if xi == i and yi == j:
        pass
    elif (xi < i) or (xi == i and yi < j):#前面的网格
        while True:
            queryStartIndex += 1
            now_dataframe = mergeOrderData.iloc[queryStartIndex]
            cellIndex = now_dataframe["cellIndex"]
            xi, yi = toXY(cellIndex)
            if xi == i and yi == j:
                break
    elif (xi > i) or (xi == i and yi > j):#后面的网格
        while True:
            queryStartIndex -= 1
            now_dataframe = mergeOrderData.iloc[queryStartIndex]
            cellIndex = now_dataframe["cellIndex"]
            xi, yi = toXY(cellIndex)
            if xi == i and yi == j:
                break

    if queryStartIndexTime < queryStartTime:
        while mergeOrderData.iloc[queryStartIndex]["recv_date"] < queryStartTime:
            queryStartIndex += 1
    elif queryStartIndexTime == queryStartTime:
        while mergeOrderData.iloc[queryStartIndex]["recv_date"] == queryStartTime:
            queryStartIndex -= 1
        queryStartIndex += 1
    elif queryStartIndexTime > queryStartTime:
        while mergeOrderData.iloc[queryStartIndex]["recv_date"] >= mergeOrderData.iloc[queryStartIndex-1]["recv_date"] \
                and mergeOrderData.iloc[queryStartIndex]["recv_date"] >= queryStartTime:
            queryStartIndex -= 1
        if mergeOrderData.iloc[queryStartIndex]["recv_date"] < queryStartTime:
            queryStartIndex += 1



    # Find end physical index position
    queryEndTime = queryBounds[5]
    end_dataframe = mergeOrderData.iloc[queryEndIndex]
    queryEndIndexTime = end_dataframe["recv_date"]
    cellIndex = end_dataframe["cellIndex"]
    xi, yi = toXY(cellIndex)
    if xi == i and yi == j:
        pass
    elif (xi < i) or (xi == i and yi < j):  # 前面的网格
        while True:
            queryEndIndex += 1
            now_dataframe = mergeOrderData.iloc[queryEndIndex]
            cellIndex = now_dataframe["cellIndex"]
            xi, yi = toXY(cellIndex)
            if xi == i and yi == j:
                break
    elif (xi > i) or (xi == i and yi > j):  # 后面的网格
        while True:
            queryEndIndex -= 1
            now_dataframe = mergeOrderData.iloc[queryEndIndex]
            cellIndex = now_dataframe["cellIndex"]
            xi, yi = toXY(cellIndex)
            if xi == i and yi == j:
                break

    try:  # 超过下边界
        if queryEndIndexTime == queryEndTime:
            while mergeOrderData.iloc[queryEndIndex]["recv_date"] == queryEndTime:
                queryEndIndex += 1
            queryEndIndex -= 1
        elif queryEndIndexTime > queryEndTime:
            while mergeOrderData.iloc[queryEndIndex]["recv_date"] > queryEndTime:
                queryEndIndex -= 1
        elif queryEndIndexTime < queryEndTime:
            while mergeOrderData.iloc[queryEndIndex]["recv_date"] <= queryEndTime and mergeOrderData.iloc[queryEndIndex]["recv_date"] <= mergeOrderData.iloc[queryEndIndex + 1]["recv_date"]:
                queryEndIndex += 1
            if mergeOrderData.iloc[queryEndIndex]["recv_date"] > queryEndTime:
                queryEndIndex -= 1
    except Exception as e:
        queryEndIndex -= 1
    for queryIndex in range(queryStartIndex, queryEndIndex + 1):
        dataSelect = mergeOrderData.iloc[queryIndex]
        location = [dataSelect['Lng'], dataSelect['Lat']]
        if withRange(location,queryBounds):
            carNum += dataSelect['platenoNum']
    carNumList.append(carNum)

# Output error messages in multiple processes
def err_call_back(err):
    print(f'出错啦~ error：{str(err)}')

# main function
if __name__=='__main__':
    warnings.filterwarnings('ignore')

    lng = DataFrame([queryBounds[0], queryBounds[3]])
    lat = DataFrame([queryBounds[1], queryBounds[4]])
    timeList = [queryBounds[2], queryBounds[5]]

    queryCellX = findCellX(lng)
    queryCellY = findCellY(lat)
    queryCellTime = timeToIndex(timeList)
    queryCellTimeArray = np.array(queryCellTime, dtype='int64').reshape(-1, 1)
    pool = multiprocessing.Pool(processes=5)
    manager = multiprocessing.Manager()

    deviceNums = []
    deviceNumList = manager.list(deviceNums)
    print("start range query")
    query_start_time = time.time()
    for i in range(queryCellX[0],queryCellX[1]+1):
        for j in range(queryCellY[0],queryCellY[1]+1):
            if data[i][j] != '':
                pool.apply_async(func=calCarNum, args=(i,j,deviceNumList,queryBounds,queryCellTimeArray),error_callback=err_call_back)

    pool.close()
    pool.join()
    deviceSum = 0
    for i in deviceNumList:
        deviceSum+=i

    print("query result by index:", deviceSum)
    query_end_time = time.time()
    learn_time = query_end_time - query_start_time
    print("Search time by index:", learn_time)

    query_start_time = time.time()
    CarNum2 = 0
    for index, row in mergeOrderData.iterrows():
        location = [row['Lng'],row['Lat']]
        if withRange(location,queryBounds):
            queryTime = row['recv_date']
            if queryTime>=timeList[0] and queryTime<=timeList[1]:
                CarNum2+=row['platenoNum']

    print("query result without index:", CarNum2)
    query_end_time = time.time()
    learn_time = query_end_time - query_start_time
    print("Search time  without index:", learn_time)