import math
import warnings
from sklearn import tree
from scipy import stats
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from Trained_NN import TrainedNN, AbstractNN, ParameterPool
import time, gc, json
import os
import csv

#The amount of data for each model
BLOCK_SIZE = 30

# Number of cells divided in the x and y directions
cx = 8
cy = 8


#Serialize the decision tree and save it to a pickle file.
def storeTree(inputTree,filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


# Store x and y for each model
def storeXY(X,Y,filename):
    with open(filename, 'w', encoding="UTF-8", newline='') as csvFile:
        csv_writer = csv.writer(csvFile)
        lens = len(X)
        for i in range(lens):
            csv_writer.writerow([X[i][0],Y[i][0]])


# Convert numpy classes to JSON serializable objects.
def default_dump(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# hybrid training structure, 2 stages
def hybrid_training(threshold, use_threshold, stage_nums, core_nums, train_step_nums, batch_size_nums, learning_rate_nums,
                    keep_ratio_nums, train_data_x, train_data_y, test_data_x, test_data_y):
    mins = []
    maxs = []
    stage_length = len(stage_nums)
    TOTAL_NUMBER = len(train_data_x)
    divisor = int(TOTAL_NUMBER/BLOCK_SIZE)
    if divisor == 0:
        divisor = 1
    col_num = stage_nums[1]
    tmp_inputs = [[[] for i in range(col_num)] for i in range(stage_length)]
    tmp_labels = [[[] for i in range(col_num)] for i in range(stage_length)]
    index = [[None for i in range(col_num)] for i in range(stage_length)]
    tmp_inputs[0][0] = train_data_x
    tmp_labels[0][0] = train_data_y
    test_inputs = test_data_x
    for i in range(0, stage_length):
        for j in range(0, stage_nums[i]):
            if len(tmp_labels[i][j]) == 0:
                continue
            inputs = tmp_inputs[i][j]
            labels = []
            test_labels = []
            if i == 0:
                minx = np.min(tmp_labels[i][j])
                maxx = np.max(tmp_labels[i][j])
                labels = []
                for t in tmp_labels[i][j]:
                    k = float(t - minx) / (maxx - minx)
                    labels.append(int(k*divisor))
                test_labels = labels
            else:
                minx = np.min(tmp_labels[i][j])
                maxx = np.max(tmp_labels[i][j])
                mins.append(minx)
                maxs.append(maxx)
                labels = []
                for t in tmp_labels[i][j]:
                    k = float(t - minx) / (maxx - minx)
                    labels.append(k)
                test_labels = labels


            tmp_index = TrainedNN(j, threshold[i], use_threshold[i], core_nums[i], train_step_nums[i], batch_size_nums[i],
                                    learning_rate_nums[i],
                                    keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)
            tmp_index.train()
            index[i][j] = AbstractNN(tmp_index.get_weights(), tmp_index.get_bias(), core_nums[i], tmp_index.cal_err())
            del tmp_index
            gc.collect()
            if i < stage_length - 1:
                for ind in range(len(tmp_inputs[i][j])):
                    p = index[i][j].predict(tmp_inputs[i][j][ind])
                    if i == 0:
                        p = round(p)
                    if p > divisor - 1:
                        p = divisor - 1
                    if p < 0:
                        p = 0
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

    return index,mins,maxs



# main function for training idnex
def train_index(i,j,modelNumDict,meanErrorList):
    filename = "dataX" + str(i) + "Y" + str(j)
    # threshold for train (judge whether stop train)
    threshold = [0.01, 0.000004]

    # whether use threshold to stop train for models in stages
    use_threshold = [True, True]

    data = pd.read_csv("./modelXY/"+filename+".csv", header=None)
    data.columns = ['X', 'Y']
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []
    parameter = ParameterPool.RANDOM.value
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    global TOTAL_NUMBER
    TOTAL_NUMBER = data.shape[0]
    stage_set = parameter.stage_set
    stage_set[1] = int(TOTAL_NUMBER/BLOCK_SIZE)
    if stage_set[1] == 0:
        stage_set[1] = 1

    modelNumDict[str(i*cx+j)] = stage_set[1]

    inputs = []
    labels = []

    for index, row in data.iterrows():
        inputs.append(row["X"])
        labels.append(row["Y"])
    train_set_x = inputs
    train_set_y = labels
    test_set_x = train_set_x
    test_set_y = train_set_y

    print("*************cell("+filename+") model start Learned NN************")
    start_time = time.time()
    # train index
    trained_index,mins,maxs = hybrid_training(threshold, use_threshold, stage_set, core_set, train_step_set, batch_size_set, learning_rate_set,
                                    keep_ratio_set, train_set_x, train_set_y, [], [])
    end_time = time.time()
    learn_time = end_time - start_time
    print("Build Learned NN time ", learn_time)
    err = 0
    start_time = time.time()
    # calculate error
    for ind in range(len(test_set_x)):
        # pick model in next stage
        pre1 = round(trained_index[0][0].predict(test_set_x[ind]))
        if pre1 < 0:
            pre1 = 0
        if pre1 > stage_set[1] - 1:
            pre1 = stage_set[1] - 1
        # predict position
        pre2 = trained_index[1][pre1].predict(test_set_x[ind])
        pre2 = int(pre2*(maxs[pre1]-mins[pre1]) + mins[pre1])
        if pre2 < 0:
            pre2 = 0
        err += abs(pre2 - test_set_y[ind])
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time %f " % search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    meanErrorList.append(mean_error)
    print("*************cell("+filename+") model end Learned NN************\n")

    # write parameter into files
    with open("./model/X"+str(cx)+"Y"+str(cy)+"/minMax/"+filename+"MinMax.csv", 'w', encoding="UTF-8", newline='') as csvFile:
        csv_writer = csv.writer(csvFile)
        lens = len(mins)
        for minIndex in range(lens):
            csv_writer.writerow([mins[minIndex],maxs[minIndex]])

    result_stage1 = {0: {"weights": trained_index[0][0].weights, "bias": trained_index[0][0].bias}}
    result_stage2 = {}
    for ind in range(len(trained_index[1])):
        if trained_index[1][ind] is None:
            continue
        result_stage2[ind] = {"weights": trained_index[1][ind].weights,
                              "bias": trained_index[1][ind].bias}
    result = [{"stage": 1, "parameters": result_stage1}, {"stage": 2, "parameters": result_stage2}]
    with open('./model/X'+str(cx)+"Y"+str(cy)+'/NN/' + filename + ".json", "w",encoding="UTF-8") as jsonFile:
        json.dump(result, jsonFile,default=default_dump)

    # wirte performance into files
    performance_NN = {"type": "NN", "build time": learn_time, "search time": search_time, "average error": mean_error,
                      "store size": os.path.getsize(
                          './model/X'+str(cx)+"Y"+str(cy)+'/NN/' + filename + ".json")}
    with open("./performance/X"+str(cx)+"Y"+str(cy)+"/NN/" + filename + ".json",
              "w",encoding="UTF-8") as jsonFile:
        json.dump(performance_NN, jsonFile,default=default_dump)

    del trained_index
    gc.collect()


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

# Simplify time values
def timeToIndex(dateList):
    length = len(dateList)
    index = []
    start_time = temporalData.iloc[0]["recv_date"]
    start_time_stamp = time.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    start_time_timestamp = int(time.mktime(start_time_stamp))
    for o in range(length):
        query_time_stamp = time.strptime(dateList[o], "%Y-%m-%d %H:%M:%S")
        query_time_timestamp = int(time.mktime(query_time_stamp))
        index.append(int((query_time_timestamp - start_time_timestamp) / 60))
    return index

# Output error messages in multiple processes
def err_call_back(err):
    print(f'出错啦~ error：{str(err)}')

# main function
if __name__=='__main__':
    warnings.filterwarnings('ignore')

    data = pd.read_csv('./trafficData/device.csv', header=None)
    data.columns = ['device', 'Lng', 'Lat']

    bounds = [115.416827, 39.442078, 117.508251, 41.058964]

    rx = bounds[2] - bounds[0]
    ry = bounds[3] - bounds[1]

    lng = data[['Lng']].values.flatten()
    res_freq = stats.relfreq(lng, numbins=len(lng))
    cdf_value = np.cumsum(res_freq.frequency)
    cdf_value[-1] = 1
    x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    X = x.reshape(-1, 1)
    Y = cdf_value.reshape(-1, 1)
    lngClf = tree.DecisionTreeRegressor()
    lngRbf = lngClf.fit(X, Y)
    if not os.path.exists("./model/X"+str(cx)+"Y"+str(cy)):
        os.mkdir("./model/X"+str(cx)+"Y"+str(cy))
    storeTree(lngRbf, "./model/X"+str(cx)+"Y"+str(cy)+"/lngRbf")


    lat = data[['Lat']].values.flatten()
    res_freq = stats.relfreq(lat, numbins=len(lat))
    cdf_value = np.cumsum(res_freq.frequency)
    cdf_value[-1] = 1
    x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    X = x.reshape(-1, 1)
    Y = cdf_value.reshape(-1, 1)
    latClf = tree.DecisionTreeRegressor()
    latRbf = latClf.fit(X, Y)
    storeTree(latRbf, "./model/X"+str(cx)+"Y"+str(cy)+"/latRbf")


    print("start build model")
    startBuild_time = time.time()

    data['cellX'] = findCellX(data[['Lng']])
    data['cellY'] = findCellY(data[['Lat']])

    temporalData = pd.read_csv('./trafficData/data.csv', header=None)
    temporalData.columns = ['device', 'platenoNum', 'recv_date']

    temporalData['timeIndex'] = timeToIndex(temporalData['recv_date'])
    mergeData = pd.merge(data, temporalData, how='inner', on='device')
    mergeOrderData = mergeData.sort_values(by=['cellX', 'cellY', 'timeIndex'])
    dataLen = len(mergeOrderData)
    index = list(range(dataLen))
    mergeOrderData['index'] = index
    mergeOrderDataDrop = mergeOrderData.drop_duplicates(subset=['cellX', 'cellY', 'recv_date'], keep='first')

    start_index = 0
    end_index = 0
    cell_range = []
    for i in range(cx + 1):
        rangeList = []
        for j in range(cy + 1):
            rangeList.append([0, 0])
        cell_range.append(rangeList)
    index = 0
    dataSelect = mergeOrderDataDrop.iloc[0]

    basic_index = []
    for i in range(cx + 1):
        basicIndexList = []
        for j in range(cy + 1):
            basicIndexList.append(0)
        basic_index.append(basicIndexList)

    cellX = dataSelect["cellX"]
    cellY = dataSelect["cellY"]
    for i, row in mergeOrderDataDrop.iterrows():
        if row["cellX"] == cellX and row["cellY"] == cellY:
            index += 1
        else:
            end_index = index - 1
            cell_range[cellX][cellY] = [start_index, end_index]
            basic_index[cellX][cellY] = mergeOrderDataDrop.iloc[start_index]["index"]
            start_index = index
            cellX = row["cellX"]
            cellY = row["cellY"]
            index += 1
    cell_range[cellX][cellY] = [start_index, index - 1]
    basic_index[cellX][cellY] = mergeOrderDataDrop.iloc[start_index]["index"]

    modelList = []
    for i in range(cx + 1):
        models = []
        for j in range(cy + 1):
            models.append("")
        modelList.append(models)

    pool = multiprocessing.Pool(processes=5)
    manager = multiprocessing.Manager()
    modelNumDict = manager.dict()
    meanErrorList = manager.list()
    modelNum = []
    for i in range(cx + 1):
        models = []
        for j in range(cy + 1):
            models.append(0)
        modelNum.append(models)

    if not os.path.exists('./model/X'+str(cx)+"Y"+str(cy)+'/NN'):
        os.mkdir('./model/X'+str(cx)+"Y"+str(cy)+'/NN')
    if not os.path.exists('./model/X'+str(cx)+"Y"+str(cy)+'/minMax'):
        os.mkdir('./model/X'+str(cx)+"Y"+str(cy)+'/minMax')
    if not os.path.exists("./performance/X"+str(cx)+"Y"+str(cy)+"/NN"):
        os.makedirs("./performance/X"+str(cx)+"Y"+str(cy)+"/NN")

    for i in range(cx + 1):
        for j in range(cy + 1):
            if cell_range[i][j][1] != 0:
                x = []
                y = []
                basic = basic_index[i][j]
                for t in range(cell_range[i][j][0], cell_range[i][j][1] + 1):
                    dataSelect = mergeOrderDataDrop.iloc[t]
                    x.append(dataSelect["timeIndex"])
                    y.append(dataSelect["index"] - basic)
                X = np.array(x, dtype='int64').reshape(-1, 1)
                Y = np.array(y, dtype='int64').reshape(-1, 1)
                filename = "./modelXY/dataX" + str(i) + "Y" + str(j) + ".csv"
                storeXY(X, Y, filename)
                pool.apply_async(func=train_index, args=(i,j,modelNumDict,meanErrorList),error_callback=err_call_back)
                modelList[i][j] = filename
    pool.close()
    pool.join()

    xi = 0
    yi = 0
    for key, value in modelNumDict.items():
        location = int(key)
        if location % cx == 0:
            xi = int(location / cx)-1
            yi = cy
        else:
            xi = int(location / cx)
            yi = location % cx
        modelNum[xi][yi]=value

    if not os.path.exists("./modelParameter/X"+str(cx)+"Y"+str(cy)):
        os.mkdir("./modelParameter/X"+str(cx)+"Y"+str(cy))

    json_data = json.dumps(modelList, ensure_ascii=False, indent=4)
    with open('./modelParameter/X'+str(cx)+"Y"+str(cy)+"/data.json", "w", encoding='utf-8') as file:
        file.write(json_data)

    json_data = json.dumps(basic_index,ensure_ascii=False, default=default_dump)
    with open('./modelParameter/X'+str(cx)+"Y"+str(cy)+"/basicIndex.json", "w", encoding='utf-8') as file:
        file.write(json_data)

    json_data = json.dumps(modelNum,ensure_ascii=False, default=default_dump)
    with open('./modelParameter/X'+str(cx)+"Y"+str(cy)+"/modelNum.json", "w", encoding='utf-8') as file:
        file.write(json_data)

    endBuild_time = time.time()
    learn_time = endBuild_time - startBuild_time
    print("build time by index:", learn_time)
    print("error list:")
    print(meanErrorList)
    totalError = 0
    for i in meanErrorList:
        totalError+=i
    print("mean error:",str(totalError/len(meanErrorList)))

    with open("./mergeData/mergeOrderDataX"+str(cx)+"Y"+str(cy)+".csv", 'w', encoding="UTF-8", newline='') as csvFile:
        csv_writer = csv.writer(csvFile)
        for index, row in mergeOrderData.iterrows():
            cellIndex = cx*row["cellX"]+row["cellY"]
            csv_writer.writerow([row['Lng'], row['Lat'], row["platenoNum"], cellIndex, row["recv_date"], row['index']])
