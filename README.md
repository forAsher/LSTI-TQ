# LSTI-TQ
Learned Spatial-Temporal Indexing for Traffic Data Fast Query

Language: Python,you need to install python3.9.x and package tensorflow, pandas, numpy.


## Files Structures
- buildIndex.py：build index file
- scopeQuery.py：perform range query file
- Trained_NN.py：neural network structure
- model/：learned NN model
- performance/：NN performance
- trafficData/：traffic data
- modelParameter/：auxiliary files for neural network models

## Generate TrafficData
use command to run the generateData.py file to generate traffic spatiotemporal data and monitoring device information
```
python generateData.py -t [type] -n [dataNumber] -d [deviceNumber]
```

Parameters:
- 'type': 'Type: Uniform, Normal. Data distribution type, default value = Normal',
- 'dataNumber': 'Number of traffic spatiotemporal datasets, default value = 100,000',
- 'deviceNumber': 'Number of traffic devices, default value = 10,000'

Example:
```
python generateData.py -t Normal -n 100000 -d 10000
```


## Build Index AND Complete Query
use command to run the buildIndex.py file to train models on traffic spatiotemporal data
```
python buildIndex.py
```

use command to run the scopeQuery.py file using the command. Using index to complete query on traffic spatiotemporal data
```
python scopeQuery.py -sn -sa -en -ea -st -et
```

# 中文