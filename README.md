# LSTI-TQ
Learned Spatial-Temporal Indexing for Traffic Data Fast Query

Language: Python

## Files Structures
- buildIndex.py：build index file
- scopeQuery.py：perform range query file
- Trained_NN.py：neural network structure
- model/：learned NN model
- performance/：NN performance
- trafficData/：traffic data
- modelParameter/：auxiliary files for neural network models

## HOW TO RUN
#### First, you need to install python3.9.x and package tensorflow, pandas, numpy.
#### Second, use command to run the generateData.py file, that is,

**python generateData.py  -t [type] -n [dataNumber] -d [deviceNumber].**

#### Parameters:
- 'type': 'Type: Uniform, Normal. Data distribution type, default value = Normal',
- 'dataNumber': 'Number of traffic spatiotemporal datasets, default value = 100,000',
- 'deviceNumber': 'Number of traffic devices, default value = 10,000'

# 中文