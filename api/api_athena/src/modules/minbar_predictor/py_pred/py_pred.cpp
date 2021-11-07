#include "py_pred.h"
/*************************************************************************
The python script must provide following functions:
1. init(date,time,open,high,low,close,tickvol), which takes initial data
    each argument is a list. 'date' and 'time' are string.
    The others are double

2. appendMinbar(date,time,open,high,low,close,tickvol), which appends
   a new entry of minbar but returns nothing

3. predict(new_open), which returns decision

**************************************************************************/
