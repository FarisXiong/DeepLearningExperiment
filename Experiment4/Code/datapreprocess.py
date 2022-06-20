import pandas as pd
import numpy as np
import time
import datetime

def weekDayIndexTrain(row):
    time_cur = row['Date Time']
    time_end = datetime.datetime.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    time_start = datetime.datetime.strptime("01.01.2009 00:10:00", "%d.%m.%Y %H:%M:%S")
    return int((time_end - time_start).days % 7)

def weekDayIndexTest(row):
    time_cur = row['Date Time']
    time_end = datetime.datetime.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    time_start = datetime.datetime.strptime("01.01.2015 00:10:00", "%d.%m.%Y %H:%M:%S")
    return int((time_end - time_start).days % 7)

def weekIndexTrain(row):
    time_cur = row['Date Time']
    time_end = datetime.datetime.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    time_start = datetime.datetime.strptime("01.01.2009 00:10:00", "%d.%m.%Y %H:%M:%S")
    return int((time_end - time_start).days / 7)

def weekIndexTest(row):
    time_cur = row['Date Time']
    time_end = datetime.datetime.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    time_start = datetime.datetime.strptime("01.01.2015 00:10:00", "%d.%m.%Y %H:%M:%S")
    return int((time_end - time_start).days / 7)

def hourSin(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    return float(np.sin(time_val.tm_hour*(2*np.pi/24)))

def hourCos(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    return float(np.cos(time_val.tm_hour*(2*np.pi/24)))

def monthSin(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    mon = time_val.tm_mon
    mon_sin = np.sin(mon*(2*np.pi/12))
    return float(mon_sin)

def monthCos(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    mon = time_val.tm_mon
    mon_cos = np.cos(mon*(2*np.pi/12))
    return float(mon_cos)

def year(row):
    time_cur = row['Date Time']
    time_val = time.strptime(time_cur, "%d.%m.%Y %H:%M:%S")
    return int(time_val.tm_year)

if __name__ == '__main__':

    weatherData = pd.read_csv("../Dataset/jena_climate_2009_2016/jena_climate_2009_2016.csv")
    weatherData['HourSin'] = weatherData.apply(hourSin, axis=1)
    weatherData['HourCos'] = weatherData.apply(hourCos, axis=1)
    weatherData['MonthSin'] = weatherData.apply(monthSin, axis=1)
    weatherData['MonthCos'] = weatherData.apply(monthCos, axis=1)
    weatherData['Year'] = weatherData.apply(year, axis=1)
    rowIndex1 = weatherData[(weatherData['Year'] <= 2014)].index.tolist()
    rowIndex2 = weatherData[(weatherData['Year'] >= 2015)].index.tolist()
    rowIndex1.append(rowIndex1[-1] + 1)
    trainData = weatherData.iloc[rowIndex1]
    rowIndex2 = rowIndex2[1:]
    testData = weatherData.iloc[rowIndex2]
    trainData['WeekIndex'] = trainData.apply(weekIndexTrain, axis=1)
    testData['WeekIndex'] = testData.apply(weekIndexTest, axis=1)
    trainData['WeekDay'] = trainData.apply(weekDayIndexTrain, axis=1)
    testData['WeekDay'] = testData.apply(weekDayIndexTest, axis=1)
    trainData.to_csv("../Dataset/jena_climate_2009_2016/train.csv", index=False)
    testData.to_csv("../Dataset/jena_climate_2009_2016/test.csv", index=False)

