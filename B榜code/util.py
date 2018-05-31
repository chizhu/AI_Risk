# -*- coding: utf-8 -*-
import time


def time_to_timestamp(times):
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(timeArray)
    return timestamp

def timestamp_to_time(timestamp):
    time_local = time.localtime(timestamp)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt

def time_get_year(times):
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    year = timeArray.tm_year
    return year

def time_get_month(times):
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    mon = timeArray.tm_mon
    return mon

def time_get_day(times):
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    day = timeArray.tm_mday
    return day

def time_get_hour(times):
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    hour = timeArray.tm_hour
    return hour

def time_get_minutes(times):
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    min = timeArray.tm_min
    return min

def time_get_seconds(times):
    timeArray = time.strptime(times, "%Y-%m-%d %H:%M:%S")
    sec = timeArray.tm_sec
    return sec


