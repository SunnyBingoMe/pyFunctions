# Set Env's

## env detection
import sys
if('ipykernel' in sys.argv[0]):
    using_jupyter_gui = True
    print("In ipy GUI.")
else:
    using_jupyter_gui = False
    print("Not in ipy GUI.")

## some packages 
  # N/A

## sunny functions
import time
from datetime import datetime

def get_datetime_str(format_str="%Y%m%d_%H%M", datetime_t=None):
    if datetime_t is None:
        datetime_t = datetime.now()
    return datetime_t.strftime(format_str)


def get_datetime_str_sec(datetime_t=None):
    if datetime_t is None:
        datetime_t = datetime.now()
    return datetime_t.strftime("%Y%m%d_%H%M%S")

def print_timestamp():
    print(get_datetime_str_sec())

def __LINE__(): # line number, similar to C/C++
    return str(sys._getframe(1).f_lineno)

import os
def get_nr_cpu_threads():
    #for Linux, Unix and MacOS
    if hasattr(os, "sysconf"):
        if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
            #Linux and Unix
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:
            #MacOS X
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    #for Windows
    if os.environ.has_key("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    #return the default value
    return 1

if using_jupyter_gui:
    print("cpu_threads: %d" % get_nr_cpu_threads())

parallel_job_nr = get_nr_cpu_threads() - 1

## data-science extra functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

def select(df, col_str, condition_str):
    return(df[df[col_str] == condition_str])
pd.DataFrame.select = select

def unselect(df, col_str, condition_str):
    return(df[df[col_str] != condition_str])
pd.DataFrame.unselect = unselect


## proj specific functions
  # N/A

#-------------- proj code starts ---------------- #
import logging
log = logging.getLogger(__name__)
my_str = 'foo bar'
log.info('INFO: ... ' + my_str)
log.warning("Warning: Error parsing search response json - skipping this batch")
log.exception('error loading json, json=%s', my_str)
