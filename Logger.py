import time
import sys
import logging
from datetime import datetime as dt
from enum import Enum


class LogLevel(Enum):
    quiet = 1
    standard = 2
    verbose = 3
    debug = 4


log_level = LogLevel.standard
python_logger_init = False

def init_logger(_logs_path= "/home/logs/", _logs_filename= "debug.log"):
    logs_path = _logs_path
    logs_filename = _logs_filename
    global python_logger_init
    logging.basicConfig(filename=logs_path+logs_filename, filemode='w', level=logging.NOTSET)
    python_logger_init = True

def log(message, severity=LogLevel.standard):
    """Prints message to standard output if the message severity is less than or equal the current logging level. Also
    flushes stdout after sending to try make it work with MPI"""
    global python_logger_init
    fmt_msg = "[{0}] {1}".format(dt.now().strftime('%H:%M'), message)
    if severity.value <= log_level.value:
        print(fmt_msg)
        sys.stdout.flush()
    if python_logger_init:
        logging.log(severity.value, fmt_msg)

def debug(message):
    """A wrapper for Logger.log(msg, LogLevel.debug) to log messages to console with debug level severity."""
    log(message, severity=LogLevel.debug)

class Runtime:
    def __init__(self, label='Total runtime: ', _log_level=LogLevel.quiet):
        self.start_time = time.time()
        self.label = label
        self.log_severity = _log_level

    def format_time(self, time):
        '''Return a string of an ETA given in seconds into a HH:MM:SS format.'''
        m, s = divmod(time, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        return "{0:2.0f} days {1:2.0f} hours {2:2.0f} minutes {3:2.0f} seconds".format(d,h,m,s)

    def now(self):
        """Returns time in seconds since the Runtime object was instantiated."""
        return time.time() - self.start_time

    def print(self):
        """Prints the elapsed time since the Runtime object was instantiated in a human readable form"""
        runtime = time.time() - self.start_time
        formatted_time_string = self.format_time(runtime)
        log(self.label + formatted_time_string, self.log_severity)


class PrintProgress(object):
    '''Manages the printing of progress of a loop to the standard output stream. Simply initialise the object outside of the loop
        by defining the start and end of the loop, then call the update function every loop. e.g.:
                
        prog = PrintProgress(0,100,interval=5,label="Process {0} at ".format(name))
        for i in range(0,100):
            DoSomething()
            prog.update()        
        
        Version: 1.0.3
        Author: Richard Grimes
        E-Mail: rig12@aber.ac.uk
        Date: 2019-06-24
        '''
    def __init__(self, start, end, interval=10,label="Progress: ", _log_level=LogLevel.quiet):
        '''Start an instance of the progress tracker.'''
        self.start = start
        self.end = end
        self.interval = interval
        self.label = label
        self.log_severity = _log_level
        
        self.currentIteration = 0
        self.startTime = time.time()
        self.lastPrintPercent = 0
        self.update(0)
    
    def getETA(self):
        '''Get the ETA of finishing the process assuming a linear progression'''
        self.currentTime = time.time()
        self.deltaTime = (self.currentTime - self.startTime)
        self.deltaProgress = self.currentIteration #(self.currentIteration - self.start)
        if self.deltaProgress == 0: self.deltaProgress = 1
        averageTime = self.deltaTime / self.deltaProgress
        return (self.end - self.currentIteration - self.start) * averageTime
    
    def formatETA(self, eta):
        '''Return a string of an ETA given in seconds into a HH:MM:SS format.'''
        m, s = divmod(eta, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)
    
    def printProgress(self, percentProgress, eta):
        log("{0} {1}% | ETA: {2}".format(self.label, percentProgress, eta),
            self.log_severity)
                
    
    def update(self, iterationsCompleted=1):
        '''Call this at the end of every iteration of the loop to determine whether or not a print statement should be issued'''
        self.currentIteration += iterationsCompleted
        try:
            percentProgress = round((self.currentIteration/(self.end - self.start)) * 100)
        except ZeroDivisionError:
            percentProgress = 100
        if(percentProgress >= self.lastPrintPercent + self.interval or self.interval == -1 or iterationsCompleted == 0):
            self.lastPrintPercent = percentProgress
            eta = self.getETA()
            self.printProgress(percentProgress, self.formatETA(eta))
            
    