# -*- coding: utf-8 -*-
"""
Provide logging functions

@author: Lantian ZHANG
"""

import logging
import traceback
from logging import handlers
from functools import wraps

# ============================================================
# Basic Logging
# ============================================================


class BaseLogger(object):
    def __init__(self, logname, level='info', logfile='logfile.log', newfile_when='D'
                 , output_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                 ):
        """Initialize logger
        Parameters:
        ----------
        logname: python string. The name of a logger instance. Different loggers should have differet names
        level: python string. Level of logs. Can choose from 'debug','info','warning','error','critical'. Default is 'info'
        logfile: python string. File name of the log file saved locally.Default is 'logfile.log'
        newfile_when: Frequency of creating new log files. Default is 'D', which means new log file will be created every day.
        output_format: Default log message format.
        """
        self.logname = logname
        self.log_level_ = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL}
        self.logfile_ = logfile
        self.level_ = level
        # When to create a new log file and rename the old log file with date
        # afterfix
        self.newfile_when_ = newfile_when
        self.output_format_ = output_format

        # Initialize a logger instance
        # Initialize logger with a name
        self.logger_ = logging.getLogger(self.logname)
        self.logger_.setLevel(self.log_level_[level])  # Set logging level

        # Clear existing handlers with the same same to avoid duplicated
        # entries
        if self.logger_.hasHandlers():
            self.logger_.handlers.clear()

        # Set up handler for displaying logs on screen
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(output_format))

        # Set up handler for writing logs in file
        th = handlers.TimedRotatingFileHandler(
            filename=self.logfile_, when=newfile_when, encoding='utf-8')
        th.setFormatter(logging.Formatter(output_format))

        # Add handlers to the logger instance
        self.logger_.addHandler(sh)
        self.logger_.addHandler(th)

    def __call__(self, func):
        """Set up logging decorator"""

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            self.logger_.info(
                "function '" + func.__name__ +
                "' was called with args=" +
                str(args) + " and kwargs=" + str(kwargs))
            return func(*args, **kwargs)
        return wrapped_function

# ============================================================
# Exception Recording decorator
# ============================================================


def exception_catch(func):
    """Set up exception catching decorator"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            # exType, exValue, exTrace = sys.exc_info()
            raise Exception(traceback.format_exc())
    return wrapped_function


def exception_logging(func, path=""):
    """Set up exception logging decorator"""

    @wraps(func, path)
    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            log_critical = BaseLogger(
                logname='critical',
                level='critical',
                logfile=path + 'critical_log.log')
            # exType, exValue, exTrace = sys.exc_info()
            # log_critical.logger_.critical([exType, exValue, exTrace])
            # raise Exception([exType, exValue, exTrace])
            log_critical.logger_.critical(
                "function '" + func.__name__ +
                "' was called with args=" +
                str(args) + " and kwargs=" + str(kwargs))
            log_critical.logger_.critical(traceback.format_exc())
            raise Exception(traceback.format_exc())
    return wrapped_function

# ============================================================
# Info-level logging decorator
# ============================================================


def info_logging(func, path=""):
    """Set up info-level logging decorator"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        log_info = BaseLogger(
            logname='info',
            level='info',
            logfile=path + 'info_log.log')
        log_info.logger_.info(
            "function '" + func.__name__ +
            "' was called with args=" +
            str(args) + " and kwargs=" + str(kwargs))
        return func(*args, **kwargs)
    return wrapped_function
