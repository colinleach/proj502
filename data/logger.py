import os
import sys
import argparse
from time import strftime
from pathlib import Path


class Logger:
    def __init__(self, def_log: Path):
        self.def_log = def_log
        self.logfile = None
        self.set_log()

    def set_log(self):
        """

        :return:
        """
        parser = argparse.ArgumentParser(description='set logfile')
        parser.add_argument('-l', '--logfile',
                            dest='logfile',
                            default=self.def_log,
                            help='path to log file')
        args = parser.parse_args()

        if os.path.isfile(args.logfile):
            self.logfile = args.logfile
        else:
            msg = "logfile %s not found" % (args.logfile,)
            print(msg)
            sys.exit("\n *** Bad path(s) specified, unable to continue \n")

    def write_log(self, text: str):
        """

        :param text:
        """
        toWrite = strftime("%Y-%m-%d %H:%M:%S") + " :  " + text + '\n'
        f = open(self.logfile, 'a')
        f.write(toWrite)
        f.close()
