
import psycopg2
import yaml
from pathlib import Path
import logging

# from utils import read_params


class DB():
    """
    A simple wrapper class for connecting to the PostgreSQL database.

    Takes no arguments. Relies on having connection information in
    `./gzoo.yaml`.
    """

    def __init__(self, params: dict):
        """
        Reads the connection parameters, makes the connection and a cursor
        """

        # params = read_params()

        inf = f"dbname={params['dbname']} user={params['username']}"
        inf += f"  host='{params['host']}' password={params['password']}"
        self.connection = psycopg2.connect(inf)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()

        # self.logger = logger

    # @staticmethod
    # def read_params():
    #     """
    #     Needs the gzoo.yaml parameter file to be in the current directory
    #
    #     :return: parameter dictionary
    #     """
    #
    #     filename = '../gzoo.yaml'
    #     with open(filename) as file:
    #         params = yaml.full_load(file)
    #         return params

    def get_cursor(self):
        "A simple getter method"

        return self.cursor

    def execute(self, sql):
        """
        A wrapper around cursor.execute() which adds error logging

        :param sql:
        :return: None. If a result is expected, use self.cursor to retrieve it.
        """

        try:
            self.cursor.execute(sql)
        except Exception as e:
            # self.logger.write_log(f"{e}  when executing {sql}")
            logging.error(f"{e}  when executing {sql}")
            raise e

    def run_select(self, query):
        """
        Runs a SQL SELECT query

        Returns results in Python list format
        (not numpy, which would need a dtype list)
        """

        self.execute(query)
        return self.cursor.fetchall()

    def run_count(self, table):
        """
        Get number of records in table

        :param table: postgres table name in current database
        :return: count (int)
        """

        self.execute(f"select count(*) from {table}")
        return self.cursor.fetchone()

    def run_admin(self, query):
        """
        For CREATE, ALTER, DROP, etc

        :param query: some valid SQL (str)
        :return: None
        """

        self.execute(query)