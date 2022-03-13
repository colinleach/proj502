import pyarrow.parquet as pq
# import pandas as pd # pyarrow handles this
import csv
from io import StringIO
from sqlalchemy import create_engine # not my preference, pandas insists
from pathlib import Path
import logging

from utils import read_params
from db import DB


class ParquetToDB:

    def __init__(self, key='decals_parquet'):

        logging.basicConfig(
            filename='gzoo.log',
            format='%(asctime)s %(levelname)s:%(message)s',
            level=logging.INFO
        )

        # find and read parquet file
        self.params = read_params()
        pq_file = Path(self.params['dataroot']) / self.params[key]
        self.df = pq.read_table(pq_file).to_pandas()

    @staticmethod
    def psql_insert_copy(table, conn, keys, data_iter):
        # taken from https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ', '.join(f'"{k}"' for k in keys)
            if table.schema:
                table_name = 'f{table.schema}.{table.name}'
            else:
                table_name = table.name

            sql = f'COPY {table_name} ({columns}) FROM STDIN WITH CSV'
            cur.copy_expert(sql=sql, file=s_buf)

    def create_table(self, table_name: str = 'decalsdata'):
        """

        :param table_name: for new table in postres, so lower case, limited character set
        """
        user = self.params['username']
        pword = self.params['password']
        host = self.params['host']
        dbname = self.params['dbname']
        engine = create_engine(f'postgresql://{user}:{pword}@{host}:5432/{dbname}')
        self.df.to_sql(table_name,
                       engine,
                       method=self.psql_insert_copy,
                       if_exists='replace',
                       index=False)
        rows = self.add_pkey()
        logging.info(f"table {table_name}, created, added {rows} rows")

    def add_pkey(self):
        """

        """
        db = DB()
        sql = """
            ALTER TABLE IF EXISTS public.decalsdata
                ADD PRIMARY KEY (iauname);"""
        db.run_admin(sql)
        return db.run_count('decalsdata')


if __name__ == "__main__":
    p2d = ParquetToDB()
    p2d.create_table()

