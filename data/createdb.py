

import pandas as pd
from pathlib import Path

from logger import Logger
from db import DB


class CreateDB:

    def __init__(self):
        self.logger = Logger(Path('gzoo.log'))
        self.pg = DB(self.logger)
        params = self.pg.read_params()
        self.mapping = params['mapping']
        self.datafile = params['datafile']

    def create_empty_table(self):
        """

        :return: None
        """

        mapping = pd.read_csv(self.mapping, sep='\t')
        self.mapping = mapping.rename(columns={"name": "gzname"})

        fieldlist = [f'{rec.gzname} {self.translate_type(rec.type, rec.length)}'
                     for index, rec in self.mapping.iterrows()]
        fields = ',\n'.join(fieldlist)

        sql = f"""
        CREATE TABLE public.gz2data
        (
            {fields},
            PRIMARY KEY (dr7objid)
        );
        
        ALTER TABLE IF EXISTS public.gz2data
            OWNER to python;
            """

        self.pg.run_admin("DROP TABLE IF EXISTS public.gz2data")
        self.pg.run_admin(sql)
        self.logger.write_log('Table gz2data created')

    @staticmethod
    def translate_type(gztype, length):
        if gztype == 'bigint':
            return 'bigint'
        if gztype == 'real':
            return 'real'
        if gztype == 'int':
            return 'integer'
        if gztype == 'float':
            return 'double precision'
        if gztype == 'real':
            return 'varchar'
        if gztype == 'varchar':
            return f'character varying({length})'
        return f'***** {gztype} not recognized'

    def load_data(self):
        """
        Uses a postgres-specific query to add records from a (large) CSV file.

        :return:
        """

        names = ','.join(list(self.mapping.gzname))

        uploadsql = f"""COPY gz2data({names})
            FROM '{self.datafile}'
            DELIMITER ','
            CSV HEADER"""
        self.pg.run_admin(uploadsql)
        rows = self.pg.run_count('gz2data')
        self.logger.write_log(f'Added {rows[0]} records to gz2data')

def main() -> None:
    """

    :return:
    """

    cd = CreateDB()
    cd.create_empty_table()
    cd.load_data()


if __name__ == "__main__":
    main()