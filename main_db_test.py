import pyodbc

print(pyodbc.drivers())

conn = pyodbc.connect(
        'Driver={ODBC Driver 17 for SQL Server};'
        'Server=ics-lgs-prod-sqlsvr.database.windows.net;'
        'Database=ICS-LGS-PROD-DB;'
        'Uid=ICS_ViewUser;'
        'Pwd=View-37B8BfP4FcQ8;'
        'Trusted_Connection=no;'
)

cursor = conn.cursor()
for table_name in cursor.tables(tableType='TABLE'):
    print(table_name)

for table_name in cursor.tables(tableType='View'):
    print(table_name)
