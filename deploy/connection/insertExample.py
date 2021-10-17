import mysqlconnection
import pymysql.cursors
from datetime import date, datetime

connection = mysqlconnection.getConnection()

try :
    cursor = connection.cursor()
     
     
    sql =  "Insert into check_logs (MSSV, time) " \
         + " values (%s, %s)"

    # Thực thi sql và truyền 3 tham số
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(dt_string)
    cursor.execute(sql,(2001160174,now))
     
    connection.commit()
 
finally: 
    connection.close()