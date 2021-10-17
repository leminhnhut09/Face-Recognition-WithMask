import pymysql.cursors  
 
# Kết nối vào database.
connection = pymysql.connect(host='127.0.0.1',
                             user='root',
                             password='',                             
                             db='insightface',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
 
print ("connect successful!!")
 
try:
 
    with connection.cursor() as cursor:
       
        # SQL 
        sql = "SELECT * FROM diemdanhsv"
         
        # Thực thi câu lệnh truy vấn (Execute Query).
        cursor.execute(sql)
         
        print("cursor.description: ", cursor.description)
 
        print()
 
        for row in cursor:
            print(row)
             
finally:
    # Đóng kết nối (Close connection).       
    connection.close()