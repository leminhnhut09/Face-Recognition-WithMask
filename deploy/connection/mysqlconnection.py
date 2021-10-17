import pymysql.cursors
# Hàm trả về một connection.
def getConnection():
     
    # Bạn có thể thay đổi các thông số kết nối.
    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password='',                      
                                 db='insightface',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection