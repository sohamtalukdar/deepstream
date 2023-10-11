import mysql.connector
import json
import config as tc
def DataToDB(predicted_name,DataToSend,TimeStamp):

    mydb = mysql.connector.connect(
    host= tc.host,
    user= tc.user,
    password= tc.password,
    port = tc.port,
    database= tc.database
    )

    mycursor = mydb.cursor()

    sql = "INSERT INTO Recog (PersonID,RecogPoint,TimeStamp) VALUES (%s,%s, %s)"
    val = (predicted_name,DataToSend,TimeStamp)
 

    mycursor.execute(sql,val)

    mydb.commit()

    print(mycursor.rowcount, "record inserted.")

# DataToDB()

#Data = '{"CameraID": "rtsp://ranu:Bharat1947@192.168.1.12:554/Profile2/media.smp", "CameraName": "Office Gate", "GateStatus": "closed"}'
#DataToDB(Data)
