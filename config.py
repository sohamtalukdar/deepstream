# Database configs goes here

host = "localhost"
user = "root"
port = "3306"
password = "Bharat1947"
database = "face_test"


# Building ID Goes here
GateID = {'0':("cam1","192.168.177.76"),
          '1': ("cam2","192.168.1.12")
         }
# camera configs goes here
Video = "file:///home/asdael/Downloads/Vishwajeet/Metro-FRS-DS-Video/new3.mp4"
Cam1 =  "rtsp://ranu:Bharat1947@192.168.177.76:554/Profile2/media.smp"
#Cam2 = "rtsp://ranu:Bharat1947@192.168.1.12:554/Profile2/media.smp"
l1 = [Video,Cam1]
print(l1)
#Cam2 = "rtsp://ranu:Bharat1947@192.168.1.12:554/Profile2/media.smp"
