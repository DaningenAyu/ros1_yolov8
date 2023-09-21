#!/usr/bin/env python3
#-*- coding: utf-8 -*

import cv2
import rospy
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO
from yolov8_pkg.msg import BoundingBox


class YOLOv8_detect():
    #初期設定
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        #self.model = YOLO("paper_bag_last.pt")
        rospy.init_node('detectYOLOv8n',anonymous=True)#ノードを立てる
        self.bridge = CvBridge()
        #Realsenseのカラー画像の情報取得(topicを取得)
        #rospy.Subscriber('/camera/color/image_raw',Image,self.img_listener)
        rospy.Subscriber('/usb_cam/image_raw',Image,self.img_listener)
        self.pub = rospy.Publisher('yolov8/BoundingBox',BoundingBox,queue_size=10)


    #カラー画像を処理
    def img_listener(self,data):
        try:
            #topicで送られてきたROS_Image型の画像をOpenCVで扱える型に変換
            self.img = self.bridge.imgmsg_to_cv2(data,"bgr8")

            results = self.model(self.img)
            imageWidth = results[0].orig_shape[0]
            imageHeight = results[0].orig_shape[1]
            names = results[0].names
            classes = results[0].boxes.cls
            boxes = results[0].boxes
            self.annotatedFrame = results[0].plot()
            

            for box,cls in zip(boxes,classes):
                box_conf = float(box.conf[0])
                name = names[int(cls)]
                box_id = int(cls)
                x1,y1,x2,y2 = [int(i) for i in box.xyxy[0]]
                #middle_x = x1 + ((x2 - x1) // 2)
                #self.annotatedFrame = cv2.line(self.annotatedFrame,pt1=(middle_x,y1),pt2=(middle_x,y2),color=(0,255,0),thickness=2,lineType=cv2.LINE_4,shift=0)
                print(f"--- \nprobability: {box_conf} \nxmin: {x1} \nymin: {y1} \nxmax: {x2} \nymax: {y2} \nid: {box_id} \nClass: {name}")
                self.pub.publish(box_conf,x1,y1,x2,y2,box_id,name)
            self.show()
        except CvBridgeError as e:
            print(e)

    #画像を表示させる
    def show(self):
        cv2.imshow('View',self.annotatedFrame)
        cv2.waitKey(1)

#main文
if __name__ == '__main__':
    try:
        YOLOv8_detect()    #classを呼び出す
        rospy.spin()    #処理を継続させる
    except rospy.ROSInitException:
        print('Shutting down')
        cv2.destroyAllWindows()
