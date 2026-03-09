import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import os

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()
        model_path = '/ros2_ws/yolov8n.onnx'
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found at {model_path}')
            return
        self.session = ort.InferenceSession(model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f'YOLOv8n loaded. Provider: {self.session.get_providers()[0]}')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Image, '/yolo/detection_image', 10)

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(img)[np.newaxis]

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]
        out = self.session.run(None, {self.input_name: self.preprocess(frame)})
        for det in out[0][0].T:
            cx, cy, bw, bh = det[:4]
            conf = float(det[4:].max())
            cls = int(det[4:].argmax())
            if conf > 0.5:
                x1=int((cx-bw/2)*w/640); y1=int((cy-bh/2)*h/640)
                x2=int((cx+bw/2)*w/640); y2=int((cy+bh/2)*h/640)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f'{cls} {conf:.2f}',(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        self.pub.publish(self.bridge.cv2_to_imgmsg(frame,'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(YoloNode())
    rclpy.shutdown()
