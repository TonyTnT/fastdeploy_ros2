import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from ament_index_python.packages import get_package_share_directory
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import fastdeploy as fd
import message_filters
from rclpy.qos import ReliabilityPolicy, QoSProfile

class InferNode(Node):
    def __init__(self):
        super().__init__('yolo_infer_node')
        self.logger = self.get_logger()

        self.declare_parameter("model_file", "/home/firefly/FastDeploy/examples/vision/detection/rkyolo/python/yolov7-tiny/yolov7-tiny_tk2_RK3588_i8.rknn", ParameterDescriptor(
            name="model_file", description="Path of rknn YOLOv7 model"))
        self.declare_parameter("image_topic", "/image_raw", ParameterDescriptor(
            name="image_topic", description="输入图像话题，默认：/image_raw"))
        self.declare_parameter("pub_result_image", True, ParameterDescriptor(
            name="pub_result_img", description="是否发布识别结果图像，默认：False"))
        self.declare_parameter("camera_qos", "best_effort", ParameterDescriptor(
            name="camera_qos", description="camera using qos, best_effort or reliable"))

        # 获取ROS参数的值
        self.model_file = self.get_parameter('model_file').value
        self.image_topic = self.get_parameter('image_topic').value
        self.pub_result_image = self.get_parameter('pub_result_image').value
        camera_qos = self.get_parameter('camera_qos').value


        # 配置runtime，加载模型
        runtime_option = fd.RuntimeOption()
        runtime_option.use_rknpu2()

        self.model = fd.vision.detection.RKYOLOV7(
            self.model_file,
            runtime_option=runtime_option,
            model_format=fd.ModelFormat.RKNN)

        self.qos = QoSProfile(depth=5)
        if camera_qos == 'best_effort':
            self.qos.reliability = ReliabilityPolicy.BEST_EFFORT
        elif camera_qos == 'reliable':
            self.qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            self.logger.error('Invalid value for camera_qos parameter')
            return
        

        self.bridge = CvBridge()
        if self.pub_result_image:
            self.result_img_pub = self.create_publisher(
                Image, "yolo_result_image", 10)

        # http://wiki.ros.org/message_filters#Example_.28Python.29



    def camera_callback(self, rgb_msg, depth_msg):

        image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        # depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="rgb8")

        result = self.model.predict(image)
        # 可视化结果
        vis_im = fd.vision.vis_detection(image, result, score_threshold=0.1)

        result_img_msg = self.bridge.cv2_to_imgmsg(vis_im, encoding="rgb8",header=rgb_msg.header)
        self.result_img_pub.publish(result_img_msg)

        return 

    def depth_callback(self):
        return        

def main(args=None):
    rclpy.init(args=args)
    infer_node = InferNode()



    image_sub = message_filters.Subscriber(infer_node, Image, '/femto_mega/color/image_raw', qos_profile=infer_node.qos)
    depth_sub = message_filters.Subscriber(infer_node, Image, '/femto_mega/depth/image_raw', qos_profile=infer_node.qos)
    time_synchronizer = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.1, allow_headerless=True)
    time_synchronizer.registerCallback(infer_node.camera_callback)

    rclpy.spin(infer_node)
    infer_node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()