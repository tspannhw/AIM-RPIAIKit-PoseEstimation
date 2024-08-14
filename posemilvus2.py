import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
from datetime import datetime
import uuid
import glob
import torch
from torchvision import transforms
from PIL import Image
import timm
import requests
import json
import os
import boto3
from botocore.client import Config
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import MilvusClient
from pymilvus import MilvusClient
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)

# -----------------------------------------------------------------------------

DIMENSION = 512
MILVUS_URL = "https://someid.serverless.gcp-us-west1.cloud.zilliz.com"
COLLECTION_NAME = "poseestimation"
TOKEN = os.environ["ZILLIZ_TOKEN"]
PATH = "/opt/demo/images"
time_list = [ 0, 5, 10, 20, 30, 40, 50, 59 ]
FREEIMAGE_KEY = os.environ["FREEIMAGE_KEY"]
FREEIMAGE_URL = url = 'https://freeimage.host/api/1/upload?key=' + str(FREEIMAGE_KEY) + '&action=upload'
S3_URL = 'http://123234234:9000'


# -----------------------------------------------------------------------------
# Connect to Milvus

# Cloud Server
milvus_client = MilvusClient( uri=MILVUS_URL, token=TOKEN )

# -----------------------------------------------------------------------------------------------
# Slack client
# -----------------------------------------------------------------------------------------------

slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

# -----------------------------------------------------------------------------------------------
# Create Milvus collection which includes the id, filepath of the image, and image embedding
# -----------------------------------------------------------------------------------------------

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='label', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='lefteye', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='righteye', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='nose', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='leftear', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='rightear', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='leftshoulder', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='rightshoulder', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='leftelbow', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='rightelbow', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='leftwrist', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='rightwrist', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='lefthip', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='righthip', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='leftknee', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='rightknee', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='leftankle', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='rightankle', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='confidence', dtype=DataType.FLOAT),
    FieldSchema(name='width', dtype=DataType.VARCHAR, max_length=8),
    FieldSchema(name='height', dtype=DataType.VARCHAR, max_length=8),
    FieldSchema(name='size', dtype=DataType.VARCHAR, max_length=12),
    FieldSchema(name='ogfilename', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='sizeformatted', dtype=DataType.VARCHAR, max_length=15),
    FieldSchema(name='filename', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='url', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='mimetype', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

schema = CollectionSchema(fields=fields)

if not milvus_client.has_collection(collection_name=COLLECTION_NAME):
    milvus_client.create_collection(COLLECTION_NAME, DIMENSION, schema=schema, metric_type="COSINE", auto_id=True)
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name = "vector", metric_type="COSINE")
    milvus_client.create_index(COLLECTION_NAME, index_params)

# -----------------------------------------------------------------------------------------------
# Milvus Feature Extractor
# -----------------------------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()

extractor = FeatureExtractor("resnet34")

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    lefteye = ""
    righteye = ""
    label = ""
    nose = ""
    left_ear = ""
    right_ear = ""
    left_shoulder = ""
    right_shoulder = ""
    left_elbow = ""
    right_elbow = ""
    left_wrist = ""
    right_wrist = ""
    left_hip = ""
    right_hip = ""
    left_knee = ""
    right_knee = ""
    left_ankle = ""
    right_ankle = ""
    file = ""
    resp = ""
    jsonresults = ""
    pwidth = ""
    pheight = ""
    psize = ""
    pogfilename = ""
    psizeformatted = ""
    pfilename = ""
    purl = ""
    pmime = ""

    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = "" # f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print += (f"Detection: {label} {confidence:.2f} ")
            # Pose estimation landmarks from detection (if available)
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                left_eye = points[1]  # assuming 1 is the index for the left eye
                right_eye = points[2]  # assuming 2 is the index for the right eye

                try:
                    nose = points[0]
                    left_ear = points[3]
                    right_ear = points[4]
                    left_shoulder = points[5]
                    right_shoulder = points[6]
                    left_elbow = points[7]
                    right_elbow = points[8]
                    left_wrist = points[9]
                    right_wrist = points[10]
                    left_hip = points[11]
                    right_hip = points[12]
                    left_knee = points[13]
                    right_knee = points[14]
                    left_ankle = points[15]
                    right_ankle = points[16]
                except Exception as e:
                    print("Body points issues error:", e)

                # The landmarks are normalized to the bounding box, we also need to convert them to the frame size
                left_eye_x = int((left_eye.x() * bbox.width() + bbox.xmin()) * width)
                left_eye_y = int((left_eye.y() * bbox.height() + bbox.ymin()) * height)
                right_eye_x = int((right_eye.x() * bbox.width() + bbox.xmin()) * width)
                right_eye_y = int((right_eye.y() * bbox.height() + bbox.ymin()) * height)
                string_to_print += (f" Left eye: x: {left_eye_x:.2f} y: {left_eye_y:.2f} Right eye: x: {right_eye_x:.2f} y: {right_eye_y:.2f}")
                if user_data.use_frame:
                    # Add markers to the frame to show eye landmarks
                    cv2.circle(frame, (left_eye_x, left_eye_y), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (right_eye_x, right_eye_y), 5, (0, 255, 0), -1)

    if user_data.use_frame:
        # Convert the frame to BGR
        framesave = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(framesave)
        # maybe reuse frame name maybe use framesave
        time_now = datetime.now()
        current_time = int(time_now.strftime("%S"))

        if current_time in time_list and len(label) > 4:
            # -----------------------------------------------------------------------------
            # Save Image
            strfilename = PATH + "/personpose.jpg"
            cv2.imwrite(strfilename, framesave)

            lefteye = (f"x: {left_eye_x:.2f} y: {left_eye_y:.2f}")
            righteye = (f"x: {right_eye_x:.2f} y: {right_eye_y:.2f}")

            left_ear_x = int((left_ear.x() * bbox.width() + bbox.xmin()) * width)
            left_ear_y = int((left_ear.y() * bbox.height() + bbox.ymin()) * height)
            right_ear_x = int((right_ear.x() * bbox.width() + bbox.xmin()) * width)
            right_ear_y = int((right_ear.y() * bbox.height() + bbox.ymin()) * height)

            left_ear = (f"x: {left_ear_x:.2f} y: {left_ear_y:.2f}")
            right_ear = (f"x: {right_ear_x:.2f} y: {right_ear_y:.2f}")

            nose_x = int((nose.x() * bbox.width() + bbox.xmin()) * width)
            nose_y = int((nose.y() * bbox.height() + bbox.ymin()) * height)

            nose = (f"x: {nose_x:.2f} y: {nose_y:.2f}")

            left_shoulder_x = int((left_shoulder.x() * bbox.width() + bbox.xmin()) * width)
            left_shoulder_y = int((left_shoulder.y() * bbox.height() + bbox.ymin()) * height)
            right_shoulder_x = int((right_shoulder.x() * bbox.width() + bbox.xmin()) * width)
            right_shoulder_y = int((right_shoulder.y() * bbox.height() + bbox.ymin()) * height)
            left_shoulder = (f"x: {left_shoulder_x:.2f} y: {left_shoulder_y:.2f}")
            right_shoulder = (f"x: {right_shoulder_x:.2f} y: {right_shoulder_y:.2f}")

            left_elbow_x = int((left_elbow.x() * bbox.width() + bbox.xmin()) * width)
            left_elbow_y = int((left_elbow.y() * bbox.height() + bbox.ymin()) * height)
            right_elbow_x = int((right_elbow.x() * bbox.width() + bbox.xmin()) * width)
            right_elbow_y = int((right_elbow.y() * bbox.height() + bbox.ymin()) * height)
            left_elbow = (f"x: {left_elbow_x:.2f} y: {left_elbow_y:.2f}")
            right_elbow = (f"x: {right_elbow_x:.2f} y: {right_elbow_y:.2f}")

            left_wrist_x = int((left_wrist.x() * bbox.width() + bbox.xmin()) * width)
            left_wrist_y = int((left_wrist.y() * bbox.height() + bbox.ymin()) * height)
            right_wrist_x = int((right_wrist.x() * bbox.width() + bbox.xmin()) * width)
            right_wrist_y = int((right_wrist.y() * bbox.height() + bbox.ymin()) * height)
            left_wrist = (f"x: {left_wrist_x:.2f} y: {left_wrist_y:.2f}")
            right_wrist = (f"x: {right_wrist_x:.2f} y: {right_wrist_y:.2f}")

            left_hip_x = int((left_hip.x() * bbox.width() + bbox.xmin()) * width)
            left_hip_y = int((left_hip.y() * bbox.height() + bbox.ymin()) * height)
            right_hip_x = int((right_hip.x() * bbox.width() + bbox.xmin()) * width)
            right_hip_y = int((right_hip.y() * bbox.height() + bbox.ymin()) * height)
            left_hip = (f"x: {left_hip_x:.2f} y: {left_hip_y:.2f}")
            right_hip = (f"x: {right_hip_x:.2f} y: {right_hip_y:.2f}")

            left_knee_x = int((left_knee.x() * bbox.width() + bbox.xmin()) * width)
            left_knee_y = int((left_knee.y() * bbox.height() + bbox.ymin()) * height)
            right_knee_x = int((right_knee.x() * bbox.width() + bbox.xmin()) * width)
            right_knee_y = int((right_knee.y() * bbox.height() + bbox.ymin()) * height)
            left_knee = (f"x: {left_knee_x:.2f} y: {left_knee_y:.2f}")
            right_knee = (f"x: {right_knee_x:.2f} y: {right_knee_y:.2f}")

            left_ankle_x = int((left_ankle.x() * bbox.width() + bbox.xmin()) * width)
            left_ankle_y = int((left_ankle.y() * bbox.height() + bbox.ymin()) * height)
            right_ankle_x = int((right_ankle.x() * bbox.width() + bbox.xmin()) * width)
            right_ankle_y = int((right_ankle.y() * bbox.height() + bbox.ymin()) * height)
            left_ankle = (f"x: {left_ankle_x:.2f} y: {left_ankle_y:.2f}")
            right_ankle = (f"x: {right_ankle_x:.2f} y: {right_ankle_y:.2f}")

            # -----------------------------------------------------------------------------
            # Slack
            try:
                response = client.chat_postMessage(
                    channel="C06NE1FU6SE",
                    text=(f"Detection: {label} {confidence:.2f}")
                )
            except SlackApiError as e:
                # You will get a SlackApiError if "ok" is False
                assert e.response["error"]

            try:
                response = client.chat_postMessage(
                    channel="C06NE1FU6SE",
                    text=(f" Left eye: x: {left_eye_x:.2f} y: {left_eye_y:.2f} Right eye: x: {right_eye_x:.2f} y: {right_eye_y:.2f}\n")
                )
            except SlackApiError as e:
                # You will get a SlackApiError if "ok" is False
                assert e.response["error"]

            try:
                file = {'source': open(strfilename, 'rb')}
                resp = requests.post(url=FREEIMAGE_URL, files=file)
                jsonresults = resp.json()
                pwidth = str(jsonresults["image"]["width"])
                pheight = str(jsonresults["image"]["height"])
                psize = str(jsonresults["image"]["size"])
                pogfilename = str(jsonresults["image"]["original_filename"])
                psizeformatted = str(jsonresults["image"]["size_formatted"])
                pfilename = str(jsonresults["image"]["filename"])
                purl = str(jsonresults["image"]["url"])
                pmime = str(jsonresults["image"]["image"]["mime"])
            except Exception as e:
                print("An error:", e)

            try:
                response = client.files_upload_v2(
                    channel="C06NE1FU6SE",
                    file=strfilename,
                    title=label,
                    initial_comment="Live Camera image ",
                )
            except SlackApiError as e:
                assert e.response["error"]
            # Slack

            # -----------------------------------------------------------------------------
            # Milvus insert
            try:
                imageembedding = extractor(strfilename)
                milvus_client.insert( COLLECTION_NAME, {"label": str(label),
                    "lefteye": str(lefteye), "righteye": str(righteye), 
                    "nose": str(nose),"leftear": str(left_ear), "rightear": str(right_ear), 
                    "leftshoulder": str(left_shoulder),"rightshoulder": str(right_shoulder), 
                    "leftelbow": str(left_elbow), "rightelbow": str(right_elbow),
                    "leftwrist": str(left_wrist), "rightwrist": str(right_wrist), "lefthip": str(left_hip),
                    "righthip": str(right_hip), "leftknee": str(left_knee), "rightknee": str(right_knee),
                    "leftankle": str(left_ankle), "rightankle": str(right_ankle),  
                    "confidence": float(confidence),"width": str(pwidth), "height": str(pheight), 
                    "size": str(psize), "ogfilename": pogfilename,"sizeformatted": str(psizeformatted), 
                    "filename": str(pfilename), 
                    "url": str(purl), "mimetype": str(pmime),"vector": imageembedding})
            except Exception as e:
                print("An error:", e)

            try: 
                randomfilename = '{0}{1}.jpg'.format(label,uuid.uuid4())
                s3 = boto3.resource('s3',
                endpoint_url=S3_URL,
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin',
                config=Config(signature_version='s3v4'))

                s3.Bucket('images').upload_file(strfilename,randomfilename)

                s3path = "images/" + randomfilename
            except Exception as e:
                print("An error:", e)
            # -----------------------------------------------------------------------------


    print(string_to_print)
    return Gst.PadProbeReturn.OK


# This function can be used to get the COCO keypoints coorespondence map
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
        'nose': 1,
        'left_eye': 2,
        'right_eye': 3,
        'left_ear': 4,
        'right_ear': 5,
        'left_shoulder': 6,
        'right_shoulder': 7,
        'left_elbow': 8,
        'right_elbow': 9,
        'left_wrist': 10,
        'right_wrist': 11,
        'left_hip': 12,
        'right_hip': 13,
        'left_knee': 14,
        'right_knee': 15,
        'left_ankle': 16,
        'right_ankle': 17,
    }

    return keypoints
#-----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class

class GStreamerPoseEstimationApp(GStreamerApp):
    def __init__(self, args, user_data):
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolov8pose_post.so')
        self.post_function_name = "filter"
        self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_pose_h8l_pi.hef')
        self.app_callback = app_callback

        # Set the process title
        setproctitle.setproctitle("Hailo Pose Estimation with Milvus")

        self.create_pipeline()

    def get_pipeline_string(self):
        if (self.source_type == "rpi"):
            source_element = f"libcamerasrc name=src_0 auto-focus-mode=2 ! "
            source_element += f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
            source_element += QUEUE("queue_src_scale")
            source_element += f"videoscale ! "
            source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "

        elif (self.source_type == "usb"):
            source_element = f"v4l2src device={self.video_source} name=src_0 ! "
            source_element += f"video/x-raw, width=640, height=480, framerate=30/1 ! "
        else:
            source_element = f"filesrc location={self.video_source} name=src_0 ! "
            source_element += QUEUE("queue_dec264")
            source_element += f" qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
            source_element += f" video/x-raw,format=I420 ! "
        source_element += QUEUE("queue_scale")
        source_element += f"videoscale n-threads=2 ! "
        source_element += QUEUE("queue_src_convert")
        source_element += f"videoconvert n-threads=3 name=src_convert qos=false ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "


        pipeline_string = "hailomuxer name=hmux "
        pipeline_string += source_element
        pipeline_string += "tee name=t ! "
        pipeline_string += QUEUE("bypass_queue", max_size_buffers=20) + "hmux.sink_0 "
        pipeline_string += "t. ! " + QUEUE("queue_hailonet")
        pipeline_string += "videoconvert n-threads=3 ! "
        pipeline_string += f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} force-writable=true ! "
        pipeline_string += QUEUE("queue_hailofilter")
        pipeline_string += f"hailofilter function-name={self.post_function_name} so-path={self.default_postprocess_so} qos=false ! "
        pipeline_string += QUEUE("queue_hmuc") + " hmux.sink_1 "
        pipeline_string += "hmux. ! " + QUEUE("queue_hailo_python")
        pipeline_string += QUEUE("queue_user_callback")
        pipeline_string += f"identity name=identity_callback ! "
        pipeline_string += QUEUE("queue_hailooverlay")
        pipeline_string += f"hailooverlay ! "
        pipeline_string += QUEUE("queue_videoconvert")
        pipeline_string += f"videoconvert n-threads=3 qos=false ! "
        pipeline_string += QUEUE("queue_hailo_display")
        pipeline_string += f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    parser = get_default_parser()
    args = parser.parse_args()
    app = GStreamerPoseEstimationApp(args, user_data)
    app.run()
