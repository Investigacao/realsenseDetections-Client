
from ast import If
from calendar import c
from math import *

import socket
import struct
import numpy as np
import time
import cv2
import pyrealsense2 as rs 
import random
import argparse
import copy

from threading import *
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Boxes, RotatedBoxes
from detectron2 import model_zoo

from detectron2.data import MetadataCatalog

import torch, torchvision
import subprocess

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pkg_resources

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from darwin.torch.utils import detectron2_register_dataset

import asyncio
import websocket
import json


# Resolution of camera streams
# RESOLUTION_X = 640  #640, 1280
# RESOLUTION_Y = 480   #360(BW:cannot work in this PC, min:480)  #480, 720

# Configuration for histogram for depth image
NUM_BINS = 500    #500 x depth_scale = e.g. 500x0.001m=50cm
MAX_RANGE = 10000  #10000xdepth_scale = e.g. 10000x0.001m=10m

AXES_SIZE = 10

# Set test score threshold
SCORE_THRESHOLD = 0.65  #vip-The smaller, the faster.

# TRESHOLD para a frente do robo
THRESHOLD_FRENTE = 0.035
#TRHESHOLD para a altura do robo
THRESHOLD_ALTURA = 0.05



class Predictor(DefaultPredictor):
    def __init__(self):
        self.config = self.setup_predictor_config()
        super().__init__(self.config)

    def create_outputs(self, color_image):
        self.outputs = self(color_image)

    def setup_predictor_config(self):
        """
        Setup config and return predictor. See config/defaults.py for more options
        """

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

        
        dataset_id = "pedro2team/oranges-apples-vases:oranges-apples-vases1.0"
        dataset_train = detectron2_register_dataset(dataset_id, partition='train', split_type='stratified')
        cfg.DATASETS.TRAIN = (dataset_train)


        # This determines the resizing of the image. At 0, resizing is disabled.
        cfg.INPUT.MIN_SIZE_TEST = 0

        return cfg

    def format_results(self, class_names, video_file):
        """
        Format results so they can be used by overlay_instances function
        """
        predictions = self.outputs['instances']
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        labels = None 
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        masks = predictions.pred_masks.cpu().numpy()
        if video_file:
            masks = [GenericMask(x, 720, 1280) for x in masks]
        else:
            #masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
            masks = [GenericMask(x, 480, 640) for x in masks]
        boxes_list = boxes.tensor.tolist()
        scores_list = scores.tolist()
        class_list = classes.tolist()

        for i in range(len(scores_list)):
            boxes_list[i].append(scores_list[i])
            boxes_list[i].append(class_list[i])
        

        boxes_list = np.array(boxes_list)

        return (masks, boxes, boxes_list, labels, scores_list, class_list)    



class OptimizedVisualizer(Visualizer):
    """
    Detectron2's altered Visualizer class which converts boxes tensor to cpu
    """
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)
    
    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.cpu().numpy()
        else:
            return np.asarray(boxes)



class DetectedObject:
    """
    Each object corresponds to all objects detected during the instance segmentation
    phase. Associated trackers, distance, position and velocity are stored as attributes
    of the object.
    masks[i], boxes[i], labels[i], scores_list[i], class_list[i]
    """
    def __init__(self, mask, box, label, score, class_name):
        self.mask = mask
        self.box = box
        self.label = label
        self.score = score
        self.class_name = class_name


class Arrow3D(FancyArrowPatch):
    """
    Arrow used to demonstrate direction of travel for each object
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



def find_mask_centre(mask, color_image):
    """
    Finding centre of mask using moments
    """
    moments = cv2.moments(np.float32(mask))

    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    return cX, cY


def find_median_depth(mask_area, num_median, histg):
    """
    Iterate through all histogram bins and stop at the median value. This is the
    median depth of the mask.
    """
    
    median_counter = 0
    centre_depth = "0.00"
    for x in range(0, len(histg)):
        median_counter += histg[x][0]
        if median_counter >= num_median:
            # Half of histogram is iterated through,
            # Therefore this bin contains the median
            centre_depth = x / 50
            break 

    return float(centre_depth)

detect_vases = False
detect_fruits = False
shared_vases_detected = Event()
shared_fruits_detected = Event()


def on_open(ws):
    print("Connected to WebSocket server.")
    for topic in topics:
        message = {"topic": topic, "data": f"Realsense connected to {topic}"}
        ws.send(json.dumps(message))

def on_message(ws, message):
    print("on_message: ", message)
    global detect_vases
    global detect_fruits
    if "vase" in message.lower():
        print("Vase request received")
        detect_vases = True
        Thread(target=handle_thread_vase, args=(ws,)).start()
    elif "fruit" in message.lower():
        print("fruit request received")
        detect_fruits = True
        Thread(target=handle_thread_fruits, args=(ws,)).start()

def handle_thread_vase(ws):
    global shared_vases_detected
    print("Vase thread started")
    # while shared_vases_detected == False:
    #     # time.sleep(0.2)
    #     pass
    shared_vases_detected.wait()
    global message_to_send_mario
    global message_to_send_ruben
    print(f"Message to send to Mario: {message_to_send_mario}")
    # new_message_to_send_mario ={"Vase": message_to_send_mario["Vase"]}
    # print(f"New message to send to Mario: {new_message_to_send_mario}")
    # print(f"message to send ruben: {message_to_send_ruben}")
    # closest_vase = None
    # closest_distance = float('inf')
    # closest_perc = float('inf')

    # Find the highest percentage vase
    highest_perc_ruben = 0
    highest_perc_vase_ruben = None
    for obj in message_to_send_ruben['detected']:
        if obj['object'] == 'Vase':
            perc = obj['perc']
            perc_num = int(perc.strip('%'))
            if perc_num > highest_perc_ruben:
                highest_perc_ruben = perc_num
                highest_perc_vase_ruben = obj
            
            # perc = obj['perc']
            # if perc < closest_perc:
            #     closest_perc = perc
            #     closest_vase = obj
    
    # print(f"Closest vase: {closest_vase}")
    
    # Find the highest percentage vase
    highest_perc = 0
    highest_perc_vase = None
    for vase in message_to_send_mario['Vase']:
        perc = vase['perc']
        perc_num = int(perc.strip('%'))
        if perc_num > highest_perc:
            highest_perc = perc_num
            highest_perc_vase = vase

    new_message_to_send_mario = {'Fruits': message_to_send_mario['Fruits'], 'Vase': highest_perc_vase}

    print(f"\n\nNew message to send to Mario: {new_message_to_send_mario}\n\n")

    # closest_vase_ruben = dict(closest_vase)
    # closest_vase_ruben['x'] -= 0.2 

    # new_message_to_send_ruben = [d for d in message_to_send_ruben if d.get("data") == "vase"]
    send_message_topic(ws, "digital_twin", new_message_to_send_mario)
    send_message_topic(ws, "turtlebot", [highest_perc_vase_ruben])
    # send_message_topic(ws, "turtlebot", [closest_vase_ruben])
    shared_vases_detected.clear()

def handle_thread_fruits(ws):
    global shared_fruits_detected
    global message_to_send_mario
    global message_to_send_ruben
    shared_fruits_detected.wait()
    new_message_to_send_mario ={"Fruits": message_to_send_mario["Fruits"]}

    print(f"message to send ruben: {message_to_send_ruben}")

    # Create a set of the target objects
    target_objects = {"Orange", "Red_Apple", "Green_Apple"}
    new_message_to_send_ruben = [obj for obj in message_to_send_ruben["detected"] if obj["object"] in target_objects]
    # send_message_topic(ws, "digital_twin", new_message_to_send_mario)

    print(f"New message to send to ruben: {new_message_to_send_ruben}")

    send_message_topic(ws, "turtlebot", new_message_to_send_ruben)
    shared_fruits_detected.clear()


def on_close(ws):
    print("Disconnected from WebSocket server.")

def send_message_topics(ws, topics):
    for topic in topics:
        message = {"topic": topic, "data": f"Realsense connected to {topic}"}
        ws.send(json.dumps(message))

def send_message_topic(ws, topic, message_to_send):
    message = {"topic": topic, "data": message_to_send}
    ws.send(json.dumps(message))

def websocket_thread(ws):
    while True:
        on_message(ws, json.loads(ws.recv()))


if __name__ == "__main__":
    print("Waiting for capture initiation...")

    # Set up socket connection
    HOST = '192.168.1.20'
    PORT = 8889
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    
    #? Define the WebSocket URL and topics to subscribe to
    ws_url = "ws://172.22.21.135:3306/"
    topics = ["turtlebot", "digital_twin"]

    ws = websocket.WebSocket()
    ws.connect(ws_url)
    send_message_topics(ws, topics)

    Thread(target=websocket_thread, args=(ws,)).start()

    print("Waiting for message from server...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='type --file=file-name.bag to stream using file instead of webcam')
    args = parser.parse_args()

    # Initialise Detectron2 predictor
    predictor = Predictor()

    # Initialise video streams from D435


    speed_time_start = time.time()

    # command = ['ffmpeg',
    # '-y',
    # '-f', 'rawvideo',
    # '-vcodec', 'rawvideo',
    # '-pix_fmt', 'rgb24',
    # '-s', "{}x{}".format(640,480),
    # '-r', str(5),
    # '-i', '-',
    # '-c:v', 'libx264',
    # '-pix_fmt', 'yuv420p',
    # '-f', 'flv',
    # '-flvflags', 'no_duration_filesize',
    # 'rtmp://127.0.0.1/live/stream']
    # # 192.168.1.203:30439
    # p = subprocess.Popen(command, stdin=subprocess.PIPE)

    result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (640, 480))

    try:
        while True:
            message_to_send_mario = {"Fruits":{"Red_Apple":[], "Green_Apple":[], "Orange":[]}, "Vase":[]}
            message_to_send_ruben={"detected":[]}
            time_start = time.time()

            depth_size_data = sock.recv(struct.calcsize('I'))
            depth_size = struct.unpack('I', depth_size_data)[0]
            depth_data = b''
            while len(depth_data) < depth_size:
                depth_data += sock.recv(depth_size - len(depth_data))
            depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)


            # Receive color frame size and data
            color_size_data = sock.recv(struct.calcsize('I'))
            color_size = struct.unpack('I', color_size_data)[0]
            color_data = b''
            while len(color_data) < color_size:
                color_data += sock.recv(color_size - len(color_data))
            color_image = np.frombuffer(color_data, dtype=np.uint8).reshape(480, 640, 3)


            detected_objects = []

            t1 = time.time()

            camera_time = t1 - time_start

            video_file = False

            if args.file != None:
                video_file = True
            
            
            if video_file:
                predictor.create_outputs(color_image[:, :, ::-1])
                RESOLUTION_X = 1280
                RESOLUTION_Y = 720
            else:
                predictor.create_outputs(color_image)
                RESOLUTION_X = 640
                RESOLUTION_Y = 480
                
            outputs = predictor.outputs

            t2 = time.time()
            model_time = t2 - t1
            # print("Model took {:.2f} time".format(model_time))

            predictions = outputs['instances']
            

            if outputs['instances'].has('pred_masks'):
                num_masks = len(predictions.pred_masks)
            
            detectron_time = time.time()

            # Create a new Visualizer object from Detectron2 

            dataset_metadata = MetadataCatalog.get(predictor.config.DATASETS.TRAIN)

            v = OptimizedVisualizer(color_image[:, :, ::-1], metadata=dataset_metadata)

            metadata = v.metadata.get("thing_classes")

            
            masks, boxes, boxes_list, labels, scores_list, class_list = predictor.format_results(v.metadata.get("thing_classes"), video_file)



            # for i in range(len(labels)):
            #     percentages = labels[i].split()[1]
            #     labels[i] = f"orange {percentages}"
            

            for i in range(num_masks):
                try:
                    detected_obj = DetectedObject(masks[i], boxes[i], labels[i], scores_list[i], class_list[i])
                except:
                    print("Object doesn't meet all parameters")
                
                detected_objects.append(detected_obj)


            
            v.overlay_instances(
                masks=masks,
                boxes=boxes,
                labels=labels,
                keypoints=None,
                assigned_colors=None,
                alpha=0.3
            )
            
            speed_time_end = time.time()
            total_speed_time = speed_time_end - speed_time_start
            speed_time_start = time.time()

            R = 6378.1 
            # These values should be replaced with real coordinates
            latDrone = radians(39.73389)
            lonDrone = radians(-8.821944)



            for i in range(num_masks):
                """
                Converting depth image to a histogram with num bins of NUM_BINS 
                and depth range of (0 - MAX_RANGE millimeters)
                """
            
                mask_area = detected_objects[i].mask.area()
                num_median = floor(mask_area / 2)
                histg = cv2.calcHist([depth_image], [0], detected_objects[i].mask.mask, [NUM_BINS], [0, MAX_RANGE])



                # Uncomment this to use the debugging function
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # debug_plots(color_image, depth_image, masks[i].mask, histg, depth_colormap)
                centre_depth = find_median_depth(mask_area, num_median, histg)


                detected_objects[i].distance = centre_depth
                cX, cY = find_mask_centre(detected_objects[i].mask._mask, v.output)

                #? _Color_Camera
                HFOV = 69
                VFOV = 42

                CENTER_POINT_X = RESOLUTION_X / 2
                CENTER_POINT_Y = RESOLUTION_Y / 2

                # cx, cy -> mask center point

                #? Angulos da relacao ao centro da camera com o centro da mascara
                H_Angle = ((cX- CENTER_POINT_X)/CENTER_POINT_X)*(HFOV/2)
                V_Angle = ((cY - CENTER_POINT_Y)/CENTER_POINT_Y)*(VFOV/2)

                v.draw_circle((cX, cY), (0, 0, 0))

                #? detected_objects[i].distance = centre_depth -> profundidade media da mascara a camera
                
                #convert degrees to radians - em vez do 45 deve tar a direcao do drone
                direction = 45 + H_Angle
                if direction > 360:
                    direction = direction - 360
                elif direction < 0:
                    direction = direction + 360
                brng = radians(direction)

                #? Distancia em linha reta da camera para o objeto
                distanceToFruit = ((centre_depth/cos(radians(H_Angle)))**2 + (centre_depth*tan(radians(V_Angle)))**2)**0.5

                #? Distancia em linha reta da camera para o objeto com threshold da garra
                depthFromObjectToClawThreshold = round(centre_depth - THRESHOLD_FRENTE, 3)

                new_Distance_to_Claw = (((centre_depth - 3.5)/cos(radians(H_Angle)))**2 + (((centre_depth-3.5)*tan(radians(V_Angle)))+5)**2)**0.5


                #? Relative Coordinates calculation
                #* Calculo do Y (o quanto o braco tem de andar para a esquerda ou direita)
                #* (após multiplicar por -1) -> se objeto estiver a esquerda do centro da camera, o valor é positivo
                distancia_lateral = (tan(radians(H_Angle)) * centre_depth * -1 ) 

                # 0.046x^2 + 0.863x + 0.038 -> quadratic function to calculate the distance lateral
                # 0.8655x + 0.03829 -> linear function to calculate the distance lateral

                distancia_lateral = 0.046*(distancia_lateral)**2 + 0.863*(distancia_lateral) + 0.038 - 0.01


                # print("Distancia lateral: ", distancia_lateral)
                # distancia_lateral += 0.038

                # #By ever 0.02 of distance lateral after 0.06, remove 0.002 of distance lateral and additionally increase in 0.001 after 0.13
                # print(f"DISTANCIA LATERAL: {distancia_lateral}")
                # if distancia_lateral < 0.02:
                #     if distancia_lateral < 0.04:
                #         print("esteve dentro do IF")
                #         distancia_lateral += abs(floor(distancia_lateral / 0.02) * 0.0025)
                #     if distancia_lateral < -0.06:
                #         distancia_lateral += 0.0025
                #     if distancia_lateral < -0.096:
                #         distancia_lateral += 0.0025
                #     if distancia_lateral < -0.117:
                #         distancia_lateral += 0.0025
                #     else:
                #         distancia_lateral += abs(floor(distancia_lateral / 0.02) * 0.0017)
                #     print(f"DISTANCIA LATERAL DEPOIS: {distancia_lateral}")


                # if distancia_lateral > 0.04:
                #     if distancia_lateral > 0.13:
                #         distancia_lateral -= floor(distancia_lateral / 0.02) * 0.0023
                #     elif distancia_lateral > 0.15:
                #         distancia_lateral -= floor(distancia_lateral / 0.02) * 0.0025
                #     else:
                #         distancia_lateral -= floor(distancia_lateral / 0.02) * 0.001

                # if distancia_lateral > 0.06:
                #     distancia_lateral -= 0.002

                # print("Distancia lateral Depois: ", distancia_lateral)
                # if distancia_lateral > 0.07:
                #     distancia_lateral -= 0.005
                # if distancia_lateral > 0.01:
                #     distancia_lateral -= 0.006
                # if distancia_lateral > 0.012:
                #     distancia_lateral -= 0.002
                # if distancia_lateral > 0.0132:
                #     distancia_lateral -= 0.002
                    # distancia_lateral += 0.01
                    # pass

                #! Calculos para acertar o Y consoante a distancia do objeto ao centro da camera lateralmente
                # diferenca = abs(int(distancia_lateral /0.01))
                # variavel = False
                
                # if distancia_lateral < 0:
                #     if distancia_lateral > (-0.03):
                #         variavel = True
                #         # print("Distancia lateral Antes: ", distancia_lateral)
                #         distancia_lateral += 0.025
                #         # print("Distancia lateral Depois: ", distancia_lateral)
                #     # print("Distancia lateral_1: ", distancia_lateral)
                #     distancia_lateral = distancia_lateral + diferenca * 0.0035
                #     # print("Distancia lateral_2: ", distancia_lateral)
                # if distancia_lateral > 0:
                #     distancia_lateral = (distancia_lateral + diferenca * 0.0035)

                distancia_lateral = round(distancia_lateral, 3)
                # print("Distancia lateral: ", distancia_lateral)
                # -1 porque quando era acima do meio era negativo e agora quero positivo
                #* Calculo do Z (o quanto o braco tem de andar para cima ou para baixo)
                #* (após multiplicar por -1) -> se objeto estiver acima do centro da camera, o valor é positivo
                distancia_vertical = (tan(radians(V_Angle)) * centre_depth * -1) + (THRESHOLD_ALTURA/2)
                #! Calculos para acertar o Z consoante a distancia do objeto ao centro da camera verticalmente
                # distancia_vertical = round((((tan(radians(V_Angle)) * centre_depth) * -1) + THRESHOLD_ALTURA + (sin(radians(5)) * distanceToFruit)), 3)


                # distancia_vertical = 0.30759*(distancia_vertical)**2+1.48689*(distancia_vertical)-2.9605

                if distancia_vertical < -0.02:
                    distancia_vertical += 0.025
                elif distancia_vertical < 0:
                    distancia_vertical += 0.032
                elif distancia_vertical < 0.025:
                    distancia_vertical += 0.035
                elif distancia_vertical < 0.05:
                    distancia_vertical += 0.043
                elif distancia_vertical < 0.075:
                    distancia_vertical += 0.05
                elif distancia_vertical < 0.1:
                    distancia_vertical += 0.057
                elif distancia_vertical < 0.125:
                    distancia_vertical += 0.064
                

                #? Calculus of the fruit width and height considering the depth to the object
                fruit_width_pixels = (detected_objects[i].box.tensor[0][2] - detected_objects[i].box.tensor[0][0]).item()
                fruit_height_pixels = (detected_objects[i].box.tensor[0][3] - detected_objects[i].box.tensor[0][1]).item()
                fruit_width = ((fruit_width_pixels * distanceToFruit) / RESOLUTION_X)
                fruit_height = ((fruit_height_pixels * distanceToFruit) / RESOLUTION_Y)

                claw_origin = (0.035, 0, -0.05)
                fruit_location = (depthFromObjectToClawThreshold, distancia_lateral, distancia_vertical)
                distance_claw_to_fruit = ((fruit_location[0] - claw_origin[0])**2 + (fruit_location[1] - claw_origin[1])**2 + (fruit_location[2] - claw_origin[2])**2)**0.5


                #! Global Coordinates calculation
                new_Distance_km = distanceToFruit/1000
                
                latFruit = asin(sin(latDrone) * cos(new_Distance_km/R) + cos(latDrone) * sin(new_Distance_km/R) * cos(brng))
                lonFruit = lonDrone + atan2(sin(brng) * sin(new_Distance_km/R) * cos(latDrone), cos(new_Distance_km/R) - sin(latDrone) * sin(latFruit))

                lateral_distance_with_FOV = tan(radians(HFOV)) * centre_depth * 2
                vertical_distance_with_FOV = tan(radians(VFOV)) * centre_depth * 2

                #? Heights are to water level
                gimbal_inclination = 0
                drone_height = 100 # Replace 100 with real drone height
                # (talvez seja - em vez de +)
                new_Angle = gimbal_inclination + V_Angle 
                fruit_altitude = drone_height - distanceToFruit * sin(radians(new_Angle))
                
                
                # v.draw_text(f"X: {depthFromObjectToClawThreshold:.3f}m\nY: {distancia_lateral:.3f}m\nZ: {distancia_vertical:3f}m\nW: {fruit_width:3f}m", (cX, cY + 20))


                # v.draw_text(f"X: {depthFromObjectToClawThreshold:.3f}m\nY: {distancia_lateral:.3f}m\nZ: {distancia_vertical:3f}m\nD: {distance_claw_to_fruit:3f}m", (cX, cY + 20))
                
                # v.draw_text(f"D: {distance_claw_to_fruit:3f}m", (cX, cY + 20))
                
                
                
                
                # v.draw_text("{:.2f}m".format(centre_depth), (cX, cY + 20))
                # v.draw_text(f"H_Angle:{H_Angle:.2f}\nV_Angle:{V_Angle:.2f}", (cX, cY + 35))
                # v.draw_text(f"latFruit:{round(degrees(latFruit),3):.8f}\nLon_B:{round(degrees(lonFruit),3):.8f}\nalt:{round(fruit_altitude,3):.2f}", (cX, cY + 20))
                # v.draw_text("{:.2f}m".format(distanceToFruit), (cX, cY + 70))
                # v.draw_text(f"{distance_claw_to_fruit:.3f}m\n{fruit_width}m", (cX, cY + 20))
                # v.draw_text(f"Fruit Width in Meters: {fruit_width:.2f}\nFruit Height in Meters: {fruit_height:.2f}", (cX, cY + 20))

                # v.draw_circle((CENTER_POINT_X, CENTER_POINT_Y), '#eeefff')
                


                short_label = labels[i].split()[0]
                if short_label == 'Red_Apple':
                    message_to_send_mario["Fruits"]["Red_Apple"].append({"x": depthFromObjectToClawThreshold, "y": distancia_lateral, "z": round(distancia_vertical,3), "width": round(fruit_width, 3), "height": round(fruit_height, 3), "weeks": random.randint(1,7)})
                elif short_label == 'Green_Apple':
                    message_to_send_mario["Fruits"]["Green_Apple"].append({"x": depthFromObjectToClawThreshold, "y": distancia_lateral, "z": round(distancia_vertical,3), "width": round(fruit_width, 3), "height": round(fruit_height, 3), "weeks": random.randint(1,7)})
                elif short_label == 'Orange':
                    message_to_send_mario["Fruits"]["Orange"].append({"x": depthFromObjectToClawThreshold, "y": distancia_lateral, "z": round(distancia_vertical,3), "width": round(fruit_width, 3), "height": round(fruit_height, 3), "weeks": random.randint(1,7)})
                elif short_label == 'Vase':
                    message_to_send_mario["Vase"].append({"x": depthFromObjectToClawThreshold, "y": distancia_lateral, "z": round(distancia_vertical,3), "width": round(fruit_width, 3), "height": round(fruit_height, 3), "perc":labels[i].split()[1]})

                if depthFromObjectToClawThreshold > 0 and distance_claw_to_fruit < 0.28 and not video_file: #! VER DA ALTURA para adicionar ao if
                    # message_to_send_ruben["detected"].append({"object": short_label, "x": depthFromObjectToClawThreshold, "y": distancia_lateral, "z": round(distancia_vertical,3), "dClawToFruit": round(distance_claw_to_fruit,3),"fruit_width": round(fruit_width, 2)})
                    message_to_send_ruben["detected"].append({"object": short_label, "x": depthFromObjectToClawThreshold, "y": distancia_lateral, "z": round(distancia_vertical,3), "dClawToFruit": round(distance_claw_to_fruit,3),"fruit_width": round(fruit_width, 2), "perc":labels[i].split()[1]})
                else:
                    if short_label == 'Vase':
                        message_to_send_ruben["detected"].append({"object": short_label, "x": depthFromObjectToClawThreshold, "y": distancia_lateral, "z": round(distancia_vertical,3), "dClawToFruit": round(distance_claw_to_fruit,3),"fruit_width": round(fruit_width, 2), "perc":labels[i].split()[1]})
                
                
                # Sort the list by the distance to the claw to send the closest fruit first
                message_to_send_ruben["detected"].sort(key = lambda i: i['dClawToFruit'])


            # check if there were vases detetect
            if detect_vases and len(message_to_send_mario["Vase"]) > 0:
                shared_vases_detected.set()
            # check if there were fruits detected
            if detect_fruits and any(message_to_send_mario["Fruits"].values()):
                shared_fruits_detected.set()

            result.write(v.output.get_image()[:,:,::-1])
            if video_file:
                cv2.imshow('Segmented Image', v.output.get_image())
            else:
                cv2.imshow('Segmented Image', v.output.get_image()[:,:,::-1])
                # cv2.imshow('Segmented Image', cv2.resize(v.output.get_image()[:,:,::-1], (960, 900)))

            time_end = time.time()
            total_time = time_end - time_start

            # # #? RTMP
            # rtmp_frames = v.output.get_image()
            # p.stdin.write(rtmp_frames.tobytes())

            # print("Time to process frame: {:.2f}".format(total_time))
            # print("FPS: {:.2f}\n".format(1/total_time))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                result.release()
                cv2.destroyAllWindows()
                sock.close()
                ws.close()
                break

            if video_file:
                cv2.imwrite('output.png', cv2.resize(v.output.get_image(), (1280, 960)))
            else:
                cv2.imwrite('output.png', cv2.resize(v.output.get_image()[:,:,::-1], (1280, 960)))
        
    finally:
        print("finally")
        sock.close()
        ws.close()