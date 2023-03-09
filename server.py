import socket
import struct
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from threading import *
import time



# Set up socket connection
HOST = '192.168.1.48'
PORT = 8880
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print(f"Waiting for client to connect on port {PORT}...")




class VideoStreamer:
    """
    Video streamer that takes advantage of multi-threading, and continuously is reading frames.
    Frames are then ready to read when program requires.
    """
    def __init__(self, video_file=None):
        """
        When initialised, VideoStreamer object should be reading frames
        """
        self.setup_image_config(video_file)
        self.configure_streams()
        self.stopped = False

    def start(self):
        """
        Initialise thread, update method will run under thread
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        Constantly read frames until stop() method is introduced
        """
        # colorizer = rs.colorizer()
        while True:

            if self.stopped:
                return

            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            
            # Convert image to numpy array and initialise images
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())
            # self.depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    def stop(self):
        self.pipeline.stop()
        self.stopped = True

    def read(self):
        return (self.color_image, self.depth_image)
        # return (self.color_image, self.depth_image, self.depth_colormap)

    def setup_image_config(self, video_file=None):
        """
        Setup config and video steams. If --file is specified as an argument, setup
        stream from file. The input of --file is a .bag file in the bag_files folder.
        .bag files can be created using d435_to_file in the tools folder.
        video_file is by default None, and thus will by default stream from the 
        device connected to the USB.
        """
        config = rs.config()

        if video_file is None:
            
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
        else:
            try:
                config.enable_device_from_file("{}".format(video_file))
                
            except:
                print("Cannot enable device from: '{}'".format(video_file))

        self.config = config

    def configure_streams(self):
        # Configure video streams
        self.pipeline = rs.pipeline()
    
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def get_depth_scale(self):
        return self.profile.get_device().first_depth_sensor().get_depth_scale()

# Main loop
try:
    video_streamer = VideoStreamer().start()
    time.sleep(1)
    while True:
        # Wait for a client to connect
        conn, addr = sock.accept()
        print(f"Client connected from {addr}")
        
        # Send depth and color frames
        while True:
            # # Wait for a new frame
            # frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            # color_frame = frames.get_color_frame()

            # # Convert depth frame to a numpy array for visualization
            # depth_image = np.asanyarray(depth_frame.get_data())
            # print(f"depth image: {depth_image}")

            # # Convert color frame to a numpy array for visualization
            # color_image = np.asanyarray(color_frame.get_data())

            color_image, depth_image = video_streamer.read()

            print(f"depth image: {depth_image}")

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


            # Display the depth and color frames on the server side
            # cv2.imshow("Depth Stream (Server)", depth_image)
            # cv2.imshow("Color Stream (Server)", color_image)
            # cv2.imshow("Depth ColorMap Stream (Server)", depth_colormap)

            # Send depth frame size and data
            depth_data = depth_image.tobytes()
            depth_size = struct.pack('I', len(depth_data))
            conn.send(depth_size)
            conn.send(depth_data)

            # Send color frame size and data
            color_data = color_image.tobytes()
            color_size = struct.pack('I', len(color_data))
            conn.send(color_size)
            conn.send(color_data)

            # cv2.imshow("Depth Stream (Server)", depth_image)
            # cv2.imshow("Color Stream (Server)", color_image)

            cv2.imshow("Depth Stream (Server)", depth_image.astype(np.float32) / 65535.0)
            cv2.imshow("Depth ColorMap Stream (Server)", depth_colormap)


            # Check for key press to exit
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break

        # Clean up the connection and window on server exit
        cv2.destroyAllWindows()
        conn.close()

finally:
    # pipeline.stop()
    sock.close()




























# import socket
# import struct
# import pyrealsense2 as rs
# import cv2
# import numpy as np

# # Set up socket connection
# HOST = 'localhost'
# PORT = 8880
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.bind((HOST, PORT))
# sock.listen(1)
# print(f"Waiting for client to connect on port {PORT}...")

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# # Main loop
# try:
#     while True:
#         # Wait for a client to connect
#         conn, addr = sock.accept()
#         print(f"Client connected from {addr}")
        
#         # Send depth and color frames
#         while True:
#             # Wait for a new frame
#             frames = pipeline.wait_for_frames()
#             depth_frame = frames.get_depth_frame()
#             color_frame = frames.get_color_frame()

#             # Send depth frame size and data
#             depth_image = depth_frame.get_data()
#             depth_data = bytearray(depth_image)
#             depth_size = struct.pack('I', len(depth_data))
#             conn.send(depth_size)
#             conn.send(depth_data)

#             # Send color frame size and data
#             color_image = color_frame.get_data()
#             color_data = bytearray(color_image)
#             color_size = struct.pack('I', len(color_data))
#             conn.send(color_size)
#             conn.send(color_data)

#             # cv2.imshow('Depth Stream', depth_image)
#             # print(f"depth image: {depth_image}")

#             color_image_cv = np.asanyarray(color_frame.get_data())
#             depth_image_cv = np.asanyarray(depth_frame.get_data())

#             cv2.imshow('Color Stream', color_image_cv)
#             cv2.imshow('Depth Stream', depth_image_cv)

# finally:
#     pipeline.stop()
#     sock.close()
