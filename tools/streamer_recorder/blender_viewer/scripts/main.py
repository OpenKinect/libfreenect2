from bge import (
        logic,
        texture,
        )

import socket
import bgl

# # add cv2 lib (requires numpy higher version than current in blender)
import sys
sys.path.append('/usr/local/Cellar/python3/3.4.3/Frameworks/Python.framework/Versions/3.4/lib/python3.4/site-packages')
import numpy
import cv2
# from PIL import Image
import itertools

def init(controller):
    """
    Init, run until socket is connected
    """

    # run once per frame
    if controller.sensors[0].positive and controller.sensors[1].positive:

        ### SOCKET DEFINITION
        # run once
        # if not hasattr(logic, 'socket'):

        # define socket, set socket parameters
        logic.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        logic.socket.setblocking(0)
        timeOut = 0.001
        logic.socket.settimeout(timeOut)

        # bind socket (run until binded)
        host = '127.0.0.1'
        port_rcv = 10000
        logic.socket.bind((host,port_rcv))
        print('bind socket: IP = {} Port = {}'.format(host, port_rcv))

        controller.owner['SocketConnected'] = True

        ### IMAGE TEXTURE
        # get screen object
        obj = logic.getCurrentScene().objects['VideoScreen']
        # get the reference pointer (ID) of the texture
        ID = texture.materialID(obj, 'MAVideoMat')
        # create a texture object
        logic.texture = texture.Texture(obj, ID)

        # logic.imageIndex = 0

def run(controller):
    """
    Run, run once every frames
    """
    if controller.sensors[0].positive:
        try:
            buff_size = 4096
            msg_raw = logic.socket.recv(buff_size)

            # check for header msg indicating the number of packet that'll follow
            # containing a unique frame (image)
            if len(msg_raw) == 4:
                nb_of_packet_per_frame = msg_raw[0]

                # loop over next packets to reconstitute image
                frame_raw = b''
                for i in range(nb_of_packet_per_frame):
                    frame_raw = frame_raw + logic.socket.recv(buff_size)

                # frame = cv2.imdecode(numpy.fromstring(frame_raw, dtype=numpy.uint8), cv2.IMREAD_COLOR)
                frame = cv2.imdecode(numpy.fromstring(frame_raw, dtype=numpy.uint8), cv2.IMREAD_GRAYSCALE)


                if not frame is None:

                    width = frame.shape[1]
                    height = frame.shape[0]

                    l = frame.tolist()
                    lll = list(itertools.chain(*l))

                    # image_buffer = bgl.Buffer(bgl.GL_INT, [width*height*3], lll)
                    image_buffer = bgl.Buffer(bgl.GL_BYTE, width*height, lll)

                    source = texture.ImageBuff()

                    # Apply a filter, that way source.load does not except a 3(RGB) pixel image
                    source.filter = texture.FilterBlueScreen()
                    source.load(image_buffer, width, height)

                    logic.texture.source = source
                    logic.texture.refresh(False)


        except socket.timeout:
            pass


def end(controller):
    """
    called when ending BGE (e.g. to properly close network connections)
    """
    if controller.sensors[0].positive:

        # close socket
        logic.socket.close()
        controller.owner['SocketConnected'] = False
        controller.owner['End'] = True
        print('close socket')

        # close cv2 window
        cv2.destroyAllWindows()

        # end game engine
        logic.endGame()
