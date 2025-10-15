from threading import Thread

import cv2
import torch
import numpy as np
import signal


# from Switch4EmbodiedAI.utils.helpers import signal_handler
class SimpleStreamModuleConfig():
    capture_card_index: int = 0
    viz_stream: bool = True  # Whether to visualize the stream module output
    save_stream: bool = False  # Whether to save the istream module output
    save_path: str = None  # Path to save the stream module output

class SimpleStreamModule:
    '''
    Get image from capture card.
    No complex processing, just a placeholder for future image processing tasks.
    '''
    def __init__(self, config):
        self.config = config
        self.capture_card_index = config.capture_card_index
        self.stream = cv2.VideoCapture(self.capture_card_index)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = True


    def start(self):
        # start the thread to read frames from the video stream
        self.stopped = False
        Thread(target=self.update, args=(), daemon=True).start()
        return self
    

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.frame = self.process_frame(self.frame)



    def read(self):
        # return the frame most recently read
        return self.frame
    
    def viz_frame(self):
        if self.config.viz_stream:
            cv2.imshow("Stream Module Output", self.frame)

    

    def close(self):
        # indicate that the thread should be stopped
        self.stopped = True



    def save_frame(self, frame, path):
        if frame:
            cv2.imwrite(path, frame)
            return True
        return False


    def process_frame(self, frame):
        
        return frame
    
    



# def test_StreamModule(stream_module_cfg):

#     Stream_module = SimpleStreamModule(stream_module_cfg)
#     Stream_module.start()

#     while True:
#         frame = Stream_module.read()
#         Stream_module.viz_frame()
#         if frame is None:
#             break

#         # Exit on 'esc' key press
#         if cv2.waitKey(1) &  0xFF == 27 or Stream_module.stopped:
#             break


#     Stream_module.close()
#     cv2.destroyAllWindows()




# if __name__ == '__main__':
    
#     from Switch4EmbodiedAI.utils.helpers import get_args, parse_StreamModule_cfg
#     signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
#     signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
#     args = get_args()
#     stream_module_cfg = parse_StreamModule_cfg(args)
#     test_StreamModule(stream_module_cfg)