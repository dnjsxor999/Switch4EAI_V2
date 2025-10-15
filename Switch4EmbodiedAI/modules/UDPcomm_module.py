import socket
import struct
from collections import deque
import numpy as np

class UDPComm:
    def __init__(self, host, port, block = True):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if not block:
            self.sock.setblocking(0)

    def start_server(self):
        """ Initialize the socket for use as a server """
        self.sock.bind((self.host, self.port))
        print(f"Server started on {self.host}:{self.port}")

    def receive_message(self, num_elements, data_type='f'):
        """ Receive messages (as server or client with listening capability) """
        data_format = f'{num_elements}{data_type}'
        expected_size = struct.calcsize(data_format)
        data, addr = self.sock.recvfrom(expected_size)

        if len(data) == expected_size:
            unpacked_data = struct.unpack(data_format, data)
            # print(f"Received message: {unpacked_data} from {addr}")
            return unpacked_data, True
        if not data:
            # print(f"Received incomplete data from {addr}")
            return None, False
        else:
            return None, True

    def receive_message_cont(self, num_elements, data_type='f'):
        """ Receive messages in a non-blocking way """
        data_format = f'{num_elements}{data_type}'
        expected_size = struct.calcsize(data_format)
        try:
            data, addr = self.sock.recvfrom(expected_size)
            unpacked_data = struct.unpack(data_format, data)
            if len(unpacked_data*4) == expected_size:
                return unpacked_data, True
        except socket.error as e:
            if e.errno == socket.errno.EWOULDBLOCK:
                # No data available
                return None, False
            else:
                # Some other socket error has occurred
                raise
        return None, False

    def send_message(self, floats, dest_host, dest_port):
        """ Send messages to a specified server """
        data = struct.pack(f'{len(floats)}f', *floats)
        self.sock.sendto(data, (dest_host, dest_port))
        # print(f"Message sent to {dest_host}:{dest_port}")

    def close(self):
        """ Close the socket """
        self.sock.close()
        print("Socket closed.")





class HistoryBuffer:
    def __init__(self, max_size):
        self.history = deque(maxlen=max_size)

    def update(self, new_obs, num_obs):
        """Update the history buffer with a new observation vector."""
        obs_packet = new_obs.flatten()[:num_obs]  # Assuming the obs_packet takes the first 8 elements
        self.history.append(obs_packet)  # Append new observation packet to the history

    def get_history(self):
        """Get the current history of observation packets as a flattened vector."""
        if self.history:
            return np.vstack(self.history).flatten()  # Stack and then flatten into a 1D vector
        else:
            return np.array([])  # Return an empty array if no history

    def update_multiple_envs(self, new_obs):
        self.history.append(new_obs)  # Append new observation packet to the history


    def get_history_multiple_envs(self):
        """Get the current history of observation packets as a flattened vector."""
        if self.history:
            return np.array(self.history)
        else:
            return np.array([])