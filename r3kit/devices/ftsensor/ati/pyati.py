import os
from typing import List, Dict, Union, Optional
import struct
import time
import numpy as np
from threading import Thread, Lock, Event
from copy import deepcopy
from functools import partial
import socket

from r3kit.devices.ftsensor.base import FTSensorBase
from r3kit.devices.ftsensor.ati.config import *
from r3kit.utils.vis import draw_time, draw_items

'''
Modified from: https://github.com/Liuyvjin/ati-sensor/blob/master/pyati/ati_sensor.py
'''


class RDTCommand():
    HEADER = 0x1234
    # Possible values for command
    CMD_STOP_STREAMING = 0
    CMD_START_STREAMING = 2
    CMD_SET_SOFTWARE_BIAS = 0x0042
    # Special values for sample count
    INFINITE_SAMPLES = 0

    @classmethod
    def pack(self, command, count=INFINITE_SAMPLES):
        return struct.pack('!HHI', self.HEADER, command, count)


class PyATI(FTSensorBase):
    def __init__(self, id:str=PYATI_ID, port:int=PYATI_PORT, fps:int=PYATI_FPS, name:str='PyATI') -> None:
        super().__init__(name=name)

        self._id = id
        self._port = port
        self._fps = fps

        # set up socket
        retry = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._connected = Event()
        self._sock.setblocking(1)
        self._sock.settimeout(10)
        while retry < PYATI_RETRY:
            try:
                self._sock.connect((self._id, self._port))
                self._connected.set()
                break
            except socket.timeout:
                retry += 1
            except Exception as e:
                retry += 1
            time.sleep(PYATI_RETRY_DELAY)
        if retry >= PYATI_RETRY:
            raise RuntimeError("Cannot connect to ATI sensor")
        
        # stream
        self.in_streaming = Event()
    
    def __del__(self):
        if self.in_streaming.is_set():
            self.stop_streaming()
        if hasattr(self, "_sock"):
            self._sock.close()

    def _send_cmd(self, command, count=0) -> None:
        if self._connected.wait(1):
            self._sock.send(RDTCommand.pack(command, count))
        else:
            raise RuntimeError("Cannot connect to ATI sensor")

    def _recv_data(self) -> np.ndarray:
        # return: (6,)
        if self._connected.wait(1):
            raw_data = self._sock.recv(1024)
            raw_data = np.array(struct.unpack('!3I6i', raw_data)[3:])
            return raw_data * PYATI_SCALE
        else:
            raise RuntimeError("Cannot connect to ATI sensor")

    def _read(self, n:int=1) -> Dict[str, Union[float, np.ndarray]]:
        # return: (N, 6)
        fts = np.empty((n, 6))
        self._send_cmd(RDTCommand.CMD_START_STREAMING, n)
        for i in range(n):
            fts[i] = self._recv_data()
        receive_time = time.time() * 1000
        return {'ft': fts, 'timestamp_ms': receive_time}
    
    def get(self) -> Dict[str, Union[float, np.ndarray]]:
        if not self.in_streaming.is_set():
            data = self._read()
        else:
            self.streaming_mutex.acquire()
            data = {}
            data['ft'] = self.streaming_data['ft'][-1]
            data['timestamp_ms'] = self.streaming_data['timestamp_ms'][-1]
            self.streaming_mutex.release()
        return data
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        
        self.in_streaming.set()
        if callback is None:
            self.streaming_mutex = Lock()
            self.streaming_data = {
                "ft": [], 
                "timestamp_ms": []
            }
        else:
            pass
        self._send_cmd(RDTCommand.CMD_START_STREAMING, RDTCommand.INFINITE_SAMPLES)
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()
    
    def stop_streaming(self) -> Dict[str, Union[List[np.ndarray], List[float]]]:
        self.in_streaming.clear()
        self.thread.join()
        self._send_cmd(RDTCommand.CMD_STOP_STREAMING, 0)
        streaming_data = self.streaming_data
        self.streaming_data = {
            "ft": [], 
            "timestamp_ms": []
        }
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        assert len(streaming_data["ft"]) ==  len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
        draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        np.save(os.path.join(save_path, "ft.npy"), np.array(streaming_data["ft"], dtype=float))
        draw_items(np.array(streaming_data["ft"], dtype=float), os.path.join(save_path, "ft.png"))
    
    def collect_streaming(self, collect:bool=True) -> None:
        self._collect_streaming_data = collect
    
    def _streaming_data(self, callback:Optional[callable]=None):
        while self.in_streaming.is_set():
            # fps
            time.sleep(1/self._fps)

            # get data
            if not self._collect_streaming_data:
                continue
            ft = self._recv_data()
            receive_time = time.time() * 1000
            data = {'ft': ft, 'timestamp_ms': receive_time}
            if callback is None:
                self.streaming_mutex.acquire()
                self.streaming_data['ft'].append(data['ft'])
                self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                self.streaming_mutex.release()
            else:
                callback(deepcopy(data))
    
    @staticmethod
    def raw2tare(raw_ft:np.ndarray, tare:Dict[str, Union[float, np.ndarray]], pose:np.ndarray) -> np.ndarray:
        '''
        raw_ft: raw force torque data
        pose: 3x3 rotation matrix from ft300 to base
        '''
        raw_f, raw_t = raw_ft[:3], raw_ft[3:]
        f = raw_f - tare['f0']
        f -= np.linalg.inv(pose) @ np.array([0., 0., -9.8 * tare['m']])
        t = raw_t - tare['t0']
        t -= np.linalg.inv(pose) @ np.cross(np.linalg.inv(pose) @ np.array(tare['c']), np.array([0., 0., -9.8 * tare['m']]))
        return np.concatenate([f, t])


if __name__ == '__main__':
    sensor = PyATI(id='192.168.1.10', port=49152, fps=100, name='PyATI')

    while True:
        data = sensor.get()
        print(data)
        time.sleep(0.1)
