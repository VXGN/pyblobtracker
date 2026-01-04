"""
Async frame buffer for optimized video loading
Uses multiprocessing for frame decoding and async queues for buffering
"""

import cv2
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue
import time


class FrameBuffer:
    """Thread-safe frame buffer with prefetching"""
    
    def __init__(self, video_path, buffer_size=30):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        self.is_exhausted = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Video properties
        cap = cv2.VideoCapture(video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_position = 0
        cap.release()
        
        # Reopen for reading
        self.cap = None
    
    def start(self):
        """Start the buffer filling thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.is_exhausted = False
        self.cap = cv2.VideoCapture(self.video_path)
        self._thread = threading.Thread(target=self._fill_buffer, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the buffer filling thread"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        # Clear buffer
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break
    
    def _fill_buffer(self):
        """Background thread to fill the buffer"""
        while self.is_running:
            if self.buffer.full():
                time.sleep(0.001)  # Brief sleep when buffer is full
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                self.is_exhausted = True
                break
            
            try:
                self.buffer.put((ret, frame), timeout=0.1)
                with self._lock:
                    self.current_position += 1
            except queue.Full:
                continue
    
    def read_frame(self):
        """Read a frame from the buffer"""
        if self.is_exhausted and self.buffer.empty():
            return False, None
        
        try:
            return self.buffer.get(timeout=0.5)
        except queue.Empty:
            if self.is_exhausted:
                return False, None
            return False, None
    
    def seek(self, frame_num):
        """Seek to a specific frame"""
        self.stop()
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        with self._lock:
            self.current_position = frame_num
        self.start()
    
    def reset(self):
        """Reset to beginning"""
        self.seek(0)
    
    def release(self):
        """Release all resources"""
        self.stop()


class AsyncVideoProcessor:
    """Async video processor with frame buffering"""
    
    def __init__(self, video_path, buffer_size=30):
        self.video_path = video_path
        self.frame_buffer = FrameBuffer(video_path, buffer_size)
        
        # Copy properties from buffer
        self.fps = self.frame_buffer.fps
        self.width = self.frame_buffer.width
        self.height = self.frame_buffer.height
        self.total_frames = self.frame_buffer.total_frames
        
        # Direct cap for fallback
        self._direct_cap = None
    
    def start_buffering(self):
        """Start frame buffering"""
        self.frame_buffer.start()
    
    def read_frame(self):
        """Read next frame"""
        return self.frame_buffer.read_frame()
    
    def read_frame_direct(self):
        """Read frame directly without buffer (for seeking)"""
        if self._direct_cap is None:
            self._direct_cap = cv2.VideoCapture(self.video_path)
        return self._direct_cap.read()
    
    def seek(self, frame_num):
        """Seek to specific frame"""
        self.frame_buffer.seek(frame_num)
    
    def reset(self):
        """Reset to beginning"""
        self.frame_buffer.reset()
    
    def release(self):
        """Release all resources"""
        self.frame_buffer.release()
        if self._direct_cap:
            self._direct_cap.release()
    
    def create_writer(self, output_path):
        """Create video writer"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
    
    @property
    def is_exhausted(self):
        return self.frame_buffer.is_exhausted and self.frame_buffer.buffer.empty()


class ParallelFrameProcessor:
    """Process frames in parallel using thread pool"""
    
    def __init__(self, max_workers=None):
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []
    
    def submit(self, func, *args, **kwargs):
        """Submit a processing task"""
        future = self.executor.submit(func, *args, **kwargs)
        self._futures.append(future)
        return future
    
    def get_results(self, timeout=None):
        """Get all results"""
        results = []
        for future in self._futures:
            try:
                results.append(future.result(timeout=timeout))
            except Exception as e:
                results.append(None)
        self._futures.clear()
        return results
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=False)
