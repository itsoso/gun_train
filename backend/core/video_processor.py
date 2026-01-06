"""
视频流处理模块
实时处理多路摄像头视频流
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import threading
import queue
import time
from datetime import datetime


@dataclass
class CameraConfig:
    """摄像头配置"""
    camera_id: int
    workstation_id: int
    view_angle: str  # front/side/top
    rtsp_url: str
    enabled: bool = True


@dataclass
class VideoFrame:
    """视频帧数据"""
    camera_id: int
    workstation_id: int
    frame: np.ndarray
    timestamp: datetime
    frame_number: int


class VideoStreamCapture:
    """视频流捕获器"""
    
    def __init__(
        self,
        camera_id: int,
        rtsp_url: str,
        target_fps: int = 10
    ):
        """
        Args:
            camera_id: 摄像头ID
            rtsp_url: RTSP流地址
            target_fps: 目标帧率（降低帧率以减少计算负担）
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.frame_count = 0
        
    def start(self):
        """启动视频流捕获"""
        if self.is_running:
            return
        
        # 打开视频流
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频流: {self.rtsp_url}")
        
        # 设置缓冲区大小
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """停止视频流捕获"""
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _capture_loop(self):
        """捕获循环（在独立线程中运行）"""
        last_capture_time = 0
        
        while self.is_running:
            current_time = time.time()
            
            # 按目标帧率捕获
            if current_time - last_capture_time < self.frame_interval:
                time.sleep(0.001)
                continue
            
            ret, frame = self.cap.read()
            
            if not ret:
                # 尝试重连
                print(f"⚠️ 摄像头 {self.camera_id} 读取失败，尝试重连...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.rtsp_url)
                continue
            
            # 添加到队列（如果队列满了则丢弃旧帧）
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put_nowait(frame)
                self.frame_count += 1
                last_capture_time = current_time
            except queue.Full:
                pass
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        获取一帧图像
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            图像数组，如果超时则返回None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class MultiCameraManager:
    """多摄像头管理器"""
    
    def __init__(self, camera_configs: List[CameraConfig]):
        """
        Args:
            camera_configs: 摄像头配置列表
        """
        self.camera_configs = {c.camera_id: c for c in camera_configs}
        self.captures: Dict[int, VideoStreamCapture] = {}
        self.is_running = False
        
    def start_all(self):
        """启动所有摄像头"""
        for camera_id, config in self.camera_configs.items():
            if not config.enabled:
                continue
            
            try:
                capture = VideoStreamCapture(
                    camera_id=camera_id,
                    rtsp_url=config.rtsp_url,
                    target_fps=10
                )
                capture.start()
                self.captures[camera_id] = capture
                print(f"✅ 摄像头 {camera_id} (工位{config.workstation_id}-{config.view_angle}) 启动成功")
            except Exception as e:
                print(f"❌ 摄像头 {camera_id} 启动失败: {e}")
        
        self.is_running = True
    
    def stop_all(self):
        """停止所有摄像头"""
        for capture in self.captures.values():
            capture.stop()
        
        self.captures.clear()
        self.is_running = False
    
    def get_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """获取指定摄像头的帧"""
        capture = self.captures.get(camera_id)
        if capture:
            return capture.get_frame()
        return None
    
    def get_all_frames(self) -> Dict[int, Optional[np.ndarray]]:
        """获取所有摄像头的当前帧"""
        frames = {}
        for camera_id, capture in self.captures.items():
            frames[camera_id] = capture.get_frame(timeout=0.5)
        return frames
    
    def get_workstation_frames(self, workstation_id: int) -> Dict[str, Optional[np.ndarray]]:
        """
        获取指定工位的所有摄像头帧
        
        Returns:
            {"front": frame, "side": frame, "top": frame}
        """
        frames = {}
        
        for camera_id, config in self.camera_configs.items():
            if config.workstation_id == workstation_id:
                capture = self.captures.get(camera_id)
                if capture:
                    frames[config.view_angle] = capture.get_frame(timeout=0.5)
        
        return frames
    
    def __enter__(self):
        self.start_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()


class VideoProcessor:
    """视频处理器（集成AI分析）"""
    
    def __init__(
        self,
        camera_manager: MultiCameraManager,
        process_callback: Optional[Callable[[VideoFrame], None]] = None
    ):
        """
        Args:
            camera_manager: 多摄像头管理器
            process_callback: 处理回调函数
        """
        self.camera_manager = camera_manager
        self.process_callback = process_callback
        self.is_running = False
        self.threads: List[threading.Thread] = []
        
    def start_processing(self):
        """启动视频处理"""
        self.is_running = True
        
        # 为每个摄像头创建处理线程
        for camera_id in self.camera_manager.captures.keys():
            thread = threading.Thread(
                target=self._process_loop,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
    
    def stop_processing(self):
        """停止视频处理"""
        self.is_running = False
        
        for thread in self.threads:
            thread.join(timeout=2)
        
        self.threads.clear()
    
    def _process_loop(self, camera_id: int):
        """处理循环（为每个摄像头独立运行）"""
        config = self.camera_manager.camera_configs[camera_id]
        frame_number = 0
        
        while self.is_running:
            # 获取帧
            frame = self.camera_manager.get_frame(camera_id)
            
            if frame is None:
                continue
            
            # 创建VideoFrame对象
            video_frame = VideoFrame(
                camera_id=camera_id,
                workstation_id=config.workstation_id,
                frame=frame,
                timestamp=datetime.now(),
                frame_number=frame_number
            )
            
            # 调用处理回调
            if self.process_callback:
                try:
                    self.process_callback(video_frame)
                except Exception as e:
                    print(f"❌ 处理帧时出错 (相机{camera_id}): {e}")
            
            frame_number += 1


class VideoRecorder:
    """视频录制器"""
    
    def __init__(self, output_dir: str = "recordings"):
        """
        Args:
            output_dir: 录像输出目录
        """
        self.output_dir = output_dir
        self.writers: Dict[int, cv2.VideoWriter] = {}
        self.is_recording = False
        
    def start_recording(
        self,
        camera_id: int,
        workstation_id: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 10
    ):
        """
        开始录制指定摄像头
        
        Args:
            camera_id: 摄像头ID
            workstation_id: 工位ID
            width: 视频宽度
            height: 视频高度
            fps: 帧率
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/workstation_{workstation_id}_camera_{camera_id}_{timestamp}.mp4"
        
        # 创建VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {filename}")
        
        self.writers[camera_id] = writer
        print(f"📹 开始录制: {filename}")
    
    def write_frame(self, camera_id: int, frame: np.ndarray):
        """
        写入一帧
        
        Args:
            camera_id: 摄像头ID
            frame: 图像帧
        """
        writer = self.writers.get(camera_id)
        if writer:
            writer.write(frame)
    
    def stop_recording(self, camera_id: int):
        """停止录制"""
        writer = self.writers.get(camera_id)
        if writer:
            writer.release()
            del self.writers[camera_id]
            print(f"⏹️ 停止录制摄像头 {camera_id}")
    
    def stop_all(self):
        """停止所有录制"""
        for camera_id in list(self.writers.keys()):
            self.stop_recording(camera_id)


# 使用示例
if __name__ == "__main__":
    # 配置摄像头（示例）
    camera_configs = [
        CameraConfig(
            camera_id=1,
            workstation_id=1,
            view_angle="front",
            rtsp_url="rtsp://192.168.1.101:554/stream"  # 替换为实际RTSP地址
        ),
        CameraConfig(
            camera_id=2,
            workstation_id=1,
            view_angle="side",
            rtsp_url="rtsp://192.168.1.102:554/stream"
        ),
    ]
    
    # 如果没有真实的RTSP流，使用本地摄像头测试
    camera_configs = [
        CameraConfig(
            camera_id=0,
            workstation_id=1,
            view_angle="front",
            rtsp_url=0  # 使用本地摄像头
        ),
    ]
    
    # 定义处理回调
    def process_frame(video_frame: VideoFrame):
        print(f"处理帧: 相机{video_frame.camera_id}, "
              f"工位{video_frame.workstation_id}, "
              f"帧号{video_frame.frame_number}")
        
        # 这里可以调用AI分析模块
        # analyzer.analyze(video_frame.frame)
        
        # 显示帧（仅用于测试）
        cv2.imshow(f"Camera {video_frame.camera_id}", video_frame.frame)
        cv2.waitKey(1)
    
    # 创建管理器
    try:
        with MultiCameraManager(camera_configs) as manager:
            # 创建处理器
            processor = VideoProcessor(manager, process_callback=process_frame)
            processor.start_processing()
            
            print("🎥 视频流处理中... 按Ctrl+C停止")
            
            # 运行一段时间
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️ 停止处理...")
            finally:
                processor.stop_processing()
                cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"❌ 错误: {e}")

