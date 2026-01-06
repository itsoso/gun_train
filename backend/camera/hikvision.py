"""
海康威视摄像头集成模块
支持RTSP流接入、自动重连、多路并发处理
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
import logging
from datetime import datetime
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """摄像头状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class HikvisionCameraConfig:
    """海康威视摄像头配置"""
    camera_id: int                          # 摄像头ID
    workstation_id: int                     # 工位号
    position: str                           # 位置：front/side/top
    ip: str                                 # IP地址
    port: int = 554                         # RTSP端口
    username: str = "admin"                 # 用户名
    password: str = ""                      # 密码
    channel: int = 1                        # 通道号
    stream_type: str = "main"               # 码流类型: main/sub
    enabled: bool = True                    # 是否启用
    
    @property
    def rtsp_url(self) -> str:
        """生成RTSP URL"""
        # 主码流101，子码流102
        stream_code = f"{self.channel}0{'1' if self.stream_type == 'main' else '2'}"
        return f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}/Streaming/Channels/{stream_code}"
    
    @property
    def rtsp_url_masked(self) -> str:
        """生成隐藏密码的RTSP URL（用于日志）"""
        stream_code = f"{self.channel}0{'1' if self.stream_type == 'main' else '2'}"
        return f"rtsp://{self.username}:****@{self.ip}:{self.port}/Streaming/Channels/{stream_code}"


@dataclass
class CameraFrame:
    """摄像头帧数据"""
    camera_id: int
    workstation_id: int
    position: str
    frame: np.ndarray
    timestamp: datetime
    frame_number: int
    width: int
    height: int
    fps: float


class HikvisionCamera:
    """海康威视摄像头控制类"""
    
    def __init__(
        self,
        config: HikvisionCameraConfig,
        target_fps: int = 10,
        reconnect_interval: int = 5,
        max_reconnect_attempts: int = 10
    ):
        """
        Args:
            config: 摄像头配置
            target_fps: 目标帧率（降低以减少计算负担）
            reconnect_interval: 重连间隔（秒）
            max_reconnect_attempts: 最大重连次数
        """
        self.config = config
        self.target_fps = target_fps
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.status = CameraStatus.DISCONNECTED
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=5)
        
        self.frame_count = 0
        self.reconnect_count = 0
        self.last_frame_time = 0
        self.actual_fps = 0.0
        
        # 统计信息
        self.stats = {
            "total_frames": 0,
            "dropped_frames": 0,
            "reconnects": 0,
            "errors": 0,
            "start_time": None
        }
    
    def connect(self) -> bool:
        """连接摄像头"""
        self.status = CameraStatus.CONNECTING
        logger.info(f"📹 连接摄像头 {self.config.camera_id}: {self.config.rtsp_url_masked}")
        
        try:
            # 设置FFmpeg后端选项（降低延迟）
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000"
            
            # 打开RTSP流
            self.cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
            
            # 设置参数
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10秒超时
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5秒读取超时
            
            if not self.cap.isOpened():
                raise ConnectionError("无法打开视频流")
            
            # 读取第一帧验证连接
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ConnectionError("无法读取视频帧")
            
            # 获取视频信息
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"✅ 摄像头 {self.config.camera_id} 连接成功: {width}x{height}@{fps}fps")
            
            self.status = CameraStatus.CONNECTED
            self.reconnect_count = 0
            self.stats["start_time"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 摄像头 {self.config.camera_id} 连接失败: {e}")
            self.status = CameraStatus.ERROR
            self.stats["errors"] += 1
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.status = CameraStatus.DISCONNECTED
        logger.info(f"📹 摄像头 {self.config.camera_id} 已断开")
    
    def reconnect(self) -> bool:
        """重新连接"""
        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.error(f"❌ 摄像头 {self.config.camera_id} 超过最大重连次数")
            self.status = CameraStatus.ERROR
            return False
        
        self.status = CameraStatus.RECONNECTING
        self.reconnect_count += 1
        self.stats["reconnects"] += 1
        
        logger.warning(f"🔄 摄像头 {self.config.camera_id} 尝试重连 ({self.reconnect_count}/{self.max_reconnect_attempts})")
        
        self.disconnect()
        time.sleep(self.reconnect_interval)
        
        return self.connect()
    
    def start(self):
        """启动视频采集线程"""
        if self.is_running:
            return
        
        if not self.connect():
            logger.error(f"❌ 摄像头 {self.config.camera_id} 启动失败")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"🎥 摄像头 {self.config.camera_id} 采集线程已启动")
    
    def stop(self):
        """停止视频采集"""
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=3)
        
        self.disconnect()
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info(f"⏹️ 摄像头 {self.config.camera_id} 已停止")
    
    def _capture_loop(self):
        """采集循环"""
        frame_interval = 1.0 / self.target_fps
        fps_update_interval = 1.0
        fps_frame_count = 0
        fps_start_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            # 控制帧率
            elapsed = current_time - self.last_frame_time
            if elapsed < frame_interval:
                time.sleep(0.001)
                continue
            
            # 检查连接状态
            if self.cap is None or not self.cap.isOpened():
                if not self.reconnect():
                    time.sleep(self.reconnect_interval)
                    continue
            
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning(f"⚠️ 摄像头 {self.config.camera_id} 读取失败")
                    self.stats["errors"] += 1
                    
                    if not self.reconnect():
                        time.sleep(self.reconnect_interval)
                    continue
                
                # 创建帧对象
                camera_frame = CameraFrame(
                    camera_id=self.config.camera_id,
                    workstation_id=self.config.workstation_id,
                    position=self.config.position,
                    frame=frame,
                    timestamp=datetime.now(),
                    frame_number=self.frame_count,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    fps=self.actual_fps
                )
                
                # 放入队列
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.stats["dropped_frames"] += 1
                    except queue.Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(camera_frame)
                    self.frame_count += 1
                    self.stats["total_frames"] += 1
                    self.last_frame_time = current_time
                    fps_frame_count += 1
                except queue.Full:
                    self.stats["dropped_frames"] += 1
                
                # 更新FPS统计
                if current_time - fps_start_time >= fps_update_interval:
                    self.actual_fps = fps_frame_count / (current_time - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = current_time
                
            except Exception as e:
                logger.error(f"❌ 摄像头 {self.config.camera_id} 采集错误: {e}")
                self.stats["errors"] += 1
                time.sleep(0.1)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[CameraFrame]:
        """
        获取一帧图像
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            CameraFrame对象，超时返回None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["status"] = self.status.value
        stats["actual_fps"] = round(self.actual_fps, 2)
        stats["reconnect_count"] = self.reconnect_count
        
        if stats["start_time"]:
            uptime = (datetime.now() - stats["start_time"]).total_seconds()
            stats["uptime_seconds"] = int(uptime)
        
        return stats


class HikvisionCameraManager:
    """海康威视摄像头管理器"""
    
    def __init__(self, target_fps: int = 10):
        """
        Args:
            target_fps: 目标帧率
        """
        self.target_fps = target_fps
        self.cameras: Dict[int, HikvisionCamera] = {}
        self.configs: Dict[int, HikvisionCameraConfig] = {}
        self.is_running = False
        
    def add_camera(self, config: HikvisionCameraConfig):
        """添加摄像头"""
        self.configs[config.camera_id] = config
        logger.info(f"📹 添加摄像头配置: ID={config.camera_id}, 工位={config.workstation_id}, 位置={config.position}")
    
    def add_cameras_batch(self, configs: List[HikvisionCameraConfig]):
        """批量添加摄像头"""
        for config in configs:
            self.add_camera(config)
    
    def remove_camera(self, camera_id: int):
        """移除摄像头"""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
        
        if camera_id in self.configs:
            del self.configs[camera_id]
    
    def start_all(self):
        """启动所有摄像头"""
        self.is_running = True
        
        for camera_id, config in self.configs.items():
            if not config.enabled:
                continue
            
            camera = HikvisionCamera(
                config=config,
                target_fps=self.target_fps
            )
            camera.start()
            self.cameras[camera_id] = camera
        
        logger.info(f"🎥 已启动 {len(self.cameras)} 个摄像头")
    
    def stop_all(self):
        """停止所有摄像头"""
        self.is_running = False
        
        for camera in self.cameras.values():
            camera.stop()
        
        self.cameras.clear()
        logger.info("⏹️ 所有摄像头已停止")
    
    def get_frame(self, camera_id: int) -> Optional[CameraFrame]:
        """获取指定摄像头的帧"""
        camera = self.cameras.get(camera_id)
        if camera:
            return camera.get_frame(timeout=0.5)
        return None
    
    def get_workstation_frames(self, workstation_id: int) -> Dict[str, Optional[CameraFrame]]:
        """
        获取指定工位的所有摄像头帧
        
        Returns:
            {"front": frame, "side": frame, "top": frame}
        """
        frames = {}
        
        for camera_id, config in self.configs.items():
            if config.workstation_id == workstation_id:
                camera = self.cameras.get(camera_id)
                if camera:
                    frames[config.position] = camera.get_frame(timeout=0.3)
        
        return frames
    
    def get_all_stats(self) -> Dict[int, Dict]:
        """获取所有摄像头统计信息"""
        stats = {}
        for camera_id, camera in self.cameras.items():
            stats[camera_id] = camera.get_stats()
        return stats
    
    def get_status_summary(self) -> Dict:
        """获取状态汇总"""
        total = len(self.cameras)
        connected = sum(1 for c in self.cameras.values() if c.status == CameraStatus.CONNECTED)
        reconnecting = sum(1 for c in self.cameras.values() if c.status == CameraStatus.RECONNECTING)
        error = sum(1 for c in self.cameras.values() if c.status == CameraStatus.ERROR)
        
        return {
            "total": total,
            "connected": connected,
            "reconnecting": reconnecting,
            "error": error,
            "health_rate": round(connected / total * 100, 1) if total > 0 else 0
        }


def generate_camera_configs(
    workstation_count: int = 50,
    base_ip: str = "192.168.1",
    start_ip: int = 64,
    username: str = "admin",
    password: str = "password123"
) -> List[HikvisionCameraConfig]:
    """
    生成摄像头配置（50工位，每工位3个摄像头）
    
    Args:
        workstation_count: 工位数量
        base_ip: IP前缀
        start_ip: 起始IP
        username: 用户名
        password: 密码
        
    Returns:
        摄像头配置列表
    """
    configs = []
    camera_id = 1
    
    positions = ["front", "side", "top"]
    
    for ws_id in range(1, workstation_count + 1):
        for pos in positions:
            ip = f"{base_ip}.{start_ip + camera_id - 1}"
            
            config = HikvisionCameraConfig(
                camera_id=camera_id,
                workstation_id=ws_id,
                position=pos,
                ip=ip,
                username=username,
                password=password,
                stream_type="sub"  # 使用子码流减少带宽
            )
            
            configs.append(config)
            camera_id += 1
    
    return configs


# 使用示例
if __name__ == "__main__":
    # 单摄像头测试
    config = HikvisionCameraConfig(
        camera_id=1,
        workstation_id=1,
        position="front",
        ip="192.168.1.64",
        username="admin",
        password="your_password",
        stream_type="sub"
    )
    
    print(f"RTSP URL: {config.rtsp_url_masked}")
    
    camera = HikvisionCamera(config, target_fps=10)
    camera.start()
    
    try:
        print("🎥 开始采集，按Ctrl+C停止...")
        while True:
            frame = camera.get_frame()
            if frame:
                print(f"帧 {frame.frame_number}: {frame.width}x{frame.height}, FPS={frame.fps:.1f}")
                
                # 显示画面（测试用）
                cv2.imshow(f"Camera {frame.camera_id}", frame.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n⏹️ 停止采集...")
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        
        # 打印统计
        stats = camera.get_stats()
        print(f"\n📊 统计信息:")
        print(f"  总帧数: {stats['total_frames']}")
        print(f"  丢帧数: {stats['dropped_frames']}")
        print(f"  重连次数: {stats['reconnects']}")
        print(f"  错误次数: {stats['errors']}")

