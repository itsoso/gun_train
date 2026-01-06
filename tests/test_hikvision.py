"""
海康威视摄像头模块单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os
import queue
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.camera.hikvision import (
    HikvisionCameraConfig,
    HikvisionCamera,
    HikvisionCameraManager,
    CameraStatus,
    CameraFrame,
    generate_camera_configs
)


class TestHikvisionCameraConfig:
    """测试摄像头配置类"""
    
    def test_config_creation(self):
        """测试配置对象创建"""
        config = HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64",
            username="admin",
            password="password123"
        )
        
        assert config.camera_id == 1
        assert config.workstation_id == 1
        assert config.position == "front"
        assert config.ip == "192.168.1.64"
        assert config.port == 554  # 默认端口
        assert config.channel == 1  # 默认通道
        assert config.stream_type == "main"  # 默认主码流
        assert config.enabled == True
    
    def test_rtsp_url_main_stream(self):
        """测试主码流RTSP URL生成"""
        config = HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64",
            username="admin",
            password="password123",
            channel=1,
            stream_type="main"
        )
        
        expected = "rtsp://admin:password123@192.168.1.64:554/Streaming/Channels/101"
        assert config.rtsp_url == expected
    
    def test_rtsp_url_sub_stream(self):
        """测试子码流RTSP URL生成"""
        config = HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64",
            username="admin",
            password="password123",
            channel=1,
            stream_type="sub"
        )
        
        expected = "rtsp://admin:password123@192.168.1.64:554/Streaming/Channels/102"
        assert config.rtsp_url == expected
    
    def test_rtsp_url_channel_2(self):
        """测试通道2的RTSP URL"""
        config = HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64",
            username="admin",
            password="password123",
            channel=2,
            stream_type="main"
        )
        
        expected = "rtsp://admin:password123@192.168.1.64:554/Streaming/Channels/201"
        assert config.rtsp_url == expected
    
    def test_rtsp_url_masked(self):
        """测试密码隐藏的RTSP URL"""
        config = HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64",
            username="admin",
            password="password123"
        )
        
        masked_url = config.rtsp_url_masked
        
        assert "password123" not in masked_url
        assert "****" in masked_url
        assert "admin" in masked_url
    
    def test_custom_port(self):
        """测试自定义端口"""
        config = HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64",
            port=8554,
            username="admin",
            password="password123"
        )
        
        assert ":8554/" in config.rtsp_url


class TestCameraFrame:
    """测试帧数据类"""
    
    def test_frame_creation(self):
        """测试帧对象创建"""
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame = CameraFrame(
            camera_id=1,
            workstation_id=1,
            position="front",
            frame=frame_data,
            timestamp=datetime.now(),
            frame_number=100,
            width=640,
            height=480,
            fps=10.0
        )
        
        assert frame.camera_id == 1
        assert frame.workstation_id == 1
        assert frame.position == "front"
        assert frame.frame_number == 100
        assert frame.width == 640
        assert frame.height == 480
        assert frame.fps == 10.0
        assert frame.frame.shape == (480, 640, 3)


class TestHikvisionCamera:
    """测试摄像头控制类"""
    
    @pytest.fixture
    def camera_config(self):
        """摄像头配置fixture"""
        return HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64",
            username="admin",
            password="password123"
        )
    
    def test_camera_initialization(self, camera_config):
        """测试摄像头初始化"""
        camera = HikvisionCamera(
            config=camera_config,
            target_fps=10,
            reconnect_interval=5,
            max_reconnect_attempts=3
        )
        
        assert camera.config == camera_config
        assert camera.target_fps == 10
        assert camera.reconnect_interval == 5
        assert camera.max_reconnect_attempts == 3
        assert camera.status == CameraStatus.DISCONNECTED
        assert camera.is_running == False
    
    def test_camera_initial_stats(self, camera_config):
        """测试初始统计信息"""
        camera = HikvisionCamera(config=camera_config)
        
        stats = camera.get_stats()
        
        assert stats["total_frames"] == 0
        assert stats["dropped_frames"] == 0
        assert stats["reconnects"] == 0
        assert stats["errors"] == 0
        assert stats["status"] == "disconnected"
    
    @patch('cv2.VideoCapture')
    def test_connect_success(self, mock_video_capture, camera_config):
        """测试成功连接"""
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {3: 640, 4: 480, 5: 25}.get(prop, 0)
        mock_cap.set.return_value = True
        mock_video_capture.return_value = mock_cap
        
        camera = HikvisionCamera(config=camera_config)
        result = camera.connect()
        
        assert result == True
        assert camera.status == CameraStatus.CONNECTED
        assert camera.reconnect_count == 0
    
    @patch('cv2.VideoCapture')
    def test_connect_failure(self, mock_video_capture, camera_config):
        """测试连接失败"""
        # Mock失败的VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap.set.return_value = True
        mock_video_capture.return_value = mock_cap
        
        camera = HikvisionCamera(config=camera_config)
        result = camera.connect()
        
        assert result == False
        assert camera.status == CameraStatus.ERROR
    
    @patch('cv2.VideoCapture')
    def test_disconnect(self, mock_video_capture, camera_config):
        """测试断开连接"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.set.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640, 4: 480, 5: 25}.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        camera = HikvisionCamera(config=camera_config)
        camera.connect()
        camera.disconnect()
        
        assert camera.status == CameraStatus.DISCONNECTED
        assert camera.cap is None
        mock_cap.release.assert_called_once()
    
    def test_get_frame_empty_queue(self, camera_config):
        """测试空队列获取帧"""
        camera = HikvisionCamera(config=camera_config)
        
        frame = camera.get_frame(timeout=0.1)
        
        assert frame is None
    
    def test_get_frame_with_data(self, camera_config):
        """测试有数据时获取帧"""
        camera = HikvisionCamera(config=camera_config)
        
        # 手动放入帧数据
        test_frame = CameraFrame(
            camera_id=1,
            workstation_id=1,
            position="front",
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=datetime.now(),
            frame_number=1,
            width=640,
            height=480,
            fps=10.0
        )
        camera.frame_queue.put(test_frame)
        
        frame = camera.get_frame(timeout=1.0)
        
        assert frame is not None
        assert frame.camera_id == 1
        assert frame.frame_number == 1
    
    @patch('cv2.VideoCapture')
    def test_reconnect(self, mock_video_capture, camera_config):
        """测试重连机制"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.set.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 640, 4: 480, 5: 25}.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        camera = HikvisionCamera(config=camera_config, reconnect_interval=0.1)
        camera.connect()
        
        # 模拟重连
        result = camera.reconnect()
        
        assert result == True
        assert camera.reconnect_count == 1
        assert camera.stats["reconnects"] == 1


class TestHikvisionCameraManager:
    """测试摄像头管理器"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = HikvisionCameraManager(target_fps=10)
        
        assert manager.target_fps == 10
        assert len(manager.cameras) == 0
        assert len(manager.configs) == 0
        assert manager.is_running == False
    
    def test_add_camera(self):
        """测试添加摄像头"""
        manager = HikvisionCameraManager()
        
        config = HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64"
        )
        
        manager.add_camera(config)
        
        assert 1 in manager.configs
        assert manager.configs[1] == config
    
    def test_add_cameras_batch(self):
        """测试批量添加摄像头"""
        manager = HikvisionCameraManager()
        
        configs = [
            HikvisionCameraConfig(camera_id=1, workstation_id=1, position="front", ip="192.168.1.64"),
            HikvisionCameraConfig(camera_id=2, workstation_id=1, position="side", ip="192.168.1.65"),
            HikvisionCameraConfig(camera_id=3, workstation_id=1, position="top", ip="192.168.1.66"),
        ]
        
        manager.add_cameras_batch(configs)
        
        assert len(manager.configs) == 3
        assert 1 in manager.configs
        assert 2 in manager.configs
        assert 3 in manager.configs
    
    def test_remove_camera(self):
        """测试移除摄像头"""
        manager = HikvisionCameraManager()
        
        config = HikvisionCameraConfig(camera_id=1, workstation_id=1, position="front", ip="192.168.1.64")
        manager.add_camera(config)
        
        manager.remove_camera(1)
        
        assert 1 not in manager.configs
    
    def test_get_status_summary_empty(self):
        """测试空管理器状态"""
        manager = HikvisionCameraManager()
        
        summary = manager.get_status_summary()
        
        assert summary["total"] == 0
        assert summary["connected"] == 0
        assert summary["health_rate"] == 0


class TestGenerateCameraConfigs:
    """测试配置生成函数"""
    
    def test_generate_single_workstation(self):
        """测试单工位配置生成"""
        configs = generate_camera_configs(
            workstation_count=1,
            base_ip="192.168.1",
            start_ip=64,
            username="admin",
            password="password123"
        )
        
        # 每工位3个摄像头
        assert len(configs) == 3
        
        # 检查位置
        positions = [c.position for c in configs]
        assert "front" in positions
        assert "side" in positions
        assert "top" in positions
    
    def test_generate_50_workstations(self):
        """测试50工位配置生成"""
        configs = generate_camera_configs(
            workstation_count=50,
            base_ip="192.168.1",
            start_ip=64,
            username="admin",
            password="password123"
        )
        
        # 50工位 × 3摄像头 = 150
        assert len(configs) == 150
        
        # 检查摄像头ID唯一性
        camera_ids = [c.camera_id for c in configs]
        assert len(camera_ids) == len(set(camera_ids))
        
        # 检查IP唯一性
        ips = [c.ip for c in configs]
        assert len(ips) == len(set(ips))
    
    def test_generate_correct_ips(self):
        """测试IP地址正确生成"""
        configs = generate_camera_configs(
            workstation_count=2,
            base_ip="192.168.1",
            start_ip=100,
            username="admin",
            password="password123"
        )
        
        ips = [c.ip for c in configs]
        
        expected_ips = [
            "192.168.1.100",
            "192.168.1.101",
            "192.168.1.102",
            "192.168.1.103",
            "192.168.1.104",
            "192.168.1.105"
        ]
        
        assert ips == expected_ips
    
    def test_generate_correct_workstation_mapping(self):
        """测试工位映射正确"""
        configs = generate_camera_configs(
            workstation_count=3,
            base_ip="192.168.1",
            start_ip=64
        )
        
        # 工位1应该有3个摄像头
        ws1_cameras = [c for c in configs if c.workstation_id == 1]
        assert len(ws1_cameras) == 3
        
        # 工位2应该有3个摄像头
        ws2_cameras = [c for c in configs if c.workstation_id == 2]
        assert len(ws2_cameras) == 3
    
    def test_generate_uses_sub_stream(self):
        """测试默认使用子码流"""
        configs = generate_camera_configs(workstation_count=1)
        
        for config in configs:
            assert config.stream_type == "sub"


class TestCameraStatus:
    """测试摄像头状态枚举"""
    
    def test_status_values(self):
        """测试状态值"""
        assert CameraStatus.DISCONNECTED.value == "disconnected"
        assert CameraStatus.CONNECTING.value == "connecting"
        assert CameraStatus.CONNECTED.value == "connected"
        assert CameraStatus.RECONNECTING.value == "reconnecting"
        assert CameraStatus.ERROR.value == "error"


class TestConcurrency:
    """并发测试"""
    
    def test_frame_queue_thread_safety(self):
        """测试帧队列线程安全"""
        config = HikvisionCameraConfig(
            camera_id=1,
            workstation_id=1,
            position="front",
            ip="192.168.1.64"
        )
        camera = HikvisionCamera(config=config)
        
        results = {"produced": 0, "consumed": 0}
        stop_event = threading.Event()
        
        def producer():
            for i in range(50):
                frame = CameraFrame(
                    camera_id=1,
                    workstation_id=1,
                    position="front",
                    frame=np.zeros((480, 640, 3), dtype=np.uint8),
                    timestamp=datetime.now(),
                    frame_number=i,
                    width=640,
                    height=480,
                    fps=10.0
                )
                try:
                    camera.frame_queue.put(frame, timeout=0.1)
                    results["produced"] += 1
                except queue.Full:
                    pass
                time.sleep(0.01)
        
        def consumer():
            while not stop_event.is_set():
                frame = camera.get_frame(timeout=0.1)
                if frame is not None:
                    results["consumed"] += 1
        
        import threading
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        time.sleep(0.2)
        stop_event.set()
        consumer_thread.join()
        
        # 应该消费了一部分帧
        assert results["consumed"] > 0


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
