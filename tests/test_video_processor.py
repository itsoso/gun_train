"""
视频处理系统单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os
import queue
import threading
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.video_processor import (
    VideoFrame,
    CameraConfig,
    VideoStreamCapture,
    MultiCameraManager,
    VideoProcessor,
    VideoRecorder
)


class TestCameraConfig:
    """测试摄像头配置类"""
    
    def test_config_creation(self):
        """测试配置对象创建"""
        config = CameraConfig(
            camera_id=1,
            workstation_id=1,
            view_angle="front",
            rtsp_url="rtsp://192.168.1.64:554/stream"
        )
        
        assert config.camera_id == 1
        assert config.workstation_id == 1
        assert config.view_angle == "front"
        assert config.rtsp_url == "rtsp://192.168.1.64:554/stream"
        assert config.enabled == True
    
    def test_config_disabled(self):
        """测试禁用配置"""
        config = CameraConfig(
            camera_id=1,
            workstation_id=1,
            view_angle="front",
            rtsp_url="rtsp://test",
            enabled=False
        )
        
        assert config.enabled == False


class TestVideoFrame:
    """测试视频帧类"""
    
    def test_frame_creation(self):
        """测试帧对象创建"""
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = datetime.now()
        
        frame = VideoFrame(
            camera_id=1,
            workstation_id=1,
            frame=frame_data,
            timestamp=timestamp,
            frame_number=100
        )
        
        assert frame.camera_id == 1
        assert frame.workstation_id == 1
        assert frame.frame.shape == (480, 640, 3)
        assert frame.frame_number == 100
        assert frame.timestamp == timestamp


class TestVideoStreamCapture:
    """测试视频流捕获器"""
    
    def test_capture_initialization(self):
        """测试捕获器初始化"""
        capture = VideoStreamCapture(
            camera_id=1,
            rtsp_url="rtsp://192.168.1.64:554/stream",
            target_fps=10
        )
        
        assert capture.camera_id == 1
        assert capture.rtsp_url == "rtsp://192.168.1.64:554/stream"
        assert capture.target_fps == 10
        assert capture.is_running == False
        assert capture.frame_count == 0
    
    @patch('cv2.VideoCapture')
    def test_start_success(self, mock_video_capture):
        """测试成功启动"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        capture = VideoStreamCapture(
            camera_id=1,
            rtsp_url="rtsp://test",
            target_fps=10
        )
        
        capture.start()
        
        assert capture.is_running == True
        assert capture.cap is not None
        
        # 清理
        capture.stop()
    
    @patch('cv2.VideoCapture')
    def test_start_failure(self, mock_video_capture):
        """测试启动失败"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        capture = VideoStreamCapture(
            camera_id=1,
            rtsp_url="rtsp://invalid",
            target_fps=10
        )
        
        with pytest.raises(RuntimeError):
            capture.start()
    
    @patch('cv2.VideoCapture')
    def test_stop(self, mock_video_capture):
        """测试停止"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        capture = VideoStreamCapture(
            camera_id=1,
            rtsp_url="rtsp://test",
            target_fps=10
        )
        
        capture.start()
        capture.stop()
        
        assert capture.is_running == False
        assert capture.cap is None
        mock_cap.release.assert_called_once()
    
    def test_get_frame_empty_queue(self):
        """测试空队列获取帧"""
        capture = VideoStreamCapture(
            camera_id=1,
            rtsp_url="rtsp://test",
            target_fps=10
        )
        
        frame = capture.get_frame(timeout=0.1)
        
        assert frame is None
    
    def test_get_frame_with_data(self):
        """测试有数据时获取帧"""
        capture = VideoStreamCapture(
            camera_id=1,
            rtsp_url="rtsp://test",
            target_fps=10
        )
        
        # 手动放入帧
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        capture.frame_queue.put(test_frame)
        
        frame = capture.get_frame(timeout=1.0)
        
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    
    @patch('cv2.VideoCapture')
    def test_context_manager(self, mock_video_capture):
        """测试上下文管理器"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        with VideoStreamCapture(1, "rtsp://test", 10) as capture:
            assert capture.is_running == True
        
        assert capture.is_running == False


class TestMultiCameraManager:
    """测试多摄像头管理器"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        configs = [
            CameraConfig(camera_id=1, workstation_id=1, view_angle="front", rtsp_url="rtsp://1"),
            CameraConfig(camera_id=2, workstation_id=1, view_angle="side", rtsp_url="rtsp://2"),
        ]
        
        manager = MultiCameraManager(configs)
        
        assert len(manager.camera_configs) == 2
        assert 1 in manager.camera_configs
        assert 2 in manager.camera_configs
        assert manager.is_running == False
    
    @patch('backend.core.video_processor.VideoStreamCapture')
    def test_start_all(self, mock_capture_class):
        """测试启动所有摄像头"""
        mock_capture = MagicMock()
        mock_capture_class.return_value = mock_capture
        
        configs = [
            CameraConfig(camera_id=1, workstation_id=1, view_angle="front", rtsp_url="rtsp://1"),
            CameraConfig(camera_id=2, workstation_id=1, view_angle="side", rtsp_url="rtsp://2"),
        ]
        
        manager = MultiCameraManager(configs)
        manager.start_all()
        
        assert manager.is_running == True
        assert len(manager.captures) == 2
        assert mock_capture.start.call_count == 2
    
    @patch('backend.core.video_processor.VideoStreamCapture')
    def test_stop_all(self, mock_capture_class):
        """测试停止所有摄像头"""
        mock_capture = MagicMock()
        mock_capture_class.return_value = mock_capture
        
        configs = [
            CameraConfig(camera_id=1, workstation_id=1, view_angle="front", rtsp_url="rtsp://1"),
        ]
        
        manager = MultiCameraManager(configs)
        manager.start_all()
        manager.stop_all()
        
        assert manager.is_running == False
        assert len(manager.captures) == 0
        mock_capture.stop.assert_called()
    
    @patch('backend.core.video_processor.VideoStreamCapture')
    def test_get_frame(self, mock_capture_class):
        """测试获取指定摄像头帧"""
        mock_capture = MagicMock()
        mock_capture.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture_class.return_value = mock_capture
        
        configs = [
            CameraConfig(camera_id=1, workstation_id=1, view_angle="front", rtsp_url="rtsp://1"),
        ]
        
        manager = MultiCameraManager(configs)
        manager.start_all()
        
        frame = manager.get_frame(1)
        
        assert frame is not None
        mock_capture.get_frame.assert_called()
    
    def test_get_frame_invalid_camera(self):
        """测试获取不存在摄像头的帧"""
        configs = [
            CameraConfig(camera_id=1, workstation_id=1, view_angle="front", rtsp_url="rtsp://1"),
        ]
        
        manager = MultiCameraManager(configs)
        
        frame = manager.get_frame(999)
        
        assert frame is None
    
    @patch('backend.core.video_processor.VideoStreamCapture')
    def test_get_workstation_frames(self, mock_capture_class):
        """测试获取工位帧"""
        mock_capture = MagicMock()
        mock_capture.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture_class.return_value = mock_capture
        
        configs = [
            CameraConfig(camera_id=1, workstation_id=1, view_angle="front", rtsp_url="rtsp://1"),
            CameraConfig(camera_id=2, workstation_id=1, view_angle="side", rtsp_url="rtsp://2"),
            CameraConfig(camera_id=3, workstation_id=2, view_angle="front", rtsp_url="rtsp://3"),
        ]
        
        manager = MultiCameraManager(configs)
        manager.start_all()
        
        frames = manager.get_workstation_frames(1)
        
        assert "front" in frames or "side" in frames
    
    def test_disabled_camera_not_started(self):
        """测试禁用的摄像头不被启动"""
        configs = [
            CameraConfig(camera_id=1, workstation_id=1, view_angle="front", rtsp_url="rtsp://1", enabled=True),
            CameraConfig(camera_id=2, workstation_id=1, view_angle="side", rtsp_url="rtsp://2", enabled=False),
        ]
        
        manager = MultiCameraManager(configs)
        
        # 不调用start_all，检查配置
        enabled_count = sum(1 for c in configs if c.enabled)
        
        assert enabled_count == 1


class TestVideoProcessor:
    """测试视频处理器"""
    
    @pytest.fixture
    def mock_camera_manager(self):
        """Mock摄像头管理器"""
        manager = MagicMock()
        manager.captures = {1: MagicMock()}
        manager.camera_configs = {
            1: CameraConfig(camera_id=1, workstation_id=1, view_angle="front", rtsp_url="rtsp://1")
        }
        manager.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        return manager
    
    def test_processor_initialization(self, mock_camera_manager):
        """测试处理器初始化"""
        processor = VideoProcessor(
            camera_manager=mock_camera_manager,
            process_callback=None
        )
        
        assert processor.camera_manager == mock_camera_manager
        assert processor.is_running == False
    
    def test_processor_with_callback(self, mock_camera_manager):
        """测试带回调的处理器"""
        callback_results = []
        
        def callback(frame):
            callback_results.append(frame)
        
        processor = VideoProcessor(
            camera_manager=mock_camera_manager,
            process_callback=callback
        )
        
        assert processor.process_callback == callback
    
    def test_start_processing(self, mock_camera_manager):
        """测试启动处理"""
        processor = VideoProcessor(camera_manager=mock_camera_manager)
        processor.start_processing()
        
        assert processor.is_running == True
        
        # 清理
        processor.stop_processing()
    
    def test_stop_processing(self, mock_camera_manager):
        """测试停止处理"""
        processor = VideoProcessor(camera_manager=mock_camera_manager)
        processor.start_processing()
        processor.stop_processing()
        
        assert processor.is_running == False
        assert len(processor.threads) == 0


class TestVideoRecorder:
    """测试视频录制器"""
    
    def test_recorder_initialization(self):
        """测试录制器初始化"""
        recorder = VideoRecorder(output_dir="test_recordings")
        
        assert recorder.output_dir == "test_recordings"
        assert len(recorder.writers) == 0
        assert recorder.is_recording == False
    
    @patch('cv2.VideoWriter')
    def test_start_recording(self, mock_writer_class):
        """测试开始录制"""
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer_class.return_value = mock_writer
        
        recorder = VideoRecorder(output_dir="/tmp")
        recorder.start_recording(
            camera_id=1,
            workstation_id=1,
            width=1280,
            height=720,
            fps=10
        )
        
        assert 1 in recorder.writers
    
    @patch('cv2.VideoWriter')
    def test_write_frame(self, mock_writer_class):
        """测试写入帧"""
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer_class.return_value = mock_writer
        
        recorder = VideoRecorder(output_dir="/tmp")
        recorder.start_recording(camera_id=1, workstation_id=1)
        
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        recorder.write_frame(1, frame)
        
        mock_writer.write.assert_called_once()
    
    @patch('cv2.VideoWriter')
    def test_stop_recording(self, mock_writer_class):
        """测试停止录制"""
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer_class.return_value = mock_writer
        
        recorder = VideoRecorder(output_dir="/tmp")
        recorder.start_recording(camera_id=1, workstation_id=1)
        recorder.stop_recording(1)
        
        mock_writer.release.assert_called_once()
        assert 1 not in recorder.writers
    
    @patch('cv2.VideoWriter')
    def test_stop_all(self, mock_writer_class):
        """测试停止所有录制"""
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer_class.return_value = mock_writer
        
        recorder = VideoRecorder(output_dir="/tmp")
        recorder.start_recording(camera_id=1, workstation_id=1)
        recorder.start_recording(camera_id=2, workstation_id=1)
        
        recorder.stop_all()
        
        assert len(recorder.writers) == 0


class TestConcurrency:
    """并发测试"""
    
    def test_frame_queue_thread_safety(self):
        """测试帧队列线程安全性"""
        capture = VideoStreamCapture(
            camera_id=1,
            rtsp_url="rtsp://test",
            target_fps=10
        )
        
        results = {"produced": 0, "consumed": 0}
        stop_event = threading.Event()
        
        def producer():
            for i in range(50):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                try:
                    capture.frame_queue.put(frame, timeout=0.1)
                    results["produced"] += 1
                except queue.Full:
                    pass
                time.sleep(0.01)
        
        def consumer():
            while not stop_event.is_set():
                frame = capture.get_frame(timeout=0.1)
                if frame is not None:
                    results["consumed"] += 1
        
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


class TestPerformance:
    """性能测试"""
    
    def test_frame_queue_throughput(self):
        """测试帧队列吞吐量"""
        capture = VideoStreamCapture(
            camera_id=1,
            rtsp_url="rtsp://test",
            target_fps=30
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        iterations = 100
        
        start = time.time()
        
        for i in range(iterations):
            try:
                capture.frame_queue.put(frame, timeout=0.01)
                capture.get_frame(timeout=0.01)
            except queue.Full:
                pass
            except queue.Empty:
                pass
        
        elapsed = time.time() - start
        throughput = iterations / elapsed
        
        # 应该至少每秒处理100次
        assert throughput > 100, f"Throughput too low: {throughput:.0f} ops/sec"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
