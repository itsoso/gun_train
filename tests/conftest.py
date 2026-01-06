"""
pytest 配置文件
全局 fixtures 和配置
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_frame():
    """生成测试用帧"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_720p():
    """生成720p测试帧"""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_1080p():
    """生成1080p测试帧"""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def mock_pose_keypoints():
    """模拟姿态关键点"""
    # 33个关键点，每个包含 (x, y, z, visibility)
    keypoints = []
    for i in range(33):
        keypoints.append({
            'x': 0.5 + np.random.uniform(-0.1, 0.1),
            'y': 0.5 + np.random.uniform(-0.1, 0.1),
            'z': np.random.uniform(-0.1, 0.1),
            'visibility': 0.9
        })
    return keypoints


@pytest.fixture
def mock_camera_manager():
    """Mock摄像头管理器"""
    manager = MagicMock()
    manager.get_workstation_frames.return_value = {}
    manager.cameras = {}
    manager.configs = {}
    manager.is_running = False
    return manager


@pytest.fixture
def timestamp():
    """当前时间戳"""
    return datetime.now()


# pytest配置
def pytest_configure(config):
    """pytest配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试集合"""
    # 如果未指定 --runslow，跳过慢速测试
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests"
    )

