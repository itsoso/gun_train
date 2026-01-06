"""
动作分析模块单元测试
"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ai.pose_detector import PoseKeypoints, AngleCalculator


class TestAngleCalculator:
    """测试角度计算器"""
    
    def setup_method(self):
        """测试前置"""
        self.calc = AngleCalculator()
    
    def test_calculate_angle_90_degrees(self):
        """测试90度角计算"""
        # 构造一个90度角
        point1 = (0.0, 0.0, 1.0)  # 左端点
        point2 = (0.5, 0.0, 1.0)  # 顶点
        point3 = (0.5, 0.5, 1.0)  # 上端点
        
        angle = self.calc.calculate_angle(point1, point2, point3)
        
        # 允许1度误差
        assert abs(angle - 90.0) < 1.0
    
    def test_calculate_angle_180_degrees(self):
        """测试180度角（直线）"""
        point1 = (0.0, 0.0, 1.0)
        point2 = (0.5, 0.0, 1.0)
        point3 = (1.0, 0.0, 1.0)
        
        angle = self.calc.calculate_angle(point1, point2, point3)
        
        assert abs(angle - 180.0) < 1.0
    
    def test_calculate_angle_60_degrees(self):
        """测试60度角"""
        import math
        point1 = (0.0, 0.0, 1.0)
        point2 = (0.5, 0.0, 1.0)
        # 使用三角函数计算60度对应的点
        point3 = (0.5 + 0.5 * math.cos(math.radians(60)), 
                  0.5 * math.sin(math.radians(60)), 1.0)
        
        angle = self.calc.calculate_angle(point1, point2, point3)
        
        assert abs(angle - 60.0) < 1.0
    
    def test_calculate_distance(self):
        """测试距离计算"""
        point1 = (0.0, 0.0, 1.0)
        point2 = (3.0, 4.0, 1.0)
        
        distance = self.calc.calculate_distance(point1, point2)
        
        assert abs(distance - 5.0) < 0.01  # 3-4-5直角三角形
    
    def test_calculate_distance_same_point(self):
        """测试同一点距离为0"""
        point = (1.0, 1.0, 1.0)
        
        distance = self.calc.calculate_distance(point, point)
        
        assert abs(distance) < 0.001
    
    def test_is_point_visible_high_confidence(self):
        """测试高置信度点可见"""
        point = (0.5, 0.5, 0.9)
        
        assert self.calc.is_point_visible(point) == True
    
    def test_is_point_visible_low_confidence(self):
        """测试低置信度点不可见"""
        point = (0.5, 0.5, 0.3)
        
        assert self.calc.is_point_visible(point) == False
    
    def test_is_point_visible_threshold(self):
        """测试自定义阈值"""
        point = (0.5, 0.5, 0.7)
        
        assert self.calc.is_point_visible(point, threshold=0.6) == True
        assert self.calc.is_point_visible(point, threshold=0.8) == False


class TestPoseKeypoints:
    """测试姿态关键点"""
    
    def test_keypoints_creation(self):
        """测试关键点对象创建"""
        keypoints = PoseKeypoints(
            left_shoulder=(0.3, 0.3, 0.9),
            right_shoulder=(0.7, 0.3, 0.9),
            left_elbow=(0.2, 0.5, 0.8),
            right_elbow=(0.8, 0.5, 0.8),
            left_wrist=(0.1, 0.7, 0.7),
            right_wrist=(0.9, 0.7, 0.7),
            left_pinky=(0.05, 0.75, 0.6),
            right_pinky=(0.95, 0.75, 0.6),
            left_index=(0.08, 0.72, 0.6),
            right_index=(0.92, 0.72, 0.6),
            nose=(0.5, 0.2, 0.95),
            left_eye=(0.45, 0.15, 0.9),
            right_eye=(0.55, 0.15, 0.9),
            left_hip=(0.35, 0.6, 0.85),
            right_hip=(0.65, 0.6, 0.85),
        )
        
        assert keypoints.nose[0] == 0.5
        assert keypoints.left_shoulder[2] == 0.9
    
    def test_keypoints_to_dict(self):
        """测试关键点转字典"""
        keypoints = PoseKeypoints(
            left_shoulder=(0.3, 0.3, 0.9),
            right_shoulder=(0.7, 0.3, 0.9),
            left_elbow=(0.2, 0.5, 0.8),
            right_elbow=(0.8, 0.5, 0.8),
            left_wrist=(0.1, 0.7, 0.7),
            right_wrist=(0.9, 0.7, 0.7),
            left_pinky=(0.05, 0.75, 0.6),
            right_pinky=(0.95, 0.75, 0.6),
            left_index=(0.08, 0.72, 0.6),
            right_index=(0.92, 0.72, 0.6),
            nose=(0.5, 0.2, 0.95),
            left_eye=(0.45, 0.15, 0.9),
            right_eye=(0.55, 0.15, 0.9),
            left_hip=(0.35, 0.6, 0.85),
            right_hip=(0.65, 0.6, 0.85),
        )
        
        result = keypoints.to_dict()
        
        assert isinstance(result, dict)
        assert 'left_shoulder' in result
        assert 'right_wrist' in result
        assert len(result) == 15


class TestActionAnalyzerIntegration:
    """动作分析集成测试"""
    
    def create_standard_pose(self):
        """创建标准持枪姿势的关键点"""
        return PoseKeypoints(
            left_shoulder=(0.35, 0.25, 0.95),
            right_shoulder=(0.65, 0.25, 0.95),
            left_elbow=(0.25, 0.45, 0.9),
            right_elbow=(0.75, 0.45, 0.9),
            left_wrist=(0.30, 0.55, 0.85),
            right_wrist=(0.70, 0.55, 0.85),
            left_pinky=(0.28, 0.58, 0.8),
            right_pinky=(0.72, 0.58, 0.8),
            left_index=(0.32, 0.58, 0.8),
            right_index=(0.68, 0.58, 0.8),
            nose=(0.50, 0.10, 0.98),
            left_eye=(0.45, 0.08, 0.95),
            right_eye=(0.55, 0.08, 0.95),
            left_hip=(0.40, 0.55, 0.9),
            right_hip=(0.60, 0.55, 0.9),
        )
    
    def test_elbow_angle_calculation(self):
        """测试肘部角度计算"""
        keypoints = self.create_standard_pose()
        calc = AngleCalculator()
        
        # 计算右臂肘部角度
        angle = calc.calculate_angle(
            keypoints.right_shoulder,
            keypoints.right_elbow,
            keypoints.right_wrist
        )
        
        # 应该在合理范围内
        assert 0 < angle < 180
    
    def test_shoulder_level_check(self):
        """测试肩膀水平度检查"""
        keypoints = self.create_standard_pose()
        
        # 标准姿势下肩膀应该基本水平
        height_diff = abs(
            keypoints.left_shoulder[1] - keypoints.right_shoulder[1]
        )
        
        assert height_diff < 0.1  # 差距小于10%


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

