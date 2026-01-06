"""
人体姿态识别模块
使用MediaPipe进行实时姿态检测
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PoseKeypoints:
    """姿态关键点数据类"""
    # 上半身关键点
    left_shoulder: Tuple[float, float, float]  # (x, y, visibility)
    right_shoulder: Tuple[float, float, float]
    left_elbow: Tuple[float, float, float]
    right_elbow: Tuple[float, float, float]
    left_wrist: Tuple[float, float, float]
    right_wrist: Tuple[float, float, float]
    
    # 手部关键点
    left_pinky: Tuple[float, float, float]
    right_pinky: Tuple[float, float, float]
    left_index: Tuple[float, float, float]
    right_index: Tuple[float, float, float]
    
    # 头部关键点
    nose: Tuple[float, float, float]
    left_eye: Tuple[float, float, float]
    right_eye: Tuple[float, float, float]
    
    # 下半身关键点（用于姿态稳定性判断）
    left_hip: Tuple[float, float, float]
    right_hip: Tuple[float, float, float]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'left_shoulder': self.left_shoulder,
            'right_shoulder': self.right_shoulder,
            'left_elbow': self.left_elbow,
            'right_elbow': self.right_elbow,
            'left_wrist': self.left_wrist,
            'right_wrist': self.right_wrist,
            'left_pinky': self.left_pinky,
            'right_pinky': self.right_pinky,
            'left_index': self.left_index,
            'right_index': self.right_index,
            'nose': self.nose,
            'left_eye': self.left_eye,
            'right_eye': self.right_eye,
            'left_hip': self.left_hip,
            'right_hip': self.right_hip,
        }


class PoseDetector:
    """姿态检测器"""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        初始化姿态检测器
        
        Args:
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            model_complexity: 模型复杂度 (0, 1, 2)，越大越精确但越慢
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
    def detect(self, image: np.ndarray) -> Optional[PoseKeypoints]:
        """
        检测图像中的人体姿态
        
        Args:
            image: BGR格式的图像
            
        Returns:
            PoseKeypoints对象，如果未检测到人体则返回None
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # 提取关键点
        landmarks = results.pose_landmarks.landmark
        
        try:
            keypoints = PoseKeypoints(
                left_shoulder=self._get_landmark(landmarks, 11),
                right_shoulder=self._get_landmark(landmarks, 12),
                left_elbow=self._get_landmark(landmarks, 13),
                right_elbow=self._get_landmark(landmarks, 14),
                left_wrist=self._get_landmark(landmarks, 15),
                right_wrist=self._get_landmark(landmarks, 16),
                left_pinky=self._get_landmark(landmarks, 17),
                right_pinky=self._get_landmark(landmarks, 18),
                left_index=self._get_landmark(landmarks, 19),
                right_index=self._get_landmark(landmarks, 20),
                nose=self._get_landmark(landmarks, 0),
                left_eye=self._get_landmark(landmarks, 2),
                right_eye=self._get_landmark(landmarks, 5),
                left_hip=self._get_landmark(landmarks, 23),
                right_hip=self._get_landmark(landmarks, 24),
            )
            return keypoints
        except IndexError:
            return None
    
    def _get_landmark(self, landmarks, index: int) -> Tuple[float, float, float]:
        """获取特定索引的关键点坐标"""
        landmark = landmarks[index]
        return (landmark.x, landmark.y, landmark.visibility)
    
    def draw_landmarks(self, image: np.ndarray, keypoints: PoseKeypoints) -> np.ndarray:
        """
        在图像上绘制关键点
        
        Args:
            image: 原始图像
            keypoints: 关键点数据
            
        Returns:
            绘制了关键点的图像
        """
        annotated_image = image.copy()
        
        # 重新检测以获取绘图所需的landmarks对象
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2
                )
            )
        
        return annotated_image
    
    def close(self):
        """释放资源"""
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AngleCalculator:
    """角度计算工具类"""
    
    @staticmethod
    def calculate_angle(
        point1: Tuple[float, float, float],
        point2: Tuple[float, float, float],
        point3: Tuple[float, float, float]
    ) -> float:
        """
        计算三个点形成的角度（point2为顶点）
        
        Args:
            point1: 第一个点 (x, y, visibility)
            point2: 顶点 (x, y, visibility)
            point3: 第三个点 (x, y, visibility)
            
        Returns:
            角度值（0-180度）
        """
        # 提取坐标
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        x3, y3 = point3[0], point3[1]
        
        # 计算向量
        vector1 = np.array([x1 - x2, y1 - y2])
        vector2 = np.array([x3 - x2, y3 - y2])
        
        # 计算角度
        cos_angle = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    @staticmethod
    def calculate_distance(
        point1: Tuple[float, float, float],
        point2: Tuple[float, float, float]
    ) -> float:
        """
        计算两点之间的欧式距离
        
        Args:
            point1: 第一个点
            point2: 第二个点
            
        Returns:
            距离值
        """
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def is_point_visible(point: Tuple[float, float, float], threshold: float = 0.5) -> bool:
        """
        判断关键点是否可见
        
        Args:
            point: 关键点 (x, y, visibility)
            threshold: 可见性阈值
            
        Returns:
            是否可见
        """
        return point[2] > threshold


# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    detector = PoseDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测姿态
            keypoints = detector.detect(frame)
            
            if keypoints:
                # 绘制关键点
                annotated_frame = detector.draw_landmarks(frame, keypoints)
                
                # 计算右臂肘部角度（肩-肘-腕）
                angle_calc = AngleCalculator()
                elbow_angle = angle_calc.calculate_angle(
                    keypoints.right_shoulder,
                    keypoints.right_elbow,
                    keypoints.right_wrist
                )
                
                # 显示角度
                cv2.putText(
                    annotated_frame,
                    f"Right Elbow Angle: {elbow_angle:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Pose Detection', annotated_frame)
            else:
                cv2.imshow('Pose Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

