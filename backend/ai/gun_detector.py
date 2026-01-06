"""
枪支检测模块
使用YOLOv8检测枪支位置、方向和相关信息
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
import math


@dataclass
class GunDetection:
    """枪支检测结果"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float  # 置信度
    gun_type: str  # 枪支类型
    muzzle_point: Optional[Tuple[int, int]]  # 枪口位置
    muzzle_direction: Optional[float]  # 枪口方向（角度）
    trigger_point: Optional[Tuple[int, int]]  # 扳机位置
    is_magazine_loaded: bool  # 是否装有弹匣


class SafeZone:
    """安全区域定义"""
    
    def __init__(
        self,
        target_zones: List[Tuple[int, int, int, int]],
        ground_zone: Tuple[int, int, int, int],
        image_width: int,
        image_height: int
    ):
        """
        Args:
            target_zones: 靶区列表 [(x1, y1, x2, y2), ...]
            ground_zone: 地面安全区 (x1, y1, x2, y2)
            image_width: 图像宽度
            image_height: 图像高度
        """
        self.target_zones = target_zones
        self.ground_zone = ground_zone
        self.image_width = image_width
        self.image_height = image_height
    
    def is_point_in_safe_zone(self, point: Tuple[int, int]) -> bool:
        """
        判断点是否在安全区域内
        
        Args:
            point: (x, y)
            
        Returns:
            是否在安全区
        """
        x, y = point
        
        # 检查是否在靶区
        for zone in self.target_zones:
            x1, y1, x2, y2 = zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        
        # 检查是否在地面区
        x1, y1, x2, y2 = self.ground_zone
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
        
        return False


class GunDetector:
    """枪支检测器"""
    
    # 支持的枪支类型
    GUN_TYPES = {
        0: "92式手枪",
        1: "54式手枪",
        2: "77式手枪",
        3: "95式步枪",
        4: "81式步枪"
    }
    
    def __init__(self, model_path: str = "models/gun_detection_yolov8.pt"):
        """
        初始化枪支检测器
        
        Args:
            model_path: YOLO模型路径
        """
        # 加载YOLO模型
        # 注意：实际使用时需要训练专门的枪支检测模型
        try:
            self.model = YOLO(model_path)
        except:
            # 如果没有训练好的模型，使用预训练模型做演示
            print("⚠️ 未找到专用枪支检测模型，使用通用目标检测模型")
            self.model = YOLO('yolov8n.pt')  # 使用nano版本
        
        self.confidence_threshold = 0.5
    
    def detect(self, image: np.ndarray) -> List[GunDetection]:
        """
        检测图像中的枪支
        
        Args:
            image: BGR格式图像
            
        Returns:
            检测结果列表
        """
        # 运行检测
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # 获取边界框
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # 获取枪支类型
                gun_type = self.GUN_TYPES.get(class_id, "未知型号")
                
                # 估算枪口和扳机位置（基于边界框的几何关系）
                muzzle_point = self._estimate_muzzle_position(x1, y1, x2, y2, image)
                trigger_point = self._estimate_trigger_position(x1, y1, x2, y2)
                
                # 计算枪口方向
                muzzle_direction = self._calculate_muzzle_direction(
                    x1, y1, x2, y2, muzzle_point
                )
                
                # 检测是否装弹匣（这需要更精细的模型，这里简化处理）
                is_magazine_loaded = self._detect_magazine(image, x1, y1, x2, y2)
                
                detection = GunDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=confidence,
                    gun_type=gun_type,
                    muzzle_point=muzzle_point,
                    muzzle_direction=muzzle_direction,
                    trigger_point=trigger_point,
                    is_magazine_loaded=is_magazine_loaded
                )
                
                detections.append(detection)
        
        return detections
    
    def _estimate_muzzle_position(
        self,
        x1: float, y1: float, x2: float, y2: float,
        image: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        估算枪口位置（简化版本，实际应该用关键点检测）
        
        手枪通常枪口在边界框的某个角落
        """
        # 提取枪支ROI
        roi = image[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return None
        
        # 简化处理：假设枪口是ROI中最突出的点
        # 实际应该训练专门的关键点检测模型
        
        # 这里用边界框的几何中心作为简化
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # 假设枪口在右侧（需要根据实际情况调整）
        muzzle_x = int(x2)
        muzzle_y = center_y
        
        return (muzzle_x, muzzle_y)
    
    def _estimate_trigger_position(
        self,
        x1: float, y1: float, x2: float, y2: float
    ) -> Optional[Tuple[int, int]]:
        """估算扳机位置"""
        # 简化：扳机通常在枪身中后部
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # 扳机在中心偏后位置
        trigger_x = int(x1 + (x2 - x1) * 0.4)
        trigger_y = center_y
        
        return (trigger_x, trigger_y)
    
    def _calculate_muzzle_direction(
        self,
        x1: float, y1: float, x2: float, y2: float,
        muzzle_point: Optional[Tuple[int, int]]
    ) -> Optional[float]:
        """
        计算枪口方向角度
        
        Returns:
            角度（0-360度，0度为正右方）
        """
        if muzzle_point is None:
            return None
        
        # 计算从边界框中心到枪口的向量
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        dx = muzzle_point[0] - center_x
        dy = muzzle_point[1] - center_y
        
        # 计算角度
        angle = math.degrees(math.atan2(dy, dx))
        
        # 转换为0-360度
        if angle < 0:
            angle += 360
        
        return angle
    
    def _detect_magazine(
        self,
        image: np.ndarray,
        x1: float, y1: float, x2: float, y2: float
    ) -> bool:
        """
        检测是否装有弹匣
        
        简化版本，实际应该用分类模型
        """
        # 这里返回False作为默认值（空枪训练）
        # 实际应该训练专门的分类器
        return False
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[GunDetection],
        safe_zone: Optional[SafeZone] = None
    ) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
            safe_zone: 安全区域（用于判断枪口方向）
            
        Returns:
            标注后的图像
        """
        annotated = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # 判断枪口是否指向安全区
            is_safe = True
            if safe_zone and detection.muzzle_point:
                is_safe = safe_zone.is_point_in_safe_zone(detection.muzzle_point)
            
            # 根据安全性选择颜色
            color = (0, 255, 0) if is_safe else (0, 0, 255)
            
            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{detection.gun_type} {detection.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # 绘制枪口位置和方向
            if detection.muzzle_point:
                muzzle_x, muzzle_y = detection.muzzle_point
                cv2.circle(annotated, (muzzle_x, muzzle_y), 5, (255, 0, 0), -1)
                
                # 绘制枪口方向线
                if detection.muzzle_direction is not None:
                    # 计算方向线的终点
                    length = 100
                    angle_rad = math.radians(detection.muzzle_direction)
                    end_x = int(muzzle_x + length * math.cos(angle_rad))
                    end_y = int(muzzle_y + length * math.sin(angle_rad))
                    
                    cv2.arrowedLine(
                        annotated,
                        (muzzle_x, muzzle_y),
                        (end_x, end_y),
                        (255, 0, 0),
                        2
                    )
            
            # 绘制扳机位置
            if detection.trigger_point:
                trigger_x, trigger_y = detection.trigger_point
                cv2.circle(annotated, (trigger_x, trigger_y), 3, (0, 255, 255), -1)
            
            # 显示弹匣状态
            if detection.is_magazine_loaded:
                cv2.putText(
                    annotated,
                    "LOADED",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
        
        return annotated


class MuzzleDirectionChecker:
    """枪口方向检查器"""
    
    def __init__(self, safe_zone: SafeZone):
        self.safe_zone = safe_zone
    
    def check_safety(self, detection: GunDetection) -> Tuple[bool, str]:
        """
        检查枪口方向是否安全
        
        Args:
            detection: 枪支检测结果
            
        Returns:
            (是否安全, 警告信息)
        """
        if detection.muzzle_point is None:
            return False, "无法确定枪口位置"
        
        # 检查枪口是否指向安全区
        is_safe = self.safe_zone.is_point_in_safe_zone(detection.muzzle_point)
        
        if not is_safe:
            # 计算枪口方向的描述
            direction_desc = self._get_direction_description(
                detection.muzzle_direction
            )
            return False, f"⚠️ 危险！枪口指向非安全区域（{direction_desc}）"
        
        return True, "枪口方向安全"
    
    def _get_direction_description(self, angle: Optional[float]) -> str:
        """将角度转换为方向描述"""
        if angle is None:
            return "未知方向"
        
        # 将360度分为8个方向
        directions = ["右", "右下", "下", "左下", "左", "左上", "上", "右上"]
        index = int((angle + 22.5) / 45) % 8
        
        return directions[index]


# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    detector = GunDetector()
    
    # 定义安全区域（示例）
    safe_zone = SafeZone(
        target_zones=[(800, 200, 1200, 600)],  # 靶区
        ground_zone=(0, 600, 1920, 1080),      # 地面
        image_width=1920,
        image_height=1080
    )
    
    direction_checker = MuzzleDirectionChecker(safe_zone)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测枪支
            detections = detector.detect(frame)
            
            # 绘制检测结果
            annotated_frame = detector.draw_detections(frame, detections, safe_zone)
            
            # 检查安全性
            for detection in detections:
                is_safe, message = direction_checker.check_safety(detection)
                
                if not is_safe:
                    # 在屏幕上显示警告
                    cv2.putText(
                        annotated_frame,
                        message,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )
            
            cv2.imshow('Gun Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

