"""
动作规范性分析模块
基于规则引擎判断枪械操作的规范性
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

from .pose_detector import PoseKeypoints, AngleCalculator


class WarningLevel(Enum):
    """警告级别"""
    CRITICAL = "critical"  # 严重危险，立即报警
    SERIOUS = "serious"    # 严重警告
    NOTICE = "notice"      # 提示纠正


@dataclass
class ActionError:
    """动作错误信息"""
    error_type: str  # 错误类型
    description: str  # 错误描述
    level: WarningLevel  # 警告级别
    score_deduction: float  # 扣分


@dataclass
class ActionAnalysisResult:
    """动作分析结果"""
    overall_score: float  # 总体得分 0-100
    posture_score: float  # 姿势得分
    trigger_discipline_score: float  # 扳机纪律得分
    aim_line_score: float  # 瞄准线得分
    stability_score: float  # 稳定性得分
    is_qualified: bool  # 是否合格（>=80分）
    errors: List[ActionError]  # 错误列表
    warnings: List[str]  # 警告信息


class GunHoldingPostureAnalyzer:
    """持枪姿势分析器"""
    
    # 标准姿势参数范围
    STANDARD_ELBOW_ANGLE = (90, 120)  # 肘部角度范围
    STANDARD_SHOULDER_ANGLE = (30, 45)  # 肩部角度范围
    STANDARD_WRIST_SHOULDER_DISTANCE = (0.4, 0.6)  # 手腕到肩膀距离（相对身高）
    
    def __init__(self):
        self.angle_calc = AngleCalculator()
    
    def analyze(self, keypoints: PoseKeypoints) -> Tuple[float, List[ActionError]]:
        """
        分析持枪姿势
        
        Args:
            keypoints: 姿态关键点
            
        Returns:
            (姿势得分, 错误列表)
        """
        score = 100.0
        errors = []
        
        # 1. 检查右臂肘部角度（假设右手持枪）
        elbow_angle = self.angle_calc.calculate_angle(
            keypoints.right_shoulder,
            keypoints.right_elbow,
            keypoints.right_wrist
        )
        
        if not (self.STANDARD_ELBOW_ANGLE[0] <= elbow_angle <= self.STANDARD_ELBOW_ANGLE[1]):
            if elbow_angle < self.STANDARD_ELBOW_ANGLE[0]:
                error = ActionError(
                    error_type="elbow_angle_too_small",
                    description=f"右肘角度过小({elbow_angle:.1f}°)，应保持在{self.STANDARD_ELBOW_ANGLE[0]}-{self.STANDARD_ELBOW_ANGLE[1]}°",
                    level=WarningLevel.NOTICE,
                    score_deduction=10
                )
            else:
                error = ActionError(
                    error_type="elbow_angle_too_large",
                    description=f"右肘角度过大({elbow_angle:.1f}°)，应保持在{self.STANDARD_ELBOW_ANGLE[0]}-{self.STANDARD_ELBOW_ANGLE[1]}°",
                    level=WarningLevel.NOTICE,
                    score_deduction=10
                )
            errors.append(error)
            score -= error.score_deduction
        
        # 2. 检查左臂支撑位置（辅助手）
        left_elbow_angle = self.angle_calc.calculate_angle(
            keypoints.left_shoulder,
            keypoints.left_elbow,
            keypoints.left_wrist
        )
        
        if left_elbow_angle > 150:  # 左臂过直，支撑不足
            error = ActionError(
                error_type="left_arm_insufficient_support",
                description="左手支撑不足，手臂过直",
                level=WarningLevel.NOTICE,
                score_deduction=8
            )
            errors.append(error)
            score -= error.score_deduction
        
        # 3. 检查双肩水平度
        shoulder_height_diff = abs(
            keypoints.left_shoulder[1] - keypoints.right_shoulder[1]
        )
        
        if shoulder_height_diff > 0.05:  # 肩膀高度差超过5%
            error = ActionError(
                error_type="shoulder_not_level",
                description="双肩不平衡，身体倾斜",
                level=WarningLevel.NOTICE,
                score_deduction=8
            )
            errors.append(error)
            score -= error.score_deduction
        
        # 4. 检查手腕稳定性（相对于肘部）
        wrist_elbow_distance = self.angle_calc.calculate_distance(
            keypoints.right_wrist,
            keypoints.right_elbow
        )
        
        # 手腕距离肘部太远表示手臂过度伸展
        shoulder_elbow_distance = self.angle_calc.calculate_distance(
            keypoints.right_shoulder,
            keypoints.right_elbow
        )
        
        if wrist_elbow_distance > shoulder_elbow_distance * 1.2:
            error = ActionError(
                error_type="arm_overextended",
                description="手臂过度伸展，影响稳定性",
                level=WarningLevel.NOTICE,
                score_deduction=7
            )
            errors.append(error)
            score -= error.score_deduction
        
        return max(0, score), errors


class TriggerDisciplineAnalyzer:
    """扳机纪律分析器"""
    
    # 手指与手腕的距离阈值（判断手指是否在扳机上）
    FINGER_WRIST_DISTANCE_THRESHOLD = 0.15
    
    def __init__(self):
        self.angle_calc = AngleCalculator()
    
    def analyze(
        self,
        keypoints: PoseKeypoints,
        is_aiming: bool = False,
        is_shooting: bool = False
    ) -> Tuple[float, List[ActionError]]:
        """
        分析扳机纪律
        
        Args:
            keypoints: 姿态关键点
            is_aiming: 是否处于瞄准状态
            is_shooting: 是否处于射击状态
            
        Returns:
            (扳机纪律得分, 错误列表)
        """
        score = 100.0
        errors = []
        
        # 计算右手食指与手腕的距离
        index_wrist_distance = self.angle_calc.calculate_distance(
            keypoints.right_index,
            keypoints.right_wrist
        )
        
        # 非射击状态下，手指不应该在扳机附近
        if not is_shooting and not is_aiming:
            # 手指距离手腕很近，可能在扳机上
            if index_wrist_distance < self.FINGER_WRIST_DISTANCE_THRESHOLD:
                error = ActionError(
                    error_type="finger_on_trigger",
                    description="非射击状态时手指接触扳机，严重违反扳机纪律",
                    level=WarningLevel.SERIOUS,
                    score_deduction=50
                )
                errors.append(error)
                score -= error.score_deduction
        
        # 检查食指是否沿枪身放置（应该与手腕形成一定角度）
        if not is_shooting:
            # 理想情况下，食指应该伸直沿枪身
            # 这里通过检查食指-手腕-小指的角度来判断
            finger_angle = self.angle_calc.calculate_angle(
                keypoints.right_index,
                keypoints.right_wrist,
                keypoints.right_pinky
            )
            
            # 如果角度接近180度，说明食指伸直
            if finger_angle < 160:
                error = ActionError(
                    error_type="finger_not_extended",
                    description="手指未沿枪身伸直放置",
                    level=WarningLevel.NOTICE,
                    score_deduction=15
                )
                errors.append(error)
                score -= error.score_deduction
        
        return max(0, score), errors


class AimLineAnalyzer:
    """瞄准线分析器"""
    
    # 眼睛与手部的高度差阈值
    EYE_HAND_HEIGHT_THRESHOLD = 0.1
    
    def __init__(self):
        self.angle_calc = AngleCalculator()
    
    def analyze(self, keypoints: PoseKeypoints) -> Tuple[float, List[ActionError]]:
        """
        分析瞄准线
        
        Args:
            keypoints: 姿态关键点
            
        Returns:
            (瞄准线得分, 错误列表)
        """
        score = 100.0
        errors = []
        
        # 1. 检查头部位置（眼睛应该在手的上方，形成瞄准线）
        # 计算双眼中点
        eye_center_y = (keypoints.left_eye[1] + keypoints.right_eye[1]) / 2
        
        # 计算手腕高度
        wrist_y = keypoints.right_wrist[1]
        
        # 眼睛应该高于手腕（y坐标在图像中是从上到下增加的）
        if eye_center_y > wrist_y:
            error = ActionError(
                error_type="head_position_low",
                description="头部位置过低，无法形成正确的瞄准线",
                level=WarningLevel.SERIOUS,
                score_deduction=20
            )
            errors.append(error)
            score -= error.score_deduction
        
        # 2. 检查头部稳定性（双眼应该水平）
        eye_height_diff = abs(keypoints.left_eye[1] - keypoints.right_eye[1])
        
        if eye_height_diff > 0.03:  # 眼睛高度差超过3%
            error = ActionError(
                error_type="head_tilted",
                description="头部倾斜，影响瞄准精度",
                level=WarningLevel.NOTICE,
                score_deduction=10
            )
            errors.append(error)
            score -= error.score_deduction
        
        # 3. 检查肩-手-眼是否在一条线上（侧面视角）
        # 这需要侧面摄像头的数据，这里简化处理
        shoulder_wrist_vertical_distance = abs(
            keypoints.right_shoulder[0] - keypoints.right_wrist[0]
        )
        
        if shoulder_wrist_vertical_distance > 0.3:
            error = ActionError(
                error_type="arm_not_aligned",
                description="手臂与身体未对齐，影响瞄准稳定性",
                level=WarningLevel.NOTICE,
                score_deduction=8
            )
            errors.append(error)
            score -= error.score_deduction
        
        return max(0, score), errors


class StabilityAnalyzer:
    """稳定性分析器"""
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: 用于计算稳定性的历史帧数
        """
        self.window_size = window_size
        self.history_keypoints = []
        self.angle_calc = AngleCalculator()
    
    def add_keypoints(self, keypoints: PoseKeypoints):
        """添加关键点到历史记录"""
        self.history_keypoints.append(keypoints)
        if len(self.history_keypoints) > self.window_size:
            self.history_keypoints.pop(0)
    
    def analyze(self) -> Tuple[float, List[ActionError]]:
        """
        分析动作稳定性
        
        Returns:
            (稳定性得分, 错误列表)
        """
        score = 100.0
        errors = []
        
        if len(self.history_keypoints) < 10:
            # 数据不足，无法判断
            return score, errors
        
        # 1. 计算手腕位置的抖动（标准差）
        wrist_positions = np.array([
            [kp.right_wrist[0], kp.right_wrist[1]]
            for kp in self.history_keypoints
        ])
        
        wrist_std = np.std(wrist_positions, axis=0).mean()
        
        if wrist_std > 0.02:  # 抖动超过2%
            error = ActionError(
                error_type="hand_shaking",
                description=f"手部抖动严重(抖动度: {wrist_std:.4f})",
                level=WarningLevel.NOTICE,
                score_deduction=15
            )
            errors.append(error)
            score -= error.score_deduction
        elif wrist_std > 0.01:
            error = ActionError(
                error_type="hand_slight_shaking",
                description=f"手部轻微抖动(抖动度: {wrist_std:.4f})",
                level=WarningLevel.NOTICE,
                score_deduction=8
            )
            errors.append(error)
            score -= error.score_deduction
        
        # 2. 计算身体重心稳定性（通过髋部中点判断）
        hip_centers = np.array([
            [(kp.left_hip[0] + kp.right_hip[0]) / 2,
             (kp.left_hip[1] + kp.right_hip[1]) / 2]
            for kp in self.history_keypoints
        ])
        
        hip_std = np.std(hip_centers, axis=0).mean()
        
        if hip_std > 0.015:  # 身体晃动超过1.5%
            error = ActionError(
                error_type="body_unstable",
                description="身体重心不稳定，晃动明显",
                level=WarningLevel.NOTICE,
                score_deduction=12
            )
            errors.append(error)
            score -= error.score_deduction
        
        return max(0, score), errors
    
    def reset(self):
        """重置历史记录"""
        self.history_keypoints.clear()


class ActionAnalyzer:
    """综合动作分析器"""
    
    def __init__(self):
        self.posture_analyzer = GunHoldingPostureAnalyzer()
        self.trigger_analyzer = TriggerDisciplineAnalyzer()
        self.aim_analyzer = AimLineAnalyzer()
        self.stability_analyzer = StabilityAnalyzer()
    
    def analyze(
        self,
        keypoints: PoseKeypoints,
        is_aiming: bool = False,
        is_shooting: bool = False
    ) -> ActionAnalysisResult:
        """
        综合分析动作规范性
        
        Args:
            keypoints: 姿态关键点
            is_aiming: 是否处于瞄准状态
            is_shooting: 是否处于射击状态
            
        Returns:
            ActionAnalysisResult
        """
        # 添加到稳定性分析历史
        self.stability_analyzer.add_keypoints(keypoints)
        
        # 各项分析
        posture_score, posture_errors = self.posture_analyzer.analyze(keypoints)
        trigger_score, trigger_errors = self.trigger_analyzer.analyze(
            keypoints, is_aiming, is_shooting
        )
        aim_score, aim_errors = self.aim_analyzer.analyze(keypoints)
        stability_score, stability_errors = self.stability_analyzer.analyze()
        
        # 合并所有错误
        all_errors = posture_errors + trigger_errors + aim_errors + stability_errors
        
        # 计算总分（加权平均）
        overall_score = (
            posture_score * 0.3 +
            trigger_score * 0.3 +
            aim_score * 0.2 +
            stability_score * 0.2
        )
        
        # 判断是否合格
        is_qualified = overall_score >= 80 and trigger_score >= 70
        
        # 生成警告信息
        warnings = [
            f"[{error.level.value.upper()}] {error.description}"
            for error in all_errors
        ]
        
        return ActionAnalysisResult(
            overall_score=round(overall_score, 2),
            posture_score=round(posture_score, 2),
            trigger_discipline_score=round(trigger_score, 2),
            aim_line_score=round(aim_score, 2),
            stability_score=round(stability_score, 2),
            is_qualified=is_qualified,
            errors=all_errors,
            warnings=warnings
        )
    
    def reset_stability_history(self):
        """重置稳定性历史数据（用于新一轮训练）"""
        self.stability_analyzer.reset()


# 使用示例
if __name__ == "__main__":
    from .pose_detector import PoseDetector
    import cv2
    
    # 初始化
    pose_detector = PoseDetector()
    action_analyzer = ActionAnalyzer()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测姿态
            keypoints = pose_detector.detect(frame)
            
            if keypoints:
                # 分析动作
                result = action_analyzer.analyze(keypoints)
                
                # 绘制关键点
                annotated_frame = pose_detector.draw_landmarks(frame, keypoints)
                
                # 显示分析结果
                y_offset = 30
                cv2.putText(
                    annotated_frame,
                    f"Overall Score: {result.overall_score:.1f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if result.is_qualified else (0, 0, 255),
                    2
                )
                
                y_offset += 30
                cv2.putText(
                    annotated_frame,
                    f"Posture: {result.posture_score:.1f} | Trigger: {result.trigger_discipline_score:.1f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                y_offset += 25
                cv2.putText(
                    annotated_frame,
                    f"Aim: {result.aim_line_score:.1f} | Stability: {result.stability_score:.1f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                # 显示错误信息
                y_offset += 35
                for error in result.errors[:3]:  # 只显示前3个错误
                    color = (0, 0, 255) if error.level == WarningLevel.SERIOUS else (0, 165, 255)
                    cv2.putText(
                        annotated_frame,
                        f"- {error.description[:50]}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
                    y_offset += 25
                
                cv2.imshow('Action Analysis', annotated_frame)
            else:
                cv2.imshow('Action Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_detector.close()

