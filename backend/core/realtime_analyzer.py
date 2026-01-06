"""
实时动作分析系统
集成摄像头、AI分析、反馈生成
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import queue
import time
import logging
import json

from ..camera.hikvision import (
    HikvisionCameraManager, 
    HikvisionCameraConfig,
    CameraFrame,
    generate_camera_configs
)
from ..ai.pose_detector import PoseDetector, PoseKeypoints, AngleCalculator
from ..ai.action_analyzer import (
    ActionAnalyzer, 
    ActionAnalysisResult,
    WarningLevel,
    ActionError
)

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型"""
    SCORE_UPDATE = "score_update"           # 分数更新
    ERROR_ALERT = "error_alert"             # 错误提示
    DANGER_WARNING = "danger_warning"       # 危险预警
    IMPROVEMENT_TIP = "improvement_tip"     # 改进建议
    ENCOURAGEMENT = "encouragement"         # 鼓励信息
    STAGE_COMPLETE = "stage_complete"       # 阶段完成


@dataclass
class RealTimeFeedback:
    """实时反馈数据"""
    timestamp: datetime
    workstation_id: int
    student_id: int
    feedback_type: FeedbackType
    
    # 分数信息
    overall_score: Optional[float] = None
    posture_score: Optional[float] = None
    trigger_score: Optional[float] = None
    aim_score: Optional[float] = None
    stability_score: Optional[float] = None
    
    # 错误信息
    errors: List[Dict] = field(default_factory=list)
    
    # 反馈消息
    message: str = ""
    audio_message: str = ""  # 语音播报内容
    
    # 改进建议
    improvements: List[str] = field(default_factory=list)
    
    # 是否需要立即处理
    urgent: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "workstation_id": self.workstation_id,
            "student_id": self.student_id,
            "feedback_type": self.feedback_type.value,
            "overall_score": self.overall_score,
            "posture_score": self.posture_score,
            "trigger_score": self.trigger_score,
            "aim_score": self.aim_score,
            "stability_score": self.stability_score,
            "errors": self.errors,
            "message": self.message,
            "audio_message": self.audio_message,
            "improvements": self.improvements,
            "urgent": self.urgent
        }


@dataclass
class WorkstationState:
    """工位状态"""
    workstation_id: int
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    is_active: bool = False
    
    # 最新分析结果
    last_analysis: Optional[ActionAnalysisResult] = None
    last_analysis_time: Optional[datetime] = None
    
    # 历史得分（用于趋势分析）
    score_history: List[float] = field(default_factory=list)
    max_history_size: int = 100
    
    # 累计统计
    total_analyses: int = 0
    passed_count: int = 0
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    # 连续状态
    consecutive_passes: int = 0
    consecutive_fails: int = 0
    
    def add_analysis(self, result: ActionAnalysisResult):
        """添加分析结果"""
        self.last_analysis = result
        self.last_analysis_time = datetime.now()
        self.total_analyses += 1
        
        # 更新得分历史
        self.score_history.append(result.overall_score)
        if len(self.score_history) > self.max_history_size:
            self.score_history.pop(0)
        
        # 更新连续状态
        if result.is_qualified:
            self.passed_count += 1
            self.consecutive_passes += 1
            self.consecutive_fails = 0
        else:
            self.consecutive_fails += 1
            self.consecutive_passes = 0
        
        # 统计错误类型
        for error in result.errors:
            error_type = error.error_type
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_average_score(self, last_n: int = 10) -> float:
        """获取最近N次平均分"""
        if not self.score_history:
            return 0.0
        
        recent = self.score_history[-last_n:]
        return sum(recent) / len(recent)
    
    def get_top_errors(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """获取最常见的错误"""
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_errors[:top_n]
    
    def get_pass_rate(self) -> float:
        """获取通过率"""
        if self.total_analyses == 0:
            return 0.0
        return self.passed_count / self.total_analyses * 100


class ImprovementAdvisor:
    """改进建议生成器"""
    
    # 错误类型到改进建议的映射
    ERROR_TO_IMPROVEMENT = {
        "elbow_angle_too_small": [
            "放松手臂，肘部自然弯曲",
            "肘部角度应保持在90-120度之间",
            "不要过度弯曲手臂，保持舒适的持枪姿势"
        ],
        "elbow_angle_too_large": [
            "手臂不要过度伸直",
            "适当弯曲肘部，增加稳定性",
            "放松肩部，让手臂自然弯曲"
        ],
        "left_arm_insufficient_support": [
            "左手应该有力地托住枪身",
            "左臂弯曲，提供稳定支撑",
            "双手协调配合，左手是重要的辅助"
        ],
        "shoulder_not_level": [
            "保持双肩水平",
            "不要耸肩或歪斜",
            "正面朝向目标，身体放松"
        ],
        "arm_overextended": [
            "不要把手臂伸得太直",
            "保持适度弯曲增加稳定性",
            "想象肘部有一个缓冲区"
        ],
        "finger_on_trigger": [
            "⚠️ 非射击时手指必须离开扳机！",
            "食指应该沿枪身伸直放置",
            "这是最重要的安全规则，请牢记"
        ],
        "finger_not_extended": [
            "食指沿枪身伸直放置",
            "手指位置是扳机纪律的关键",
            "养成正确的手指位置习惯"
        ],
        "head_position_low": [
            "抬起头部，眼睛与准星对齐",
            "保持正确的瞄准姿势",
            "头部位置影响瞄准精度"
        ],
        "head_tilted": [
            "保持头部正直",
            "不要歪头瞄准",
            "头部稳定有助于提高命中率"
        ],
        "arm_not_aligned": [
            "手臂与身体保持对齐",
            "正面朝向目标",
            "保持自然的持枪姿势"
        ],
        "hand_shaking": [
            "深呼吸，放松手部肌肉",
            "加强握力训练",
            "在呼气末端扣扳机",
            "多练习静态持枪，增强耐力"
        ],
        "hand_slight_shaking": [
            "轻微抖动是正常的",
            "通过练习可以改善稳定性",
            "注意呼吸节奏"
        ],
        "body_unstable": [
            "调整站姿，双脚与肩同宽",
            "重心略微前倾",
            "保持下盘稳定"
        ]
    }
    
    # 鼓励语
    ENCOURAGEMENTS = [
        "做得好！继续保持！",
        "进步明显，再接再厉！",
        "姿势标准，非常棒！",
        "完美的动作！",
        "你的训练效果很好！",
        "继续努力，马上就达标了！"
    ]
    
    @classmethod
    def get_improvements(
        cls,
        result: ActionAnalysisResult,
        state: WorkstationState
    ) -> List[str]:
        """
        根据分析结果生成改进建议
        
        Args:
            result: 分析结果
            state: 工位状态
            
        Returns:
            改进建议列表
        """
        improvements = []
        
        # 根据错误生成建议
        for error in result.errors:
            error_type = error.error_type
            if error_type in cls.ERROR_TO_IMPROVEMENT:
                tips = cls.ERROR_TO_IMPROVEMENT[error_type]
                # 选择一个建议（可以根据出现次数轮换）
                tip_index = state.error_counts.get(error_type, 0) % len(tips)
                improvements.append(tips[tip_index])
        
        # 根据最常见错误添加针对性建议
        top_errors = state.get_top_errors(2)
        for error_type, count in top_errors:
            if count >= 3 and error_type in cls.ERROR_TO_IMPROVEMENT:
                tips = cls.ERROR_TO_IMPROVEMENT[error_type]
                improvements.append(f"💡 重点改进：{tips[0]}")
        
        # 去重
        improvements = list(dict.fromkeys(improvements))
        
        return improvements[:5]  # 最多返回5条
    
    @classmethod
    def get_audio_message(
        cls,
        result: ActionAnalysisResult,
        state: WorkstationState
    ) -> str:
        """
        生成语音播报内容
        
        Args:
            result: 分析结果
            state: 工位状态
            
        Returns:
            语音播报文本
        """
        messages = []
        
        # 危险警告优先
        critical_errors = [e for e in result.errors if e.level == WarningLevel.CRITICAL]
        if critical_errors:
            return f"警告！{critical_errors[0].description}"
        
        # 严重错误
        serious_errors = [e for e in result.errors if e.level == WarningLevel.SERIOUS]
        if serious_errors:
            messages.append(serious_errors[0].description)
        
        # 分数反馈
        if result.is_qualified:
            if state.consecutive_passes >= 5:
                messages.append("连续五次达标，可以申请实弹训练")
            elif state.consecutive_passes >= 3:
                messages.append(f"连续{state.consecutive_passes}次达标，继续保持")
            else:
                import random
                messages.append(random.choice(cls.ENCOURAGEMENTS))
        else:
            # 找出最大问题
            if result.posture_score < 70:
                messages.append("注意持枪姿势")
            elif result.trigger_discipline_score < 70:
                messages.append("检查手指位置")
            elif result.aim_line_score < 70:
                messages.append("调整瞄准姿势")
            elif result.stability_score < 70:
                messages.append("保持稳定")
        
        return "。".join(messages) if messages else ""
    
    @classmethod
    def get_encouragement(cls, state: WorkstationState) -> str:
        """获取鼓励语"""
        import random
        
        # 根据状态选择鼓励语
        if state.consecutive_passes >= 5:
            return "🎉 太棒了！你已经达到实弹训练标准！"
        elif state.consecutive_passes >= 3:
            return f"👏 表现出色！还差{5 - state.consecutive_passes}次就达标了！"
        elif state.get_pass_rate() > 70:
            return random.choice(cls.ENCOURAGEMENTS)
        else:
            return "💪 加油！多练习一定能掌握！"


class RealtimeAnalysisEngine:
    """实时分析引擎"""
    
    def __init__(
        self,
        camera_manager: HikvisionCameraManager,
        feedback_callback: Optional[Callable[[RealTimeFeedback], None]] = None,
        analysis_interval: float = 0.5  # 分析间隔（秒）
    ):
        """
        Args:
            camera_manager: 摄像头管理器
            feedback_callback: 反馈回调函数
            analysis_interval: 分析间隔
        """
        self.camera_manager = camera_manager
        self.feedback_callback = feedback_callback
        self.analysis_interval = analysis_interval
        
        # 初始化AI模块
        self.pose_detector = PoseDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.action_analyzer = ActionAnalyzer()
        self.angle_calc = AngleCalculator()
        
        # 工位状态
        self.workstation_states: Dict[int, WorkstationState] = {}
        
        # 学员-工位映射
        self.student_workstation_map: Dict[int, int] = {}
        
        # 运行控制
        self.is_running = False
        self.analysis_threads: List[threading.Thread] = []
        
        # 统计
        self.total_analyses = 0
        self.start_time: Optional[datetime] = None
    
    def register_student(
        self,
        workstation_id: int,
        student_id: int,
        student_name: str
    ):
        """注册学员到工位"""
        state = WorkstationState(
            workstation_id=workstation_id,
            student_id=student_id,
            student_name=student_name,
            is_active=True
        )
        
        self.workstation_states[workstation_id] = state
        self.student_workstation_map[student_id] = workstation_id
        
        logger.info(f"📝 学员 {student_name}({student_id}) 注册到工位 {workstation_id}")
    
    def unregister_student(self, workstation_id: int):
        """取消学员注册"""
        if workstation_id in self.workstation_states:
            state = self.workstation_states[workstation_id]
            if state.student_id:
                del self.student_workstation_map[state.student_id]
            
            state.is_active = False
            state.student_id = None
            state.student_name = None
    
    def start(self):
        """启动分析引擎"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # 为每个活跃工位创建分析线程
        for workstation_id, state in self.workstation_states.items():
            if state.is_active:
                thread = threading.Thread(
                    target=self._analysis_loop,
                    args=(workstation_id,),
                    daemon=True
                )
                thread.start()
                self.analysis_threads.append(thread)
        
        logger.info(f"🚀 实时分析引擎启动，{len(self.analysis_threads)} 个工位")
    
    def stop(self):
        """停止分析引擎"""
        self.is_running = False
        
        for thread in self.analysis_threads:
            thread.join(timeout=3)
        
        self.analysis_threads.clear()
        logger.info("⏹️ 实时分析引擎已停止")
    
    def _analysis_loop(self, workstation_id: int):
        """工位分析循环"""
        logger.info(f"🎯 工位 {workstation_id} 分析循环启动")
        
        last_analysis_time = 0
        
        while self.is_running:
            current_time = time.time()
            
            # 控制分析频率
            if current_time - last_analysis_time < self.analysis_interval:
                time.sleep(0.05)
                continue
            
            # 获取工位状态
            state = self.workstation_states.get(workstation_id)
            if not state or not state.is_active:
                time.sleep(0.5)
                continue
            
            try:
                # 获取该工位的视频帧（优先使用正面摄像头）
                frames = self.camera_manager.get_workstation_frames(workstation_id)
                
                frame_data = frames.get("front")
                if not frame_data:
                    frame_data = frames.get("side")
                
                if not frame_data:
                    continue
                
                # 分析帧
                feedback = self._analyze_frame(
                    frame_data.frame,
                    workstation_id,
                    state
                )
                
                if feedback and self.feedback_callback:
                    self.feedback_callback(feedback)
                
                last_analysis_time = current_time
                self.total_analyses += 1
                
            except Exception as e:
                logger.error(f"❌ 工位 {workstation_id} 分析错误: {e}")
                time.sleep(0.5)
    
    def _analyze_frame(
        self,
        frame: np.ndarray,
        workstation_id: int,
        state: WorkstationState
    ) -> Optional[RealTimeFeedback]:
        """
        分析单帧图像
        
        Args:
            frame: 图像帧
            workstation_id: 工位ID
            state: 工位状态
            
        Returns:
            RealTimeFeedback对象
        """
        # 姿态识别
        keypoints = self.pose_detector.detect(frame)
        
        if keypoints is None:
            return None
        
        # 动作分析
        result = self.action_analyzer.analyze(keypoints)
        
        # 更新工位状态
        state.add_analysis(result)
        
        # 生成反馈
        feedback = self._generate_feedback(result, state)
        
        return feedback
    
    def _generate_feedback(
        self,
        result: ActionAnalysisResult,
        state: WorkstationState
    ) -> RealTimeFeedback:
        """生成反馈"""
        
        # 确定反馈类型
        if any(e.level == WarningLevel.CRITICAL for e in result.errors):
            feedback_type = FeedbackType.DANGER_WARNING
            urgent = True
        elif any(e.level == WarningLevel.SERIOUS for e in result.errors):
            feedback_type = FeedbackType.ERROR_ALERT
            urgent = True
        elif result.is_qualified:
            if state.consecutive_passes >= 5:
                feedback_type = FeedbackType.STAGE_COMPLETE
            else:
                feedback_type = FeedbackType.ENCOURAGEMENT
            urgent = False
        else:
            feedback_type = FeedbackType.IMPROVEMENT_TIP
            urgent = False
        
        # 生成改进建议
        improvements = ImprovementAdvisor.get_improvements(result, state)
        
        # 生成语音消息
        audio_message = ImprovementAdvisor.get_audio_message(result, state)
        
        # 生成显示消息
        if result.is_qualified:
            message = f"✅ 动作合格！得分 {result.overall_score:.1f}"
        else:
            message = f"❌ 需要改进，得分 {result.overall_score:.1f}"
        
        # 错误信息
        errors = [
            {
                "type": e.error_type,
                "description": e.description,
                "level": e.level.value,
                "deduction": e.score_deduction
            }
            for e in result.errors
        ]
        
        feedback = RealTimeFeedback(
            timestamp=datetime.now(),
            workstation_id=state.workstation_id,
            student_id=state.student_id or 0,
            feedback_type=feedback_type,
            overall_score=result.overall_score,
            posture_score=result.posture_score,
            trigger_score=result.trigger_discipline_score,
            aim_score=result.aim_line_score,
            stability_score=result.stability_score,
            errors=errors,
            message=message,
            audio_message=audio_message,
            improvements=improvements,
            urgent=urgent
        )
        
        return feedback
    
    def analyze_single_frame(
        self,
        frame: np.ndarray,
        workstation_id: int = 0
    ) -> Optional[ActionAnalysisResult]:
        """
        分析单张图片（用于测试）
        
        Args:
            frame: 图像
            workstation_id: 工位ID
            
        Returns:
            分析结果
        """
        keypoints = self.pose_detector.detect(frame)
        
        if keypoints is None:
            return None
        
        return self.action_analyzer.analyze(keypoints)
    
    def get_workstation_state(self, workstation_id: int) -> Optional[WorkstationState]:
        """获取工位状态"""
        return self.workstation_states.get(workstation_id)
    
    def get_student_state(self, student_id: int) -> Optional[WorkstationState]:
        """获取学员状态"""
        workstation_id = self.student_workstation_map.get(student_id)
        if workstation_id:
            return self.workstation_states.get(workstation_id)
        return None
    
    def get_all_states(self) -> Dict[int, Dict]:
        """获取所有工位状态"""
        states = {}
        for ws_id, state in self.workstation_states.items():
            states[ws_id] = {
                "workstation_id": ws_id,
                "student_id": state.student_id,
                "student_name": state.student_name,
                "is_active": state.is_active,
                "total_analyses": state.total_analyses,
                "passed_count": state.passed_count,
                "pass_rate": round(state.get_pass_rate(), 1),
                "average_score": round(state.get_average_score(), 1),
                "consecutive_passes": state.consecutive_passes,
                "top_errors": state.get_top_errors(3)
            }
        return states
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "is_running": self.is_running,
            "total_analyses": self.total_analyses,
            "active_workstations": sum(
                1 for s in self.workstation_states.values() if s.is_active
            ),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": int(
                (datetime.now() - self.start_time).total_seconds()
            ) if self.start_time else 0
        }


# 使用示例
if __name__ == "__main__":
    import sys
    
    # 反馈回调
    def on_feedback(feedback: RealTimeFeedback):
        print(f"\n{'='*50}")
        print(f"工位 {feedback.workstation_id}: {feedback.feedback_type.value}")
        print(f"得分: {feedback.overall_score:.1f}")
        print(f"  姿势: {feedback.posture_score:.1f}")
        print(f"  扳机: {feedback.trigger_score:.1f}")
        print(f"  瞄准: {feedback.aim_score:.1f}")
        print(f"  稳定: {feedback.stability_score:.1f}")
        
        if feedback.errors:
            print(f"错误:")
            for err in feedback.errors:
                print(f"  - {err['description']}")
        
        if feedback.improvements:
            print(f"改进建议:")
            for tip in feedback.improvements:
                print(f"  💡 {tip}")
        
        if feedback.audio_message:
            print(f"🔊 语音: {feedback.audio_message}")
    
    # 使用本地摄像头测试
    print("🎯 实时动作分析系统测试")
    print("使用本地摄像头进行测试...")
    
    # 简单测试：直接读取摄像头
    from ..ai.pose_detector import PoseDetector
    from ..ai.action_analyzer import ActionAnalyzer
    
    detector = PoseDetector()
    analyzer = ActionAnalyzer()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        sys.exit(1)
    
    print("✅ 摄像头已打开，按 'q' 退出")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测姿态
            keypoints = detector.detect(frame)
            
            if keypoints:
                # 分析动作
                result = analyzer.analyze(keypoints)
                
                # 绘制关键点
                annotated = detector.draw_landmarks(frame, keypoints)
                
                # 显示分数
                y = 30
                color = (0, 255, 0) if result.is_qualified else (0, 0, 255)
                cv2.putText(annotated, f"Score: {result.overall_score:.1f}", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                y += 30
                cv2.putText(annotated, f"Posture: {result.posture_score:.1f}", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                y += 25
                cv2.putText(annotated, f"Trigger: {result.trigger_discipline_score:.1f}", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                y += 25
                cv2.putText(annotated, f"Aim: {result.aim_line_score:.1f}", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                y += 25
                cv2.putText(annotated, f"Stability: {result.stability_score:.1f}", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                # 显示错误
                y += 35
                for error in result.errors[:3]:
                    err_color = (0, 0, 255) if error.level == WarningLevel.SERIOUS else (0, 165, 255)
                    cv2.putText(annotated, f"! {error.description[:40]}", 
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, err_color, 1)
                    y += 22
                
                cv2.imshow("Action Analysis", annotated)
            else:
                cv2.imshow("Action Analysis", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("\n✅ 测试结束")

