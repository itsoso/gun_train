"""
实时动作分析系统单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.realtime_analyzer import (
    FeedbackType,
    RealTimeFeedback,
    WorkstationState,
    ImprovementAdvisor,
    RealtimeAnalysisEngine
)
from backend.ai.action_analyzer import (
    ActionAnalysisResult,
    ActionError,
    WarningLevel
)


class TestFeedbackType:
    """测试反馈类型枚举"""
    
    def test_feedback_type_values(self):
        """测试反馈类型值"""
        assert FeedbackType.SCORE_UPDATE.value == "score_update"
        assert FeedbackType.ERROR_ALERT.value == "error_alert"
        assert FeedbackType.DANGER_WARNING.value == "danger_warning"
        assert FeedbackType.IMPROVEMENT_TIP.value == "improvement_tip"
        assert FeedbackType.ENCOURAGEMENT.value == "encouragement"
        assert FeedbackType.STAGE_COMPLETE.value == "stage_complete"


class TestRealTimeFeedback:
    """测试实时反馈数据类"""
    
    def test_feedback_creation(self):
        """测试反馈对象创建"""
        feedback = RealTimeFeedback(
            timestamp=datetime.now(),
            workstation_id=1,
            student_id=101,
            feedback_type=FeedbackType.SCORE_UPDATE,
            overall_score=85.5,
            posture_score=80.0,
            trigger_score=90.0,
            aim_score=85.0,
            stability_score=82.0,
            message="动作合格"
        )
        
        assert feedback.workstation_id == 1
        assert feedback.student_id == 101
        assert feedback.feedback_type == FeedbackType.SCORE_UPDATE
        assert feedback.overall_score == 85.5
        assert feedback.urgent == False
    
    def test_feedback_with_errors(self):
        """测试带错误信息的反馈"""
        feedback = RealTimeFeedback(
            timestamp=datetime.now(),
            workstation_id=1,
            student_id=101,
            feedback_type=FeedbackType.ERROR_ALERT,
            errors=[
                {"type": "elbow_angle_too_large", "description": "肘部角度过大"},
                {"type": "hand_shaking", "description": "手部抖动"}
            ],
            urgent=True
        )
        
        assert len(feedback.errors) == 2
        assert feedback.urgent == True
    
    def test_feedback_to_dict(self):
        """测试反馈转字典"""
        timestamp = datetime.now()
        feedback = RealTimeFeedback(
            timestamp=timestamp,
            workstation_id=1,
            student_id=101,
            feedback_type=FeedbackType.SCORE_UPDATE,
            overall_score=85.0,
            message="测试消息"
        )
        
        result = feedback.to_dict()
        
        assert isinstance(result, dict)
        assert result["workstation_id"] == 1
        assert result["student_id"] == 101
        assert result["feedback_type"] == "score_update"
        assert result["overall_score"] == 85.0
        assert result["message"] == "测试消息"
        assert result["timestamp"] == timestamp.isoformat()
    
    def test_feedback_default_values(self):
        """测试反馈默认值"""
        feedback = RealTimeFeedback(
            timestamp=datetime.now(),
            workstation_id=1,
            student_id=101,
            feedback_type=FeedbackType.SCORE_UPDATE
        )
        
        assert feedback.overall_score is None
        assert feedback.errors == []
        assert feedback.improvements == []
        assert feedback.message == ""
        assert feedback.urgent == False


class TestWorkstationState:
    """测试工位状态类"""
    
    def test_state_creation(self):
        """测试状态对象创建"""
        state = WorkstationState(
            workstation_id=1,
            student_id=101,
            student_name="张三",
            is_active=True
        )
        
        assert state.workstation_id == 1
        assert state.student_id == 101
        assert state.student_name == "张三"
        assert state.is_active == True
        assert state.total_analyses == 0
        assert state.passed_count == 0
        assert state.consecutive_passes == 0
        assert state.consecutive_fails == 0
    
    def test_add_passing_analysis(self):
        """测试添加合格分析结果"""
        state = WorkstationState(workstation_id=1)
        
        result = ActionAnalysisResult(
            overall_score=85.0,
            posture_score=82.0,
            trigger_discipline_score=88.0,
            aim_line_score=85.0,
            stability_score=80.0,
            is_qualified=True,
            errors=[],
            warnings=[]
        )
        
        state.add_analysis(result)
        
        assert state.total_analyses == 1
        assert state.passed_count == 1
        assert state.consecutive_passes == 1
        assert state.consecutive_fails == 0
        assert len(state.score_history) == 1
        assert state.score_history[0] == 85.0
    
    def test_add_failing_analysis(self):
        """测试添加不合格分析结果"""
        state = WorkstationState(workstation_id=1)
        
        error = ActionError(
            error_type="elbow_angle_too_large",
            description="肘部角度过大",
            level=WarningLevel.NOTICE,
            score_deduction=10
        )
        
        result = ActionAnalysisResult(
            overall_score=70.0,
            posture_score=65.0,
            trigger_discipline_score=75.0,
            aim_line_score=70.0,
            stability_score=70.0,
            is_qualified=False,
            errors=[error],
            warnings=[]
        )
        
        state.add_analysis(result)
        
        assert state.total_analyses == 1
        assert state.passed_count == 0
        assert state.consecutive_passes == 0
        assert state.consecutive_fails == 1
        assert state.error_counts["elbow_angle_too_large"] == 1
    
    def test_consecutive_tracking(self):
        """测试连续状态跟踪"""
        state = WorkstationState(workstation_id=1)
        
        # 添加3次合格
        for _ in range(3):
            result = ActionAnalysisResult(
                overall_score=85.0, posture_score=85.0,
                trigger_discipline_score=85.0, aim_line_score=85.0,
                stability_score=85.0, is_qualified=True,
                errors=[], warnings=[]
            )
            state.add_analysis(result)
        
        assert state.consecutive_passes == 3
        assert state.consecutive_fails == 0
        
        # 添加1次不合格
        result = ActionAnalysisResult(
            overall_score=70.0, posture_score=70.0,
            trigger_discipline_score=70.0, aim_line_score=70.0,
            stability_score=70.0, is_qualified=False,
            errors=[], warnings=[]
        )
        state.add_analysis(result)
        
        assert state.consecutive_passes == 0
        assert state.consecutive_fails == 1
    
    def test_get_average_score(self):
        """测试平均分计算"""
        state = WorkstationState(workstation_id=1)
        
        # 添加分数：80, 85, 90
        for score in [80.0, 85.0, 90.0]:
            result = ActionAnalysisResult(
                overall_score=score, posture_score=score,
                trigger_discipline_score=score, aim_line_score=score,
                stability_score=score, is_qualified=True,
                errors=[], warnings=[]
            )
            state.add_analysis(result)
        
        avg = state.get_average_score()
        
        assert abs(avg - 85.0) < 0.01
    
    def test_get_average_score_last_n(self):
        """测试最近N次平均分"""
        state = WorkstationState(workstation_id=1)
        
        # 添加分数：70, 75, 80, 85, 90
        for score in [70.0, 75.0, 80.0, 85.0, 90.0]:
            result = ActionAnalysisResult(
                overall_score=score, posture_score=score,
                trigger_discipline_score=score, aim_line_score=score,
                stability_score=score, is_qualified=True,
                errors=[], warnings=[]
            )
            state.add_analysis(result)
        
        # 最近3次平均：(80+85+90)/3 = 85
        avg = state.get_average_score(last_n=3)
        
        assert abs(avg - 85.0) < 0.01
    
    def test_get_top_errors(self):
        """测试最常见错误获取"""
        state = WorkstationState(workstation_id=1)
        
        # 手动设置错误计数
        state.error_counts = {
            "elbow_angle_too_large": 10,
            "hand_shaking": 8,
            "head_tilted": 5,
            "finger_on_trigger": 2
        }
        
        top = state.get_top_errors(3)
        
        assert len(top) == 3
        assert top[0] == ("elbow_angle_too_large", 10)
        assert top[1] == ("hand_shaking", 8)
        assert top[2] == ("head_tilted", 5)
    
    def test_get_pass_rate(self):
        """测试通过率计算"""
        state = WorkstationState(workstation_id=1)
        state.total_analyses = 10
        state.passed_count = 7
        
        rate = state.get_pass_rate()
        
        assert rate == 70.0
    
    def test_get_pass_rate_empty(self):
        """测试空状态通过率"""
        state = WorkstationState(workstation_id=1)
        
        rate = state.get_pass_rate()
        
        assert rate == 0.0
    
    def test_score_history_limit(self):
        """测试分数历史限制"""
        state = WorkstationState(workstation_id=1)
        state.max_history_size = 10
        
        # 添加15个分数
        for i in range(15):
            result = ActionAnalysisResult(
                overall_score=float(i * 5 + 50), posture_score=80.0,
                trigger_discipline_score=80.0, aim_line_score=80.0,
                stability_score=80.0, is_qualified=True,
                errors=[], warnings=[]
            )
            state.add_analysis(result)
        
        # 应该只保留最近10个
        assert len(state.score_history) == 10


class TestImprovementAdvisor:
    """测试改进建议生成器"""
    
    def test_get_improvements_single_error(self):
        """测试单个错误的改进建议"""
        error = ActionError(
            error_type="elbow_angle_too_large",
            description="肘部角度过大",
            level=WarningLevel.NOTICE,
            score_deduction=10
        )
        
        result = ActionAnalysisResult(
            overall_score=75.0, posture_score=70.0,
            trigger_discipline_score=80.0, aim_line_score=75.0,
            stability_score=75.0, is_qualified=False,
            errors=[error], warnings=[]
        )
        
        state = WorkstationState(workstation_id=1)
        
        improvements = ImprovementAdvisor.get_improvements(result, state)
        
        assert len(improvements) >= 1
    
    def test_get_improvements_trigger_discipline(self):
        """测试扳机纪律错误的改进建议"""
        error = ActionError(
            error_type="finger_on_trigger",
            description="非射击时手指在扳机上",
            level=WarningLevel.SERIOUS,
            score_deduction=50
        )
        
        result = ActionAnalysisResult(
            overall_score=50.0, posture_score=80.0,
            trigger_discipline_score=50.0, aim_line_score=75.0,
            stability_score=75.0, is_qualified=False,
            errors=[error], warnings=[]
        )
        
        state = WorkstationState(workstation_id=1)
        
        improvements = ImprovementAdvisor.get_improvements(result, state)
        
        assert len(improvements) >= 1
        assert any("扳机" in tip for tip in improvements)
    
    def test_get_audio_message_critical(self):
        """测试严重错误语音消息"""
        error = ActionError(
            error_type="muzzle_direction",
            description="枪口指向非安全区域",
            level=WarningLevel.CRITICAL,
            score_deduction=100
        )
        
        result = ActionAnalysisResult(
            overall_score=0.0, posture_score=0.0,
            trigger_discipline_score=0.0, aim_line_score=0.0,
            stability_score=0.0, is_qualified=False,
            errors=[error], warnings=[]
        )
        
        state = WorkstationState(workstation_id=1)
        
        message = ImprovementAdvisor.get_audio_message(result, state)
        
        assert "警告" in message
        assert "枪口" in message
    
    def test_get_audio_message_qualified(self):
        """测试合格时语音消息"""
        result = ActionAnalysisResult(
            overall_score=85.0, posture_score=85.0,
            trigger_discipline_score=85.0, aim_line_score=85.0,
            stability_score=85.0, is_qualified=True,
            errors=[], warnings=[]
        )
        
        state = WorkstationState(workstation_id=1)
        state.consecutive_passes = 5
        
        message = ImprovementAdvisor.get_audio_message(result, state)
        
        # 应该有鼓励或达标消息
        assert len(message) > 0
    
    def test_get_encouragement(self):
        """测试鼓励语生成"""
        state = WorkstationState(workstation_id=1)
        state.consecutive_passes = 5
        
        encouragement = ImprovementAdvisor.get_encouragement(state)
        
        assert len(encouragement) > 0
        assert "实弹" in encouragement or "达标" in encouragement or "棒" in encouragement
    
    def test_get_encouragement_in_progress(self):
        """测试进行中鼓励语"""
        state = WorkstationState(workstation_id=1)
        state.consecutive_passes = 3
        
        encouragement = ImprovementAdvisor.get_encouragement(state)
        
        assert len(encouragement) > 0
        assert "2" in encouragement  # 还差2次


class TestRealtimeAnalysisEngine:
    """测试实时分析引擎"""
    
    @pytest.fixture
    def mock_camera_manager(self):
        """Mock摄像头管理器"""
        manager = MagicMock()
        manager.get_workstation_frames.return_value = {}
        return manager
    
    def test_engine_initialization(self, mock_camera_manager):
        """测试引擎初始化"""
        with patch('backend.core.realtime_analyzer.PoseDetector'), \
             patch('backend.core.realtime_analyzer.ActionAnalyzer'), \
             patch('backend.core.realtime_analyzer.AngleCalculator'):
            engine = RealtimeAnalysisEngine(
                camera_manager=mock_camera_manager,
                analysis_interval=0.5
            )
            
            assert engine.camera_manager == mock_camera_manager
            assert engine.analysis_interval == 0.5
            assert engine.is_running == False
            assert len(engine.workstation_states) == 0
    
    def test_register_student(self, mock_camera_manager):
        """测试注册学员"""
        with patch('backend.core.realtime_analyzer.PoseDetector'), \
             patch('backend.core.realtime_analyzer.ActionAnalyzer'), \
             patch('backend.core.realtime_analyzer.AngleCalculator'):
            engine = RealtimeAnalysisEngine(camera_manager=mock_camera_manager)
            
            engine.register_student(
                workstation_id=1,
                student_id=101,
                student_name="张三"
            )
            
            assert 1 in engine.workstation_states
            assert engine.workstation_states[1].student_id == 101
            assert engine.workstation_states[1].student_name == "张三"
            assert engine.workstation_states[1].is_active == True
            assert 101 in engine.student_workstation_map
            assert engine.student_workstation_map[101] == 1
    
    def test_unregister_student(self, mock_camera_manager):
        """测试取消注册学员"""
        with patch('backend.core.realtime_analyzer.PoseDetector'), \
             patch('backend.core.realtime_analyzer.ActionAnalyzer'), \
             patch('backend.core.realtime_analyzer.AngleCalculator'):
            engine = RealtimeAnalysisEngine(camera_manager=mock_camera_manager)
            
            engine.register_student(workstation_id=1, student_id=101, student_name="张三")
            engine.unregister_student(1)
            
            assert engine.workstation_states[1].is_active == False
            assert engine.workstation_states[1].student_id is None
            assert 101 not in engine.student_workstation_map
    
    def test_get_workstation_state(self, mock_camera_manager):
        """测试获取工位状态"""
        with patch('backend.core.realtime_analyzer.PoseDetector'), \
             patch('backend.core.realtime_analyzer.ActionAnalyzer'), \
             patch('backend.core.realtime_analyzer.AngleCalculator'):
            engine = RealtimeAnalysisEngine(camera_manager=mock_camera_manager)
            
            engine.register_student(workstation_id=5, student_id=105, student_name="李四")
            
            state = engine.get_workstation_state(5)
            
            assert state is not None
            assert state.workstation_id == 5
            assert state.student_id == 105
    
    def test_get_student_state(self, mock_camera_manager):
        """测试通过学员ID获取状态"""
        with patch('backend.core.realtime_analyzer.PoseDetector'), \
             patch('backend.core.realtime_analyzer.ActionAnalyzer'), \
             patch('backend.core.realtime_analyzer.AngleCalculator'):
            engine = RealtimeAnalysisEngine(camera_manager=mock_camera_manager)
            
            engine.register_student(workstation_id=5, student_id=105, student_name="李四")
            
            state = engine.get_student_state(105)
            
            assert state is not None
            assert state.student_id == 105
            assert state.workstation_id == 5
    
    def test_get_all_states(self, mock_camera_manager):
        """测试获取所有工位状态"""
        with patch('backend.core.realtime_analyzer.PoseDetector'), \
             patch('backend.core.realtime_analyzer.ActionAnalyzer'), \
             patch('backend.core.realtime_analyzer.AngleCalculator'):
            engine = RealtimeAnalysisEngine(camera_manager=mock_camera_manager)
            
            engine.register_student(workstation_id=1, student_id=101, student_name="张三")
            engine.register_student(workstation_id=2, student_id=102, student_name="李四")
            
            states = engine.get_all_states()
            
            assert len(states) == 2
            assert 1 in states
            assert 2 in states
            assert states[1]["student_name"] == "张三"
            assert states[2]["student_name"] == "李四"
    
    def test_get_stats(self, mock_camera_manager):
        """测试获取统计信息"""
        with patch('backend.core.realtime_analyzer.PoseDetector'), \
             patch('backend.core.realtime_analyzer.ActionAnalyzer'), \
             patch('backend.core.realtime_analyzer.AngleCalculator'):
            engine = RealtimeAnalysisEngine(camera_manager=mock_camera_manager)
            
            engine.register_student(workstation_id=1, student_id=101, student_name="张三")
            
            stats = engine.get_stats()
            
            assert "is_running" in stats
            assert "total_analyses" in stats
            assert "active_workstations" in stats
            assert stats["is_running"] == False
            assert stats["active_workstations"] == 1


class TestIntegration:
    """集成测试"""
    
    def test_full_training_session(self):
        """测试完整训练会话"""
        # 创建工位状态
        state = WorkstationState(
            workstation_id=1,
            student_id=101,
            student_name="测试学员",
            is_active=True
        )
        
        # 模拟5次合格训练
        for _ in range(5):
            result = ActionAnalysisResult(
                overall_score=85.0,
                posture_score=85.0,
                trigger_discipline_score=85.0,
                aim_line_score=85.0,
                stability_score=85.0,
                is_qualified=True,
                errors=[],
                warnings=[]
            )
            state.add_analysis(result)
        
        # 验证状态
        assert state.consecutive_passes == 5
        assert state.passed_count == 5
        assert state.get_pass_rate() == 100.0
        assert abs(state.get_average_score() - 85.0) < 0.01
    
    def test_training_with_errors(self):
        """测试有错误的训练会话"""
        state = WorkstationState(workstation_id=1)
        
        # 模拟混合结果
        error = ActionError(
            error_type="hand_shaking",
            description="手部抖动",
            level=WarningLevel.NOTICE,
            score_deduction=5
        )
        
        # 3次合格
        for _ in range(3):
            result = ActionAnalysisResult(
                overall_score=85.0, posture_score=85.0,
                trigger_discipline_score=85.0, aim_line_score=85.0,
                stability_score=85.0, is_qualified=True,
                errors=[], warnings=[]
            )
            state.add_analysis(result)
        
        # 2次不合格
        for _ in range(2):
            result = ActionAnalysisResult(
                overall_score=70.0, posture_score=70.0,
                trigger_discipline_score=70.0, aim_line_score=70.0,
                stability_score=70.0, is_qualified=False,
                errors=[error], warnings=[]
            )
            state.add_analysis(result)
        
        # 验证状态
        assert state.total_analyses == 5
        assert state.passed_count == 3
        assert state.consecutive_passes == 0
        assert state.consecutive_fails == 2
        assert state.get_pass_rate() == 60.0
        assert state.error_counts["hand_shaking"] == 2


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
