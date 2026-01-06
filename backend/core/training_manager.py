"""
训练流程管理模块
管理整个训练流程：空枪训练 -> 考核 -> 实弹训练
"""

from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from sqlalchemy.orm import Session

from ..db.models import (
    TrainingSession, TrainingRecord, ActionAnalysis,
    Warning, User, AmmunitionRecord
)
from ..ai.action_analyzer import ActionAnalysisResult


class TrainingStage(Enum):
    """训练阶段"""
    NOT_STARTED = "not_started"  # 未开始
    DRY_FIRE = "dry_fire"  # 空枪训练
    DRY_FIRE_ASSESSMENT = "dry_fire_assessment"  # 空枪考核
    LIVE_FIRE = "live_fire"  # 实弹训练
    COMPLETED = "completed"  # 已完成


class TrainingStatus(Enum):
    """训练状态"""
    WAITING = "waiting"  # 等待中
    IN_PROGRESS = "in_progress"  # 进行中
    PASSED = "passed"  # 通过
    FAILED = "failed"  # 未通过
    CANCELLED = "cancelled"  # 已取消


@dataclass
class TrainingProgress:
    """训练进度"""
    student_id: int
    student_name: str
    workstation_id: int
    current_stage: TrainingStage
    status: TrainingStatus
    practice_count: int  # 当前阶段练习次数
    passed_count: int  # 当前阶段合格次数
    average_score: float  # 平均分
    last_score: Optional[float]  # 最近一次得分
    qualified_for_next: bool  # 是否可进入下一阶段
    start_time: Optional[datetime]
    elapsed_time: int  # 已用时间（秒）


class TrainingFlowController:
    """训练流程控制器"""
    
    # 空枪训练要求：连续5次合格（>=80分）才能进入实弹
    DRY_FIRE_REQUIRED_PASSES = 5
    PASS_SCORE_THRESHOLD = 80
    
    def __init__(self, db: Session):
        self.db = db
    
    def start_training_session(
        self,
        session_date: datetime.date,
        session_type: str,
        instructor_id: int
    ) -> TrainingSession:
        """
        开始一个训练场次
        
        Args:
            session_date: 训练日期
            session_type: 训练类型 (dry_fire/live_fire)
            instructor_id: 教官ID
            
        Returns:
            TrainingSession对象
        """
        session = TrainingSession(
            session_date=session_date,
            session_type=session_type,
            instructor_id=instructor_id,
            start_time=datetime.now(),
            status="ongoing"
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        return session
    
    def assign_student_to_workstation(
        self,
        session_id: int,
        student_id: int,
        workstation_id: int,
        gun_type: str
    ) -> TrainingRecord:
        """
        分配学员到工位
        
        Args:
            session_id: 场次ID
            student_id: 学员ID
            workstation_id: 工位号
            gun_type: 枪支型号
            
        Returns:
            TrainingRecord对象
        """
        # 检查该工位是否已被占用
        existing = self.db.query(TrainingRecord).filter(
            TrainingRecord.session_id == session_id,
            TrainingRecord.workstation_id == workstation_id,
            TrainingRecord.status.in_(["training", "passed"])
        ).first()
        
        if existing:
            raise ValueError(f"工位 {workstation_id} 已被占用")
        
        # 获取场次信息
        session = self.db.query(TrainingSession).get(session_id)
        
        # 创建训练记录
        record = TrainingRecord(
            session_id=session_id,
            student_id=student_id,
            workstation_id=workstation_id,
            gun_type=gun_type,
            training_type=session.session_type,
            start_time=datetime.now(),
            status="training"
        )
        
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        
        # 更新场次学员数
        session.total_students = self.db.query(TrainingRecord).filter(
            TrainingRecord.session_id == session_id
        ).count()
        self.db.commit()
        
        return record
    
    def record_action_analysis(
        self,
        record_id: int,
        analysis_result: ActionAnalysisResult,
        video_clip_id: Optional[str] = None
    ) -> ActionAnalysis:
        """
        记录动作分析结果
        
        Args:
            record_id: 训练记录ID
            analysis_result: 动作分析结果
            video_clip_id: 视频片段ID
            
        Returns:
            ActionAnalysis对象
        """
        # 转换错误信息为JSON格式
        errors_json = [
            {
                "type": error.error_type,
                "description": error.description,
                "level": error.level.value,
                "score_deduction": error.score_deduction
            }
            for error in analysis_result.errors
        ]
        
        # 创建分析记录
        analysis = ActionAnalysis(
            record_id=record_id,
            timestamp=datetime.now(),
            posture_score=analysis_result.posture_score,
            trigger_discipline_score=analysis_result.trigger_discipline_score,
            aim_line_score=analysis_result.aim_line_score,
            stability_score=analysis_result.stability_score,
            overall_score=analysis_result.overall_score,
            is_qualified=analysis_result.is_qualified,
            errors=errors_json,
            video_clip_id=video_clip_id
        )
        
        self.db.add(analysis)
        self.db.commit()
        
        # 更新训练记录统计
        self._update_training_record_stats(record_id)
        
        return analysis
    
    def record_warning(
        self,
        record_id: int,
        warning_level: str,
        warning_type: str,
        description: str,
        video_clip_id: Optional[str] = None
    ) -> Warning:
        """
        记录警告信息
        
        Args:
            record_id: 训练记录ID
            warning_level: 警告级别 (critical/serious/notice)
            warning_type: 警告类型
            description: 描述
            video_clip_id: 视频片段ID
            
        Returns:
            Warning对象
        """
        warning = Warning(
            record_id=record_id,
            timestamp=datetime.now(),
            warning_level=warning_level,
            warning_type=warning_type,
            description=description,
            video_clip_id=video_clip_id
        )
        
        self.db.add(warning)
        self.db.commit()
        
        return warning
    
    def _update_training_record_stats(self, record_id: int):
        """更新训练记录的统计数据"""
        record = self.db.query(TrainingRecord).get(record_id)
        
        # 获取所有分析记录
        analyses = self.db.query(ActionAnalysis).filter(
            ActionAnalysis.record_id == record_id
        ).all()
        
        if not analyses:
            return
        
        # 更新统计
        record.practice_count = len(analyses)
        record.passed_count = sum(1 for a in analyses if a.is_qualified)
        record.average_score = sum(a.overall_score for a in analyses) / len(analyses)
        
        self.db.commit()
    
    def check_dry_fire_qualification(self, record_id: int) -> Dict:
        """
        检查空枪训练是否达标
        
        Args:
            record_id: 训练记录ID
            
        Returns:
            资格检查结果
        """
        record = self.db.query(TrainingRecord).get(record_id)
        
        # 获取最近的分析记录
        recent_analyses = self.db.query(ActionAnalysis).filter(
            ActionAnalysis.record_id == record_id
        ).order_by(ActionAnalysis.timestamp.desc()).limit(self.DRY_FIRE_REQUIRED_PASSES).all()
        
        # 检查是否有足够的练习次数
        if len(recent_analyses) < self.DRY_FIRE_REQUIRED_PASSES:
            return {
                "qualified": False,
                "reason": f"练习次数不足，需要 {self.DRY_FIRE_REQUIRED_PASSES} 次，当前 {len(recent_analyses)} 次",
                "progress": len(recent_analyses),
                "required": self.DRY_FIRE_REQUIRED_PASSES
            }
        
        # 检查最近N次是否全部合格
        all_passed = all(a.is_qualified for a in recent_analyses)
        
        if not all_passed:
            failed_count = sum(1 for a in recent_analyses if not a.is_qualified)
            return {
                "qualified": False,
                "reason": f"最近 {self.DRY_FIRE_REQUIRED_PASSES} 次中有 {failed_count} 次未达标",
                "progress": record.passed_count,
                "required": self.DRY_FIRE_REQUIRED_PASSES
            }
        
        # 检查平均分
        avg_score = sum(a.overall_score for a in recent_analyses) / len(recent_analyses)
        if avg_score < self.PASS_SCORE_THRESHOLD:
            return {
                "qualified": False,
                "reason": f"平均分 {avg_score:.1f} 低于要求 {self.PASS_SCORE_THRESHOLD}",
                "average_score": avg_score,
                "required_score": self.PASS_SCORE_THRESHOLD
            }
        
        # 全部通过
        return {
            "qualified": True,
            "reason": "恭喜！已通过空枪训练考核",
            "average_score": avg_score,
            "consecutive_passes": self.DRY_FIRE_REQUIRED_PASSES
        }
    
    def approve_live_fire_training(
        self,
        record_id: int,
        approver_id: int
    ) -> TrainingRecord:
        """
        批准进入实弹训练
        
        Args:
            record_id: 训练记录ID
            approver_id: 批准人ID
            
        Returns:
            更新后的TrainingRecord
        """
        record = self.db.query(TrainingRecord).get(record_id)
        
        # 检查资格
        qualification = self.check_dry_fire_qualification(record_id)
        
        if not qualification["qualified"]:
            raise ValueError(f"不符合实弹训练资格：{qualification['reason']}")
        
        # 更新状态
        record.status = "passed"
        record.end_time = datetime.now()
        
        if record.start_time:
            record.total_duration = int((record.end_time - record.start_time).total_seconds())
        
        self.db.commit()
        
        return record
    
    def issue_ammunition(
        self,
        session_id: int,
        student_id: int,
        ammunition_type: str,
        quantity: int,
        issuer_id: int
    ) -> AmmunitionRecord:
        """
        发放弹药
        
        Args:
            session_id: 场次ID
            student_id: 学员ID
            ammunition_type: 弹药类型
            quantity: 数量
            issuer_id: 发放人ID
            
        Returns:
            AmmunitionRecord对象
        """
        # 检查学员是否已通过空枪训练
        dry_fire_record = self.db.query(TrainingRecord).filter(
            TrainingRecord.student_id == student_id,
            TrainingRecord.training_type == "dry_fire",
            TrainingRecord.status == "passed"
        ).order_by(TrainingRecord.end_time.desc()).first()
        
        if not dry_fire_record:
            raise ValueError("学员尚未通过空枪训练，不能发放弹药")
        
        # 创建弹药记录
        ammo_record = AmmunitionRecord(
            session_id=session_id,
            student_id=student_id,
            issue_time=datetime.now(),
            ammunition_type=ammunition_type,
            quantity=quantity,
            issuer_id=issuer_id
        )
        
        self.db.add(ammo_record)
        self.db.commit()
        self.db.refresh(ammo_record)
        
        return ammo_record
    
    def get_student_progress(self, student_id: int, session_id: int) -> Optional[TrainingProgress]:
        """
        获取学员训练进度
        
        Args:
            student_id: 学员ID
            session_id: 场次ID
            
        Returns:
            TrainingProgress对象
        """
        # 获取训练记录
        record = self.db.query(TrainingRecord).filter(
            TrainingRecord.student_id == student_id,
            TrainingRecord.session_id == session_id
        ).first()
        
        if not record:
            return None
        
        # 获取学员信息
        student = self.db.query(User).get(student_id)
        
        # 获取最后一次得分
        last_analysis = self.db.query(ActionAnalysis).filter(
            ActionAnalysis.record_id == record.id
        ).order_by(ActionAnalysis.timestamp.desc()).first()
        
        last_score = last_analysis.overall_score if last_analysis else None
        
        # 检查是否可进入下一阶段
        qualification = self.check_dry_fire_qualification(record.id)
        qualified_for_next = qualification["qualified"]
        
        # 确定当前阶段
        if record.training_type == "dry_fire":
            if record.status == "passed":
                current_stage = TrainingStage.LIVE_FIRE
            else:
                current_stage = TrainingStage.DRY_FIRE
        elif record.training_type == "live_fire":
            current_stage = TrainingStage.LIVE_FIRE
        else:
            current_stage = TrainingStage.NOT_STARTED
        
        # 确定状态
        if record.status == "training":
            status = TrainingStatus.IN_PROGRESS
        elif record.status == "passed":
            status = TrainingStatus.PASSED
        elif record.status == "failed":
            status = TrainingStatus.FAILED
        else:
            status = TrainingStatus.WAITING
        
        # 计算已用时间
        elapsed_time = 0
        if record.start_time:
            if record.end_time:
                elapsed_time = int((record.end_time - record.start_time).total_seconds())
            else:
                elapsed_time = int((datetime.now() - record.start_time).total_seconds())
        
        return TrainingProgress(
            student_id=student_id,
            student_name=student.name,
            workstation_id=record.workstation_id,
            current_stage=current_stage,
            status=status,
            practice_count=record.practice_count or 0,
            passed_count=record.passed_count or 0,
            average_score=float(record.average_score or 0),
            last_score=last_score,
            qualified_for_next=qualified_for_next,
            start_time=record.start_time,
            elapsed_time=elapsed_time
        )
    
    def end_training_session(self, session_id: int):
        """
        结束训练场次
        
        Args:
            session_id: 场次ID
        """
        session = self.db.query(TrainingSession).get(session_id)
        
        session.end_time = datetime.now()
        session.status = "completed"
        
        # 结束所有未完成的训练记录
        active_records = self.db.query(TrainingRecord).filter(
            TrainingRecord.session_id == session_id,
            TrainingRecord.status == "training"
        ).all()
        
        for record in active_records:
            record.end_time = datetime.now()
            if record.start_time:
                record.total_duration = int((record.end_time - record.start_time).total_seconds())
        
        self.db.commit()


# 使用示例
if __name__ == "__main__":
    from ..db.database import get_db, init_db
    from ..db.models import User
    
    # 初始化数据库
    init_db()
    
    with get_db() as db:
        # 创建测试数据
        instructor = User(name="王教官", badge_number="I001", role="instructor")
        student = User(name="李学员", badge_number="S001", role="student")
        db.add_all([instructor, student])
        db.commit()
        
        # 创建训练流程控制器
        controller = TrainingFlowController(db)
        
        # 1. 开始训练场次
        session = controller.start_training_session(
            session_date=datetime.now().date(),
            session_type="dry_fire",
            instructor_id=instructor.id
        )
        print(f"✅ 创建训练场次: {session}")
        
        # 2. 分配学员到工位
        record = controller.assign_student_to_workstation(
            session_id=session.id,
            student_id=student.id,
            workstation_id=1,
            gun_type="92式手枪"
        )
        print(f"✅ 学员分配到工位 {record.workstation_id}")
        
        # 3. 查看训练进度
        progress = controller.get_student_progress(student.id, session.id)
        print(f"✅ 训练进度: {progress}")
        
        print("\n✅ 训练流程测试完成！")

