"""
数据统计和报告生成模块
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from dataclasses import dataclass
import pandas as pd

from ..db.models import (
    User, TrainingSession, TrainingRecord,
    ActionAnalysis, Warning, AmmunitionRecord
)


@dataclass
class StudentPerformance:
    """学员表现统计"""
    student_id: int
    student_name: str
    badge_number: str
    unit: str
    total_sessions: int
    total_practice_count: int
    average_score: float
    pass_rate: float
    dry_fire_passed: bool
    live_fire_count: int
    total_warnings: int
    critical_warnings: int
    improvement_trend: str  # improving/stable/declining


@dataclass
class SessionStatisticsData:
    """场次统计数据"""
    session_id: int
    session_date: date
    session_type: str
    instructor_name: str
    total_students: int
    completed_students: int
    passed_students: int
    average_score: float
    total_practice_count: int
    total_warnings: int
    duration_minutes: int


@dataclass
class InstructorWorkload:
    """教官工作量统计"""
    instructor_id: int
    instructor_name: str
    total_sessions: int
    total_students_trained: int
    total_hours: float
    average_student_score: float
    warnings_handled: int


class StatisticsAnalyzer:
    """统计分析器"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_student_performance(
        self,
        student_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Optional[StudentPerformance]:
        """
        获取学员表现统计
        
        Args:
            student_id: 学员ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            StudentPerformance对象
        """
        # 获取学员信息
        student = self.db.query(User).get(student_id)
        if not student:
            return None
        
        # 构建查询
        query = self.db.query(TrainingRecord).filter(
            TrainingRecord.student_id == student_id
        )
        
        if start_date:
            query = query.join(TrainingSession).filter(
                TrainingSession.session_date >= start_date
            )
        if end_date:
            query = query.join(TrainingSession).filter(
                TrainingSession.session_date <= end_date
            )
        
        records = query.all()
        
        if not records:
            return StudentPerformance(
                student_id=student_id,
                student_name=student.name,
                badge_number=student.badge_number,
                unit=student.unit or "",
                total_sessions=0,
                total_practice_count=0,
                average_score=0.0,
                pass_rate=0.0,
                dry_fire_passed=False,
                live_fire_count=0,
                total_warnings=0,
                critical_warnings=0,
                improvement_trend="stable"
            )
        
        # 统计基本数据
        total_sessions = len(records)
        total_practice_count = sum(r.practice_count or 0 for r in records)
        total_passed_count = sum(r.passed_count or 0 for r in records)
        
        # 计算平均分
        scores = [float(r.average_score) for r in records if r.average_score]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # 计算通过率
        pass_rate = (total_passed_count / total_practice_count * 100) if total_practice_count > 0 else 0.0
        
        # 检查是否通过空枪训练
        dry_fire_passed = any(
            r.training_type == "dry_fire" and r.status == "passed"
            for r in records
        )
        
        # 实弹训练次数
        live_fire_count = sum(
            1 for r in records if r.training_type == "live_fire"
        )
        
        # 统计预警
        record_ids = [r.id for r in records]
        warnings = self.db.query(Warning).filter(
            Warning.record_id.in_(record_ids)
        ).all()
        
        total_warnings = len(warnings)
        critical_warnings = sum(
            1 for w in warnings if w.warning_level == "critical"
        )
        
        # 计算进步趋势
        improvement_trend = self._calculate_improvement_trend(records)
        
        return StudentPerformance(
            student_id=student_id,
            student_name=student.name,
            badge_number=student.badge_number,
            unit=student.unit or "",
            total_sessions=total_sessions,
            total_practice_count=total_practice_count,
            average_score=round(average_score, 2),
            pass_rate=round(pass_rate, 2),
            dry_fire_passed=dry_fire_passed,
            live_fire_count=live_fire_count,
            total_warnings=total_warnings,
            critical_warnings=critical_warnings,
            improvement_trend=improvement_trend
        )
    
    def _calculate_improvement_trend(
        self,
        records: List[TrainingRecord]
    ) -> str:
        """计算进步趋势"""
        if len(records) < 2:
            return "stable"
        
        # 按时间排序
        sorted_records = sorted(records, key=lambda r: r.start_time or datetime.min)
        
        # 获取前半段和后半段的平均分
        mid = len(sorted_records) // 2
        first_half = [float(r.average_score) for r in sorted_records[:mid] if r.average_score]
        second_half = [float(r.average_score) for r in sorted_records[mid:] if r.average_score]
        
        if not first_half or not second_half:
            return "stable"
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        diff = second_avg - first_avg
        
        if diff > 5:
            return "improving"
        elif diff < -5:
            return "declining"
        else:
            return "stable"
    
    def get_session_statistics(
        self,
        session_id: int
    ) -> Optional[SessionStatisticsData]:
        """
        获取场次统计
        
        Args:
            session_id: 场次ID
            
        Returns:
            SessionStatisticsData对象
        """
        session = self.db.query(TrainingSession).get(session_id)
        if not session:
            return None
        
        instructor = self.db.query(User).get(session.instructor_id)
        instructor_name = instructor.name if instructor else "未知"
        
        # 获取所有训练记录
        records = self.db.query(TrainingRecord).filter(
            TrainingRecord.session_id == session_id
        ).all()
        
        # 统计数据
        total_students = len(records)
        completed_students = sum(
            1 for r in records if r.status in ["passed", "failed"]
        )
        passed_students = sum(
            1 for r in records if r.status == "passed"
        )
        
        # 平均分
        scores = [float(r.average_score) for r in records if r.average_score]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # 练习总次数
        total_practice_count = sum(r.practice_count or 0 for r in records)
        
        # 预警统计
        record_ids = [r.id for r in records]
        total_warnings = self.db.query(Warning).filter(
            Warning.record_id.in_(record_ids)
        ).count()
        
        # 计算时长
        duration_minutes = 0
        if session.start_time and session.end_time:
            duration_minutes = int(
                (session.end_time - session.start_time).total_seconds() / 60
            )
        
        return SessionStatisticsData(
            session_id=session_id,
            session_date=session.session_date,
            session_type=session.session_type,
            instructor_name=instructor_name,
            total_students=total_students,
            completed_students=completed_students,
            passed_students=passed_students,
            average_score=round(average_score, 2),
            total_practice_count=total_practice_count,
            total_warnings=total_warnings,
            duration_minutes=duration_minutes
        )
    
    def get_instructor_workload(
        self,
        instructor_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Optional[InstructorWorkload]:
        """
        获取教官工作量统计
        
        Args:
            instructor_id: 教官ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            InstructorWorkload对象
        """
        instructor = self.db.query(User).get(instructor_id)
        if not instructor:
            return None
        
        # 查询场次
        query = self.db.query(TrainingSession).filter(
            TrainingSession.instructor_id == instructor_id
        )
        
        if start_date:
            query = query.filter(TrainingSession.session_date >= start_date)
        if end_date:
            query = query.filter(TrainingSession.session_date <= end_date)
        
        sessions = query.all()
        
        if not sessions:
            return InstructorWorkload(
                instructor_id=instructor_id,
                instructor_name=instructor.name,
                total_sessions=0,
                total_students_trained=0,
                total_hours=0.0,
                average_student_score=0.0,
                warnings_handled=0
            )
        
        # 统计数据
        total_sessions = len(sessions)
        
        # 统计学员数
        session_ids = [s.id for s in sessions]
        total_students = self.db.query(TrainingRecord).filter(
            TrainingRecord.session_id.in_(session_ids)
        ).count()
        
        # 计算总时长
        total_hours = 0.0
        for session in sessions:
            if session.start_time and session.end_time:
                hours = (session.end_time - session.start_time).total_seconds() / 3600
                total_hours += hours
        
        # 计算学员平均分
        records = self.db.query(TrainingRecord).filter(
            TrainingRecord.session_id.in_(session_ids)
        ).all()
        
        scores = [float(r.average_score) for r in records if r.average_score]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # 处理的预警数
        warnings_handled = self.db.query(Warning).filter(
            Warning.handler_id == instructor_id
        ).count()
        
        return InstructorWorkload(
            instructor_id=instructor_id,
            instructor_name=instructor.name,
            total_sessions=total_sessions,
            total_students_trained=total_students,
            total_hours=round(total_hours, 2),
            average_student_score=round(average_score, 2),
            warnings_handled=warnings_handled
        )
    
    def get_period_summary(
        self,
        start_date: date,
        end_date: date
    ) -> Dict:
        """
        获取时间段汇总统计
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            汇总统计字典
        """
        # 场次统计
        sessions = self.db.query(TrainingSession).filter(
            and_(
                TrainingSession.session_date >= start_date,
                TrainingSession.session_date <= end_date
            )
        ).all()
        
        session_ids = [s.id for s in sessions]
        
        # 训练记录统计
        records = self.db.query(TrainingRecord).filter(
            TrainingRecord.session_id.in_(session_ids)
        ).all()
        
        # 基本统计
        total_sessions = len(sessions)
        total_students = len(set(r.student_id for r in records))
        total_practice_count = sum(r.practice_count or 0 for r in records)
        
        # 通过率
        passed_records = [r for r in records if r.status == "passed"]
        pass_rate = (len(passed_records) / len(records) * 100) if records else 0.0
        
        # 平均分
        scores = [float(r.average_score) for r in records if r.average_score]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # 弹药使用
        ammo_records = self.db.query(AmmunitionRecord).filter(
            AmmunitionRecord.session_id.in_(session_ids)
        ).all()
        
        total_ammunition_used = sum(r.used_quantity for r in ammo_records)
        
        # 预警统计
        record_ids = [r.id for r in records]
        warnings = self.db.query(Warning).filter(
            Warning.record_id.in_(record_ids)
        ).all()
        
        total_warnings = len(warnings)
        critical_warnings = sum(1 for w in warnings if w.warning_level == "critical")
        
        # 空枪训练统计
        dry_fire_records = [r for r in records if r.training_type == "dry_fire"]
        dry_fire_passed = sum(1 for r in dry_fire_records if r.status == "passed")
        
        # 实弹训练统计
        live_fire_records = [r for r in records if r.training_type == "live_fire"]
        
        return {
            "period": {
                "start_date": str(start_date),
                "end_date": str(end_date)
            },
            "sessions": {
                "total": total_sessions,
                "dry_fire": sum(1 for s in sessions if s.session_type == "dry_fire"),
                "live_fire": sum(1 for s in sessions if s.session_type == "live_fire")
            },
            "students": {
                "total": total_students,
                "practice_count": total_practice_count
            },
            "performance": {
                "average_score": round(average_score, 2),
                "pass_rate": round(pass_rate, 2)
            },
            "training": {
                "dry_fire_completed": len(dry_fire_records),
                "dry_fire_passed": dry_fire_passed,
                "live_fire_completed": len(live_fire_records)
            },
            "ammunition": {
                "total_used": total_ammunition_used
            },
            "safety": {
                "total_warnings": total_warnings,
                "critical_warnings": critical_warnings
            }
        }
    
    def get_top_students(
        self,
        limit: int = 10,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[StudentPerformance]:
        """
        获取优秀学员排行
        
        Args:
            limit: 返回数量
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            学员表现列表（按平均分降序）
        """
        # 获取所有学员ID
        student_ids = self.db.query(User.id).filter(
            User.role == "student"
        ).all()
        
        # 获取每个学员的表现
        performances = []
        for (student_id,) in student_ids:
            perf = self.get_student_performance(student_id, start_date, end_date)
            if perf and perf.total_practice_count > 0:
                performances.append(perf)
        
        # 按平均分排序
        performances.sort(key=lambda p: p.average_score, reverse=True)
        
        return performances[:limit]
    
    def export_report_to_excel(
        self,
        start_date: date,
        end_date: date,
        output_file: str
    ):
        """
        导出Excel报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            output_file: 输出文件路径
        """
        # 创建Excel写入器
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. 汇总统计
            summary = self.get_period_summary(start_date, end_date)
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)
            
            # 2. 学员表现
            students = self.get_top_students(limit=1000, start_date=start_date, end_date=end_date)
            students_data = [
                {
                    "警号": s.badge_number,
                    "姓名": s.student_name,
                    "单位": s.unit,
                    "训练场次": s.total_sessions,
                    "练习次数": s.total_practice_count,
                    "平均分": s.average_score,
                    "通过率": f"{s.pass_rate}%",
                    "空枪训练": "已通过" if s.dry_fire_passed else "未通过",
                    "实弹次数": s.live_fire_count,
                    "预警次数": s.total_warnings,
                    "趋势": s.improvement_trend
                }
                for s in students
            ]
            students_df = pd.DataFrame(students_data)
            students_df.to_excel(writer, sheet_name='学员表现', index=False)
            
            # 3. 场次统计
            sessions = self.db.query(TrainingSession).filter(
                and_(
                    TrainingSession.session_date >= start_date,
                    TrainingSession.session_date <= end_date
                )
            ).all()
            
            sessions_data = []
            for session in sessions:
                stat = self.get_session_statistics(session.id)
                if stat:
                    sessions_data.append({
                        "日期": str(stat.session_date),
                        "类型": stat.session_type,
                        "教官": stat.instructor_name,
                        "学员数": stat.total_students,
                        "完成数": stat.completed_students,
                        "通过数": stat.passed_students,
                        "平均分": stat.average_score,
                        "练习次数": stat.total_practice_count,
                        "预警数": stat.total_warnings,
                        "时长(分钟)": stat.duration_minutes
                    })
            
            sessions_df = pd.DataFrame(sessions_data)
            sessions_df.to_excel(writer, sheet_name='场次统计', index=False)
        
        print(f"✅ 报告已导出到: {output_file}")


# 使用示例
if __name__ == "__main__":
    from ..db.database import get_db, init_db
    from datetime import timedelta
    
    init_db()
    
    with get_db() as db:
        analyzer = StatisticsAnalyzer(db)
        
        # 获取最近30天的汇总统计
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        summary = analyzer.get_period_summary(start_date, end_date)
        
        print("=" * 50)
        print("最近30天训练汇总")
        print("=" * 50)
        print(f"训练场次: {summary['sessions']['total']}")
        print(f"参训学员: {summary['students']['total']}")
        print(f"平均成绩: {summary['performance']['average_score']}")
        print(f"通过率: {summary['performance']['pass_rate']}%")
        print(f"预警次数: {summary['safety']['total_warnings']}")
        print("=" * 50)

