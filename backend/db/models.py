"""
数据库模型定义
使用SQLAlchemy ORM
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, 
    ForeignKey, Numeric, Text, Date, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, comment="姓名")
    badge_number = Column(String(20), unique=True, nullable=False, comment="警号")
    unit = Column(String(100), comment="单位")
    role = Column(String(20), nullable=False, comment="角色：instructor/student/leader")
    face_encoding = Column(Text, comment="人脸特征编码")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    
    # 关系
    training_records = relationship("TrainingRecord", back_populates="student", foreign_keys="TrainingRecord.student_id")
    instructed_sessions = relationship("TrainingSession", back_populates="instructor")
    
    def __repr__(self):
        return f"<User(name='{self.name}', badge_number='{self.badge_number}', role='{self.role}')>"


class TrainingSession(Base):
    """训练场次表"""
    __tablename__ = "training_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_date = Column(Date, nullable=False, comment="训练日期")
    session_type = Column(String(20), nullable=False, comment="训练类型：dry_fire/live_fire")
    instructor_id = Column(Integer, ForeignKey("users.id"), comment="教官ID")
    start_time = Column(DateTime, comment="开始时间")
    end_time = Column(DateTime, comment="结束时间")
    total_students = Column(Integer, default=0, comment="参训学员数")
    status = Column(String(20), default="ongoing", comment="状态：ongoing/completed/cancelled")
    notes = Column(Text, comment="备注")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    
    # 关系
    instructor = relationship("User", back_populates="instructed_sessions")
    training_records = relationship("TrainingRecord", back_populates="session")
    ammunition_records = relationship("AmmunitionRecord", back_populates="session")
    
    def __repr__(self):
        return f"<TrainingSession(date={self.session_date}, type='{self.session_type}', status='{self.status}')>"


class TrainingRecord(Base):
    """训练记录表"""
    __tablename__ = "training_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"), nullable=False, comment="场次ID")
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="学员ID")
    workstation_id = Column(Integer, comment="工位号")
    gun_type = Column(String(50), comment="枪支型号")
    training_type = Column(String(20), nullable=False, comment="训练类型：dry_fire/live_fire")
    start_time = Column(DateTime, comment="开始时间")
    end_time = Column(DateTime, comment="结束时间")
    total_duration = Column(Integer, comment="总时长（秒）")
    practice_count = Column(Integer, default=0, comment="练习次数")
    passed_count = Column(Integer, default=0, comment="合格次数")
    average_score = Column(Numeric(5, 2), comment="平均分")
    status = Column(String(20), default="training", comment="状态：training/passed/failed")
    notes = Column(Text, comment="备注")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    
    # 关系
    session = relationship("TrainingSession", back_populates="training_records")
    student = relationship("User", back_populates="training_records", foreign_keys=[student_id])
    action_analyses = relationship("ActionAnalysis", back_populates="record")
    warnings = relationship("Warning", back_populates="record")
    
    def __repr__(self):
        return f"<TrainingRecord(student_id={self.student_id}, workstation={self.workstation_id}, status='{self.status}')>"


class ActionAnalysis(Base):
    """动作分析表"""
    __tablename__ = "action_analysis"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(Integer, ForeignKey("training_records.id"), nullable=False, comment="训练记录ID")
    timestamp = Column(DateTime, nullable=False, comment="时间戳")
    posture_score = Column(Numeric(5, 2), comment="姿势评分")
    trigger_discipline_score = Column(Numeric(5, 2), comment="扳机纪律评分")
    aim_line_score = Column(Numeric(5, 2), comment="瞄准线评分")
    stability_score = Column(Numeric(5, 2), comment="稳定性评分")
    overall_score = Column(Numeric(5, 2), comment="总分")
    is_qualified = Column(Boolean, default=False, comment="是否合格")
    errors = Column(JSON, comment="错误详情（JSON格式）")
    video_clip_id = Column(String(100), comment="视频片段ID")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    
    # 关系
    record = relationship("TrainingRecord", back_populates="action_analyses")
    
    def __repr__(self):
        return f"<ActionAnalysis(record_id={self.record_id}, score={self.overall_score}, qualified={self.is_qualified})>"


class Warning(Base):
    """预警记录表"""
    __tablename__ = "warnings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(Integer, ForeignKey("training_records.id"), nullable=False, comment="训练记录ID")
    timestamp = Column(DateTime, nullable=False, comment="时间戳")
    warning_level = Column(String(20), nullable=False, comment="警告级别：critical/serious/notice")
    warning_type = Column(String(50), nullable=False, comment="警告类型")
    description = Column(Text, nullable=False, comment="描述")
    handled = Column(Boolean, default=False, comment="是否已处理")
    handler_id = Column(Integer, ForeignKey("users.id"), comment="处理人ID")
    handle_time = Column(DateTime, comment="处理时间")
    video_clip_id = Column(String(100), comment="视频片段ID")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    
    # 关系
    record = relationship("TrainingRecord", back_populates="warnings")
    
    def __repr__(self):
        return f"<Warning(level='{self.warning_level}', type='{self.warning_type}', handled={self.handled})>"


class AmmunitionRecord(Base):
    """弹药记录表"""
    __tablename__ = "ammunition_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"), nullable=False, comment="场次ID")
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="学员ID")
    issue_time = Column(DateTime, nullable=False, comment="发放时间")
    ammunition_type = Column(String(50), nullable=False, comment="弹药类型")
    quantity = Column(Integer, nullable=False, comment="发放数量")
    used_quantity = Column(Integer, default=0, comment="使用数量")
    returned_quantity = Column(Integer, default=0, comment="归还数量")
    issuer_id = Column(Integer, ForeignKey("users.id"), comment="发放人ID")
    receiver_id = Column(Integer, ForeignKey("users.id"), comment="回收人ID")
    notes = Column(Text, comment="备注")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    
    # 关系
    session = relationship("TrainingSession", back_populates="ammunition_records")
    
    def __repr__(self):
        return f"<AmmunitionRecord(student_id={self.student_id}, quantity={self.quantity}, used={self.used_quantity})>"


class SystemConfig(Base):
    """系统配置表"""
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    config_key = Column(String(50), unique=True, nullable=False, comment="配置键")
    config_value = Column(Text, nullable=False, comment="配置值")
    description = Column(String(200), comment="描述")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")
    
    def __repr__(self):
        return f"<SystemConfig(key='{self.config_key}', value='{self.config_value}')>"


class AuditLog(Base):
    """操作审计日志表"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), comment="操作人ID")
    action = Column(String(50), nullable=False, comment="操作类型")
    resource_type = Column(String(50), comment="资源类型")
    resource_id = Column(Integer, comment="资源ID")
    details = Column(JSON, comment="详细信息（JSON格式）")
    ip_address = Column(String(45), comment="IP地址")
    timestamp = Column(DateTime, default=datetime.now, comment="时间戳")
    
    def __repr__(self):
        return f"<AuditLog(user_id={self.user_id}, action='{self.action}', timestamp={self.timestamp})>"

