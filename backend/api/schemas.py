"""
Pydantic数据模型（用于API请求和响应）
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum


# ==================== 用户相关 ====================

class UserBase(BaseModel):
    """用户基础模型"""
    name: str = Field(..., description="姓名")
    badge_number: str = Field(..., description="警号")
    unit: Optional[str] = Field(None, description="单位")
    role: str = Field(..., description="角色：instructor/student/leader")


class UserCreate(UserBase):
    """创建用户"""
    pass


class UserResponse(UserBase):
    """用户响应"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# ==================== 训练场次相关 ====================

class SessionCreate(BaseModel):
    """创建训练场次"""
    session_date: date = Field(..., description="训练日期")
    session_type: str = Field(..., description="训练类型：dry_fire/live_fire")
    instructor_id: int = Field(..., description="教官ID")


class SessionResponse(BaseModel):
    """训练场次响应"""
    id: int
    session_date: date
    session_type: str
    instructor_id: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    total_students: int
    status: str
    
    class Config:
        from_attributes = True


class SessionDetailResponse(SessionResponse):
    """训练场次详情"""
    notes: Optional[str]
    training_records: List['TrainingRecordResponse']
    
    class Config:
        from_attributes = True


# ==================== 训练记录相关 ====================

class TrainingRecordCreate(BaseModel):
    """创建训练记录（分配工位）"""
    session_id: int = Field(..., description="场次ID")
    student_id: int = Field(..., description="学员ID")
    workstation_id: int = Field(..., description="工位号")
    gun_type: str = Field(..., description="枪支型号")


class TrainingRecordResponse(BaseModel):
    """训练记录响应"""
    id: int
    session_id: int
    student_id: int
    workstation_id: int
    gun_type: str
    training_type: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    total_duration: Optional[int]
    practice_count: int
    passed_count: int
    average_score: Optional[float]
    status: str
    
    class Config:
        from_attributes = True


class TrainingProgressResponse(BaseModel):
    """训练进度响应"""
    student_id: int
    student_name: str
    workstation_id: int
    current_stage: str
    status: str
    practice_count: int
    passed_count: int
    average_score: float
    last_score: Optional[float]
    qualified_for_next: bool
    start_time: Optional[datetime]
    elapsed_time: int


# ==================== 动作分析相关 ====================

class ActionAnalysisResponse(BaseModel):
    """动作分析响应"""
    id: int
    record_id: int
    timestamp: datetime
    posture_score: Optional[float]
    trigger_discipline_score: Optional[float]
    aim_line_score: Optional[float]
    stability_score: Optional[float]
    overall_score: Optional[float]
    is_qualified: bool
    errors: Optional[Dict[str, Any]]
    video_clip_id: Optional[str]
    
    class Config:
        from_attributes = True


# ==================== 预警相关 ====================

class WarningResponse(BaseModel):
    """预警响应"""
    id: int
    record_id: int
    timestamp: datetime
    warning_level: str
    warning_type: str
    description: str
    handled: bool
    handler_id: Optional[int]
    handle_time: Optional[datetime]
    video_clip_id: Optional[str]
    
    class Config:
        from_attributes = True


# ==================== WebSocket消息 ====================

class WSMessage(BaseModel):
    """WebSocket消息基础模型"""
    type: str = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.now)


class RealTimeAnalysisMessage(WSMessage):
    """实时分析消息"""
    type: str = "real_time_analysis"
    data: Dict[str, Any] = Field(..., description="""
    {
        "workstation_id": int,
        "student_id": int,
        "overall_score": float,
        "posture_score": float,
        "trigger_discipline_score": float,
        "aim_line_score": float,
        "stability_score": float,
        "is_qualified": bool,
        "errors": [{"type": str, "description": str, "level": str}]
    }
    """)


class WarningMessage(WSMessage):
    """预警消息"""
    type: str = "warning"
    data: Dict[str, Any] = Field(..., description="""
    {
        "workstation_id": int,
        "student_id": int,
        "warning_level": str,
        "warning_type": str,
        "description": str
    }
    """)


# ==================== 统计相关 ====================

class WorkstationStatus(BaseModel):
    """工位状态"""
    workstation_id: int
    is_occupied: bool
    student_id: Optional[int]
    student_name: Optional[str]
    current_score: Optional[float]
    practice_count: int
    status: str  # training/passed/waiting


class SessionStatistics(BaseModel):
    """场次统计"""
    session_id: int
    total_students: int
    active_students: int
    passed_students: int
    failed_students: int
    average_score: float
    total_warnings: int
    critical_warnings: int
    workstations: List[WorkstationStatus]

