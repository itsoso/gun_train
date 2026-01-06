"""
FastAPI主应用
提供REST API和WebSocket实时通信
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime, date
import json

from ..db.database import get_db_session, init_db
from ..db.models import (
    User, TrainingSession, TrainingRecord,
    ActionAnalysis, Warning
)
from ..core.training_manager import TrainingFlowController
from .schemas import *

# 创建FastAPI应用
app = FastAPI(
    title="智能枪械训练监控系统",
    description="基于AI视觉识别的枪械训练智能监控平台",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket连接管理器
class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 存储所有活跃连接
        self.active_connections: Dict[str, List[WebSocket]] = {
            "instructors": [],  # 教官端
            "students": [],     # 学员端
            "leaders": []       # 领导端
        }
    
    async def connect(self, websocket: WebSocket, client_type: str):
        """连接客户端"""
        await websocket.accept()
        if client_type in self.active_connections:
            self.active_connections[client_type].append(websocket)
    
    def disconnect(self, websocket: WebSocket, client_type: str):
        """断开客户端"""
        if client_type in self.active_connections:
            self.active_connections[client_type].remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """发送个人消息"""
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict, client_type: str):
        """广播消息到指定类型的所有客户端"""
        if client_type in self.active_connections:
            for connection in self.active_connections[client_type]:
                try:
                    await connection.send_json(message)
                except:
                    pass


manager = ConnectionManager()


# ==================== 基础路由 ====================

@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "智能枪械训练监控系统 API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ==================== 用户管理 ====================

@app.get("/api/users", response_model=List[UserResponse])
async def get_users(
    role: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """获取用户列表"""
    query = db.query(User)
    
    if role:
        query = query.filter(User.role == role)
    
    users = query.all()
    return users


@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db_session)):
    """获取用户详情"""
    user = db.query(User).get(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return user


@app.post("/api/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db_session)):
    """创建用户"""
    # 检查警号是否已存在
    existing = db.query(User).filter(User.badge_number == user.badge_number).first()
    if existing:
        raise HTTPException(status_code=400, detail="警号已存在")
    
    new_user = User(**user.dict())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


# ==================== 训练场次管理 ====================

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(
    session: SessionCreate,
    db: Session = Depends(get_db_session)
):
    """创建训练场次"""
    controller = TrainingFlowController(db)
    
    new_session = controller.start_training_session(
        session_date=session.session_date,
        session_type=session.session_type,
        instructor_id=session.instructor_id
    )
    
    # 广播通知
    await manager.broadcast({
        "type": "session_started",
        "data": {
            "session_id": new_session.id,
            "session_date": str(new_session.session_date),
            "session_type": new_session.session_type
        }
    }, "instructors")
    
    return new_session


@app.get("/api/sessions", response_model=List[SessionResponse])
async def get_sessions(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """获取训练场次列表"""
    query = db.query(TrainingSession)
    
    if start_date:
        query = query.filter(TrainingSession.session_date >= start_date)
    if end_date:
        query = query.filter(TrainingSession.session_date <= end_date)
    if status:
        query = query.filter(TrainingSession.status == status)
    
    sessions = query.order_by(TrainingSession.session_date.desc()).all()
    return sessions


@app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: int, db: Session = Depends(get_db_session)):
    """获取训练场次详情"""
    session = db.query(TrainingSession).get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="训练场次不存在")
    
    return session


@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: int, db: Session = Depends(get_db_session)):
    """结束训练场次"""
    controller = TrainingFlowController(db)
    controller.end_training_session(session_id)
    
    # 广播通知
    await manager.broadcast({
        "type": "session_ended",
        "data": {"session_id": session_id}
    }, "instructors")
    
    return {"message": "训练场次已结束"}


# ==================== 训练记录管理 ====================

@app.post("/api/training-records", response_model=TrainingRecordResponse)
async def assign_workstation(
    record: TrainingRecordCreate,
    db: Session = Depends(get_db_session)
):
    """分配学员到工位"""
    controller = TrainingFlowController(db)
    
    try:
        new_record = controller.assign_student_to_workstation(
            session_id=record.session_id,
            student_id=record.student_id,
            workstation_id=record.workstation_id,
            gun_type=record.gun_type
        )
        
        # 广播通知
        await manager.broadcast({
            "type": "student_assigned",
            "data": {
                "record_id": new_record.id,
                "workstation_id": new_record.workstation_id,
                "student_id": new_record.student_id
            }
        }, "instructors")
        
        return new_record
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/training-records/{record_id}/progress", response_model=TrainingProgressResponse)
async def get_training_progress(
    record_id: int,
    db: Session = Depends(get_db_session)
):
    """获取训练进度"""
    record = db.query(TrainingRecord).get(record_id)
    
    if not record:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    
    controller = TrainingFlowController(db)
    progress = controller.get_student_progress(record.student_id, record.session_id)
    
    if not progress:
        raise HTTPException(status_code=404, detail="无法获取训练进度")
    
    return progress


@app.post("/api/training-records/{record_id}/qualification-check")
async def check_qualification(
    record_id: int,
    db: Session = Depends(get_db_session)
):
    """检查空枪训练资格"""
    controller = TrainingFlowController(db)
    result = controller.check_dry_fire_qualification(record_id)
    
    return result


@app.post("/api/training-records/{record_id}/approve-live-fire")
async def approve_live_fire(
    record_id: int,
    approver_id: int,
    db: Session = Depends(get_db_session)
):
    """批准进入实弹训练"""
    controller = TrainingFlowController(db)
    
    try:
        updated_record = controller.approve_live_fire_training(record_id, approver_id)
        
        # 广播通知
        await manager.broadcast({
            "type": "live_fire_approved",
            "data": {
                "record_id": record_id,
                "student_id": updated_record.student_id
            }
        }, "instructors")
        
        return {"message": "已批准进入实弹训练", "record_id": record_id}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== 动作分析 ====================

@app.get("/api/training-records/{record_id}/analyses", response_model=List[ActionAnalysisResponse])
async def get_action_analyses(
    record_id: int,
    limit: int = 50,
    db: Session = Depends(get_db_session)
):
    """获取动作分析记录"""
    analyses = db.query(ActionAnalysis).filter(
        ActionAnalysis.record_id == record_id
    ).order_by(ActionAnalysis.timestamp.desc()).limit(limit).all()
    
    return analyses


# ==================== 预警管理 ====================

@app.get("/api/warnings", response_model=List[WarningResponse])
async def get_warnings(
    session_id: Optional[int] = None,
    warning_level: Optional[str] = None,
    handled: Optional[bool] = None,
    limit: int = 100,
    db: Session = Depends(get_db_session)
):
    """获取预警列表"""
    query = db.query(Warning)
    
    if session_id:
        query = query.join(TrainingRecord).filter(
            TrainingRecord.session_id == session_id
        )
    
    if warning_level:
        query = query.filter(Warning.warning_level == warning_level)
    
    if handled is not None:
        query = query.filter(Warning.handled == handled)
    
    warnings = query.order_by(Warning.timestamp.desc()).limit(limit).all()
    return warnings


@app.post("/api/warnings/{warning_id}/handle")
async def handle_warning(
    warning_id: int,
    handler_id: int,
    db: Session = Depends(get_db_session)
):
    """处理预警"""
    warning = db.query(Warning).get(warning_id)
    
    if not warning:
        raise HTTPException(status_code=404, detail="预警不存在")
    
    warning.handled = True
    warning.handler_id = handler_id
    warning.handle_time = datetime.now()
    
    db.commit()
    
    return {"message": "预警已处理"}


# ==================== WebSocket实时通信 ====================

@app.websocket("/ws/instructor/{instructor_id}")
async def instructor_websocket(websocket: WebSocket, instructor_id: int):
    """教官端WebSocket"""
    await manager.connect(websocket, "instructors")
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理不同类型的消息
            if message["type"] == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "instructors")


@app.websocket("/ws/student/{student_id}/workstation/{workstation_id}")
async def student_websocket(websocket: WebSocket, student_id: int, workstation_id: int):
    """学员端WebSocket"""
    await manager.connect(websocket, "students")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "students")


@app.websocket("/ws/monitor")
async def monitor_websocket(websocket: WebSocket):
    """监控大屏WebSocket"""
    await manager.connect(websocket, "leaders")
    
    try:
        while True:
            data = await websocket.receive_text()
            # 处理监控端消息
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "leaders")


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("🚀 智能枪械训练监控系统启动中...")
    
    # 初始化数据库
    try:
        init_db()
        print("✅ 数据库初始化成功")
    except Exception as e:
        print(f"⚠️ 数据库初始化失败: {e}")
    
    print("✅ 系统启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    print("👋 系统正在关闭...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

