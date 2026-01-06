"""
WebSocket 实时通信管理器
支持多客户端连接、房间管理、消息广播
"""

import asyncio
import json
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from fastapi import WebSocket, WebSocketDisconnect
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型"""
    # 系统消息
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    
    # 训练消息
    TRAINING_FEEDBACK = "training_feedback"     # 训练反馈
    SCORE_UPDATE = "score_update"               # 分数更新
    ACTION_ERROR = "action_error"               # 动作错误
    DANGER_WARNING = "danger_warning"           # 危险警告
    STAGE_COMPLETE = "stage_complete"           # 阶段完成
    
    # 状态消息
    WORKSTATION_STATUS = "workstation_status"   # 工位状态
    CAMERA_STATUS = "camera_status"             # 摄像头状态
    SYSTEM_STATUS = "system_status"             # 系统状态
    
    # 控制消息
    START_TRAINING = "start_training"           # 开始训练
    STOP_TRAINING = "stop_training"             # 停止训练
    PAUSE_TRAINING = "pause_training"           # 暂停训练


@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    sender_id: Optional[str] = None
    target_id: Optional[str] = None  # 目标客户端ID或房间ID
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "target_id": self.target_id
        }, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            sender_id=data.get("sender_id"),
            target_id=data.get("target_id")
        )


@dataclass
class ClientInfo:
    """客户端信息"""
    client_id: str
    websocket: WebSocket
    client_type: str = "unknown"  # student, instructor, admin, leader
    workstation_id: Optional[int] = None
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    rooms: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    @property
    def is_alive(self) -> bool:
        """检查客户端是否存活（30秒超时）"""
        return (datetime.now() - self.last_heartbeat).total_seconds() < 30


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 活跃连接: {client_id: ClientInfo}
        self.active_connections: Dict[str, ClientInfo] = {}
        
        # 房间: {room_id: Set[client_id]}
        self.rooms: Dict[str, Set[str]] = {}
        
        # 工位订阅: {workstation_id: Set[client_id]}
        self.workstation_subscriptions: Dict[int, Set[str]] = {}
        
        # 消息处理器: {message_type: handler_fn}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # 统计
        self.total_connections = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        client_type: str = "unknown",
        workstation_id: Optional[int] = None,
        user_id: Optional[int] = None,
        user_name: Optional[str] = None
    ) -> ClientInfo:
        """
        接受WebSocket连接
        
        Args:
            websocket: WebSocket实例
            client_id: 客户端ID
            client_type: 客户端类型
            workstation_id: 工位ID
            user_id: 用户ID
            user_name: 用户名
            
        Returns:
            客户端信息
        """
        await websocket.accept()
        
        client = ClientInfo(
            client_id=client_id,
            websocket=websocket,
            client_type=client_type,
            workstation_id=workstation_id,
            user_id=user_id,
            user_name=user_name
        )
        
        self.active_connections[client_id] = client
        self.total_connections += 1
        
        # 自动加入对应的房间
        if client_type:
            await self.join_room(client_id, f"type:{client_type}")
        
        if workstation_id:
            await self.subscribe_workstation(client_id, workstation_id)
        
        logger.info(f"🔌 客户端连接: {client_id} ({client_type})")
        
        # 发送连接成功消息
        await self.send_to_client(client_id, WebSocketMessage(
            type=MessageType.CONNECT,
            data={
                "client_id": client_id,
                "message": "连接成功"
            }
        ))
        
        return client
    
    async def disconnect(self, client_id: str):
        """断开连接"""
        if client_id not in self.active_connections:
            return
        
        client = self.active_connections[client_id]
        
        # 离开所有房间
        for room_id in list(client.rooms):
            await self.leave_room(client_id, room_id)
        
        # 取消工位订阅
        if client.workstation_id:
            await self.unsubscribe_workstation(client_id, client.workstation_id)
        
        del self.active_connections[client_id]
        
        logger.info(f"🔌 客户端断开: {client_id}")
    
    async def join_room(self, client_id: str, room_id: str):
        """加入房间"""
        if client_id not in self.active_connections:
            return
        
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        
        self.rooms[room_id].add(client_id)
        self.active_connections[client_id].rooms.add(room_id)
        
        logger.debug(f"📦 {client_id} 加入房间 {room_id}")
    
    async def leave_room(self, client_id: str, room_id: str):
        """离开房间"""
        if room_id in self.rooms:
            self.rooms[room_id].discard(client_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
        
        if client_id in self.active_connections:
            self.active_connections[client_id].rooms.discard(room_id)
    
    async def subscribe_workstation(self, client_id: str, workstation_id: int):
        """订阅工位消息"""
        if workstation_id not in self.workstation_subscriptions:
            self.workstation_subscriptions[workstation_id] = set()
        
        self.workstation_subscriptions[workstation_id].add(client_id)
        logger.debug(f"📡 {client_id} 订阅工位 {workstation_id}")
    
    async def unsubscribe_workstation(self, client_id: str, workstation_id: int):
        """取消订阅工位"""
        if workstation_id in self.workstation_subscriptions:
            self.workstation_subscriptions[workstation_id].discard(client_id)
    
    async def send_to_client(self, client_id: str, message: WebSocketMessage):
        """发送消息到指定客户端"""
        if client_id not in self.active_connections:
            return False
        
        try:
            await self.active_connections[client_id].websocket.send_text(message.to_json())
            self.total_messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            await self.disconnect(client_id)
            return False
    
    async def send_to_room(self, room_id: str, message: WebSocketMessage):
        """发送消息到房间"""
        if room_id not in self.rooms:
            return
        
        for client_id in list(self.rooms[room_id]):
            await self.send_to_client(client_id, message)
    
    async def send_to_workstation(self, workstation_id: int, message: WebSocketMessage):
        """发送消息到工位订阅者"""
        if workstation_id not in self.workstation_subscriptions:
            return
        
        for client_id in list(self.workstation_subscriptions[workstation_id]):
            await self.send_to_client(client_id, message)
    
    async def broadcast(self, message: WebSocketMessage, exclude: Optional[Set[str]] = None):
        """广播消息到所有客户端"""
        exclude = exclude or set()
        
        for client_id in list(self.active_connections.keys()):
            if client_id not in exclude:
                await self.send_to_client(client_id, message)
    
    async def broadcast_to_type(self, client_type: str, message: WebSocketMessage):
        """广播消息到指定类型的客户端"""
        room_id = f"type:{client_type}"
        await self.send_to_room(room_id, message)
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
    
    async def handle_message(self, client_id: str, message_str: str):
        """处理收到的消息"""
        self.total_messages_received += 1
        
        try:
            message = WebSocketMessage.from_json(message_str)
            message.sender_id = client_id
            
            # 更新心跳时间
            if client_id in self.active_connections:
                self.active_connections[client_id].last_heartbeat = datetime.now()
            
            # 心跳消息
            if message.type == MessageType.HEARTBEAT:
                await self.send_to_client(client_id, WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={"status": "ok"}
                ))
                return
            
            # 调用注册的处理器
            if message.type in self.message_handlers:
                handler = self.message_handlers[message.type]
                await handler(client_id, message)
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await self.send_to_client(client_id, WebSocketMessage(
                type=MessageType.ERROR,
                data={"error": str(e)}
            ))
    
    def get_client(self, client_id: str) -> Optional[ClientInfo]:
        """获取客户端信息"""
        return self.active_connections.get(client_id)
    
    def get_clients_by_type(self, client_type: str) -> List[ClientInfo]:
        """获取指定类型的所有客户端"""
        return [
            client for client in self.active_connections.values()
            if client.client_type == client_type
        ]
    
    def get_workstation_clients(self, workstation_id: int) -> List[ClientInfo]:
        """获取工位的所有客户端"""
        client_ids = self.workstation_subscriptions.get(workstation_id, set())
        return [
            self.active_connections[cid]
            for cid in client_ids
            if cid in self.active_connections
        ]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.total_connections,
            "rooms": len(self.rooms),
            "workstation_subscriptions": len(self.workstation_subscriptions),
            "messages_sent": self.total_messages_sent,
            "messages_received": self.total_messages_received,
            "clients_by_type": {
                ctype: len([c for c in self.active_connections.values() if c.client_type == ctype])
                for ctype in set(c.client_type for c in self.active_connections.values())
            }
        }


class TrainingFeedbackBroadcaster:
    """训练反馈广播器"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
    
    async def send_training_feedback(
        self,
        workstation_id: int,
        feedback_data: Dict
    ):
        """发送训练反馈"""
        message = WebSocketMessage(
            type=MessageType.TRAINING_FEEDBACK,
            data={
                "workstation_id": workstation_id,
                **feedback_data
            }
        )
        
        # 发送到工位订阅者
        await self.manager.send_to_workstation(workstation_id, message)
        
        # 同时发送到教官
        await self.manager.broadcast_to_type("instructor", message)
    
    async def send_score_update(
        self,
        workstation_id: int,
        student_id: int,
        scores: Dict[str, float]
    ):
        """发送分数更新"""
        message = WebSocketMessage(
            type=MessageType.SCORE_UPDATE,
            data={
                "workstation_id": workstation_id,
                "student_id": student_id,
                "scores": scores
            }
        )
        
        await self.manager.send_to_workstation(workstation_id, message)
        await self.manager.broadcast_to_type("instructor", message)
    
    async def send_danger_warning(
        self,
        workstation_id: int,
        warning_type: str,
        message_text: str
    ):
        """发送危险警告"""
        message = WebSocketMessage(
            type=MessageType.DANGER_WARNING,
            data={
                "workstation_id": workstation_id,
                "warning_type": warning_type,
                "message": message_text,
                "urgent": True
            }
        )
        
        # 广播到所有相关方
        await self.manager.send_to_workstation(workstation_id, message)
        await self.manager.broadcast_to_type("instructor", message)
        await self.manager.broadcast_to_type("admin", message)
    
    async def send_stage_complete(
        self,
        workstation_id: int,
        student_id: int,
        stage: str,
        stats: Dict
    ):
        """发送阶段完成通知"""
        message = WebSocketMessage(
            type=MessageType.STAGE_COMPLETE,
            data={
                "workstation_id": workstation_id,
                "student_id": student_id,
                "stage": stage,
                "stats": stats
            }
        )
        
        await self.manager.send_to_workstation(workstation_id, message)
        await self.manager.broadcast_to_type("instructor", message)
        await self.manager.broadcast_to_type("leader", message)


# 创建全局连接管理器实例
manager = ConnectionManager()
broadcaster = TrainingFeedbackBroadcaster(manager)


# FastAPI WebSocket端点示例
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    client_type: str = "student",
    workstation_id: Optional[int] = None
):
    """
    WebSocket端点
    
    使用示例:
        ws://localhost:8000/ws/{client_id}?client_type=student&workstation_id=1
    """
    await manager.connect(
        websocket,
        client_id,
        client_type=client_type,
        workstation_id=workstation_id
    )
    
    try:
        while True:
            data = await websocket.receive_text()
            await manager.handle_message(client_id, data)
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        await manager.disconnect(client_id)

