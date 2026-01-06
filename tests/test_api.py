"""
API接口测试
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthEndpoint:
    """健康检查端点测试"""
    
    def test_health_check(self):
        """测试健康检查接口"""
        # 由于数据库依赖，这里使用mock
        with patch('backend.api.main.init_db'):
            from backend.api.main import app
            
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
    
    def test_root_endpoint(self):
        """测试根路径"""
        with patch('backend.api.main.init_db'):
            from backend.api.main import app
            
            client = TestClient(app)
            response = client.get("/")
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "version" in data


class TestUserAPI:
    """用户API测试"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock数据库会话"""
        mock = MagicMock()
        return mock
    
    def test_create_user_schema(self):
        """测试用户创建Schema"""
        from backend.api.schemas import UserCreate
        
        user = UserCreate(
            name="测试用户",
            badge_number="TEST001",
            unit="测试单位",
            role="student"
        )
        
        assert user.name == "测试用户"
        assert user.badge_number == "TEST001"
        assert user.role == "student"
    
    def test_user_response_schema(self):
        """测试用户响应Schema"""
        from backend.api.schemas import UserResponse
        from datetime import datetime
        
        # 使用字典数据模拟
        user_data = {
            "id": 1,
            "name": "测试用户",
            "badge_number": "TEST001",
            "unit": "测试单位",
            "role": "student",
            "created_at": datetime.now()
        }
        
        user = UserResponse(**user_data)
        
        assert user.id == 1
        assert user.name == "测试用户"


class TestTrainingAPI:
    """训练API测试"""
    
    def test_session_create_schema(self):
        """测试场次创建Schema"""
        from backend.api.schemas import SessionCreate
        from datetime import date
        
        session = SessionCreate(
            session_date=date.today(),
            session_type="dry_fire",
            instructor_id=1
        )
        
        assert session.session_type == "dry_fire"
        assert session.instructor_id == 1
    
    def test_training_record_create_schema(self):
        """测试训练记录创建Schema"""
        from backend.api.schemas import TrainingRecordCreate
        
        record = TrainingRecordCreate(
            session_id=1,
            student_id=1,
            workstation_id=5,
            gun_type="92式手枪"
        )
        
        assert record.workstation_id == 5
        assert record.gun_type == "92式手枪"
    
    def test_training_progress_schema(self):
        """测试训练进度Schema"""
        from backend.api.schemas import TrainingProgressResponse
        from datetime import datetime
        
        progress_data = {
            "student_id": 1,
            "student_name": "张三",
            "workstation_id": 5,
            "current_stage": "dry_fire",
            "status": "in_progress",
            "practice_count": 10,
            "passed_count": 7,
            "average_score": 82.5,
            "last_score": 85.0,
            "qualified_for_next": False,
            "start_time": datetime.now(),
            "elapsed_time": 1800
        }
        
        progress = TrainingProgressResponse(**progress_data)
        
        assert progress.practice_count == 10
        assert progress.average_score == 82.5


class TestWarningAPI:
    """预警API测试"""
    
    def test_warning_response_schema(self):
        """测试预警响应Schema"""
        from backend.api.schemas import WarningResponse
        from datetime import datetime
        
        warning_data = {
            "id": 1,
            "record_id": 1,
            "timestamp": datetime.now(),
            "warning_level": "serious",
            "warning_type": "trigger_discipline",
            "description": "非射击状态手指接触扳机",
            "handled": False,
            "handler_id": None,
            "handle_time": None,
            "video_clip_id": None
        }
        
        warning = WarningResponse(**warning_data)
        
        assert warning.warning_level == "serious"
        assert warning.handled == False


class TestWebSocketMessage:
    """WebSocket消息测试"""
    
    def test_ws_message_schema(self):
        """测试WebSocket消息Schema"""
        from backend.api.schemas import WSMessage
        from datetime import datetime
        
        message = WSMessage(
            type="real_time_analysis",
            data={
                "workstation_id": 5,
                "overall_score": 85.5
            }
        )
        
        assert message.type == "real_time_analysis"
        assert message.data["workstation_id"] == 5


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

