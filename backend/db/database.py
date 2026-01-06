"""
数据库连接和会话管理
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import os

from .models import Base

# 数据库连接配置
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/gun_training"
)

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 检测连接是否有效
    pool_size=10,  # 连接池大小
    max_overflow=20,  # 超过pool_size后最多创建的连接数
    echo=False  # 生产环境设为False
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db():
    """初始化数据库（创建所有表）"""
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表创建成功")


def drop_db():
    """删除所有表（谨慎使用！）"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️ 数据库表已删除")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话的上下文管理器
    
    用法：
        with get_db() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_db_session() -> Session:
    """
    获取数据库会话（用于依赖注入）
    
    用法（FastAPI）：
        @app.get("/users")
        def get_users(db: Session = Depends(get_db_session)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # 在FastAPI中会自动关闭


# MongoDB配置（用于存储视频片段）
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = "gun_training_videos"

try:
    from pymongo import MongoClient
    
    mongo_client = MongoClient(MONGODB_URL)
    mongodb = mongo_client[MONGODB_DB_NAME]
    
    # 视频片段集合
    video_clips_collection = mongodb["video_clips"]
    
    print("✅ MongoDB连接成功")
except Exception as e:
    print(f"⚠️ MongoDB连接失败: {e}")
    mongo_client = None
    mongodb = None
    video_clips_collection = None


# Redis配置（用于消息队列和缓存）
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    import redis
    
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    
    print("✅ Redis连接成功")
except Exception as e:
    print(f"⚠️ Redis连接失败: {e}")
    redis_client = None


if __name__ == "__main__":
    # 测试数据库连接
    print("测试数据库连接...")
    
    try:
        # 初始化数据库
        init_db()
        
        # 测试插入数据
        from .models import User
        
        with get_db() as db:
            # 创建测试用户
            test_user = User(
                name="张三",
                badge_number="001001",
                unit="市公安局",
                role="student"
            )
            db.add(test_user)
            db.commit()
            
            # 查询
            user = db.query(User).filter_by(badge_number="001001").first()
            print(f"✅ 测试用户创建成功: {user}")
            
            # 删除测试用户
            db.delete(user)
            db.commit()
            print("✅ 测试用户删除成功")
        
        print("\n✅ 数据库连接测试通过！")
        
    except Exception as e:
        print(f"\n❌ 数据库连接测试失败: {e}")

