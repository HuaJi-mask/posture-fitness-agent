"""
数据库连接
"""

import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# SQLite数据库文件路径（基于本文件拼绝对路径，避免受启动目录影响）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'app.db')}"

# 创建引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    connect_args={
        "check_same_thread": False  # SQLite 多线程需要这个
    }
)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类（SQLAlchemy 2.0 推荐写法）
class Base(DeclarativeBase):
    pass


def get_db():
    """获取数据库会话（FastAPI 依赖注入用）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """初始化数据库"""
    Base.metadata.create_all(bind=engine)

    # 轻量迁移：老库的 sessions 表可能没有 title/last_message 列，补上（保留原数据）
    inspector = inspect(engine)
    if "sessions" in inspector.get_table_names():
        columns = [c["name"] for c in inspector.get_columns("sessions")]
        with engine.begin() as conn:
            if "title" not in columns:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN title VARCHAR(64) DEFAULT '新对话'"))
            if "last_message" not in columns:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN last_message VARCHAR(256) DEFAULT ''"))