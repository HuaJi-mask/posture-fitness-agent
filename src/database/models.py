"""
数据库模型
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from src.database.connection import Base


def _utcnow():
    """当前UTC时间（naive，兼容SQLite存储，避免utcnow弃用警告）"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Session(Base):
    """会话表"""
    __tablename__ = "sessions"

    session_id = Column(String(64), primary_key=True, index=True)
    user_profile = Column(Text, default="{}")  # JSON字符串，配合crud里的json.dumps写入
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # 关联消息表
    message = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    plans = relationship("Plan", back_populates="session", cascade="all, delete-orphan")

    title = Column(String(64), default="新对话")  # 会话标题，用于显示在会话列表中
    last_message = Column(String(256), default="")  # 最后一条消息，用于显示在会话列表中

class Message(Base):
    """消息表"""
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"))
    role = Column(String(16))   # 角色，user 或 assistant
    content = Column(Text)
    created_at = Column(DateTime, default=_utcnow)

    session = relationship("Session", back_populates="message")


class Plan(Base):
    """计划表"""
    __tablename__ = "plans"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), ForeignKey("sessions.session_id"))
    training_plan = Column(Text, default="")
    diet_plan = Column(Text, default="")
    validation_report = Column(Text, default="")
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=_utcnow)

    session = relationship("Session", back_populates="plans")
