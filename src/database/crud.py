"""
数据库增删改查操作
"""

import json
from typing import List
from sqlalchemy.orm import Session

from src.database.models import _utcnow, Session as DBSession, Message, Plan


# ==================== 会话操作 ====================
def get_or_create_session(db: Session, session_id: str) -> DBSession:
    """获取或创建会话"""
    session = db.query(DBSession).filter(DBSession.session_id == session_id).first()
    if not session:
        session = DBSession(session_id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session

def update_session_profile(db: Session, session_id: str, user_profile: dict):
    """更新会话用户信息"""
    session = get_or_create_session(db, session_id)
    session.user_profile = json.dumps(user_profile, ensure_ascii=False)
    session.updated_at = _utcnow()
    db.commit()

def get_session_profile(db: Session, session_id: str) -> dict:
    """获取会话用户信息"""
    session = db.query(DBSession).filter(DBSession.session_id == session_id).first()
    if session:
        return json.loads(session.user_profile) if session.user_profile else {}
    return {}


def get_all_sessions(db: Session) -> List[dict]:
    """获取所有会话列表（按更新时间倒序）"""
    sessions = db.query(DBSession).order_by(DBSession.updated_at.desc()).all()
    result = []
    for session in sessions:
        result.append({
            "session_id": session.session_id,
            "title": session.title or "新对话",
            "last_message": session.last_message or "",
            "created_at": session.created_at.strftime("%Y-%m-%d %H:%M:%S") if session.created_at else "",
            "updated_at": session.updated_at.strftime("%Y-%m-%d %H:%M:%S") if session.updated_at else "",
        })
    return result


def delete_session(db: Session, session_id: str):
    """删除会话"""
    session = db.query(DBSession).filter(DBSession.session_id == session_id).first()
    if not session:
        return False
    db.delete(session)
    db.commit()
    return True


def update_session_info(db: Session, session_id: str, title: str=None, last_message: str=None):
    """更新会话的标题和最后消息"""
    session = db.query(DBSession).filter(DBSession.session_id == session_id).first()
    if not session:
        return
    
    if title is not None:
        if session.title == "新对话" or not session.title:
            session.title = title[:50]

    if last_message is not None:
        session.last_message = last_message[:100]

    session.updated_at = _utcnow()
    db.commit()


# ==================== 消息操作 ====================
def add_message(db: Session, session_id: str, role: str, content: str):
    """添加消息"""
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message

def get_messages(db: Session, session_id: str) -> list:
    """获取会话消息"""
    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at).all()
    return [{"role": msg.role, "content": msg.content} for msg in messages]


# ==================== 方案操作 ====================
def save_plan(db: Session, session_id: str, training_plan: str, diet_plan: str, validation_report: str):
    """保存方案"""

    # 计算版本号
    latest_plan = db.query(Plan).filter(Plan.session_id == session_id).order_by(Plan.version.desc()).first()
    version = latest_plan.version + 1 if latest_plan else 1

    plan = Plan(
        session_id=session_id,
        training_plan=training_plan,
        diet_plan=diet_plan,
        validation_report=validation_report,
        version=version,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan

def get_latest_plan(db: Session, session_id: str) -> dict:
    """获取最新方案"""
    plan = db.query(Plan).filter(Plan.session_id == session_id).order_by(Plan.version.desc()).first()
    if plan:
        return {
            "training_plan": plan.training_plan,
            "diet_plan": plan.diet_plan,
            "validation_report": plan.validation_report,
            "version": plan.version,
        }
    return {}


