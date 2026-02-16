import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    custom_signs = relationship('CustomSign', back_populates='user', cascade='all, delete-orphan')
    def to_dict(self):
        return {
            'id': str(self.id),
            'email': self.email,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'custom_signs_count': len(self.custom_signs) if self.custom_signs else 0
        }

class CustomSign(Base):
    __tablename__ = 'custom_signs'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    sign_name = Column(String(50), nullable=False)
    model_path = Column(Text, nullable=False)
    sample_count = Column(String(10), default='0')
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship('User', back_populates='custom_signs')
    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'sign_name': self.sign_name,
            'sample_count': self.sample_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TokenBlocklist(Base):
    __tablename__ = 'token_blocklist'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    jti = Column(String(36), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
