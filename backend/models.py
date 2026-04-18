import uuid
import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))
    total_practice_seconds = Column(Integer, default=0)
    global_rank_score = Column(Float, default=0.0)

    entries = relationship("AudioEntry", back_populates="user", cascade="all, delete-orphan")
    learning_progress = relationship("UserLearningProgress", back_populates="user", cascade="all, delete-orphan")

class AudioEntry(Base):
    __tablename__ = "audio_entries"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    target_word = Column(String(255), nullable=False)
    audio_storage_url = Column(String(1024), nullable=True)
    
    overall_score = Column(Float, nullable=False, default=0.0)
    phoneme_score = Column(Float)
    duration_score = Column(Float)
    pitch_score = Column(Float)
    stress_score = Column(Float)
    
    pitch_trajectory = Column(JSONB)
    phoneme_alignment = Column(JSONB)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True)

    user = relationship("User", back_populates="entries")

class LearningPath(Base):
    __tablename__ = "learning_paths"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    focus_category = Column(String(100))
    color_theme = Column(String(50))

class UserLearningProgress(Base):
    __tablename__ = "user_learning_progress"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    path_id = Column(UUID(as_uuid=True), ForeignKey("learning_paths.id", ondelete="CASCADE"), index=True)
    progress_percentage = Column(Float, default=0.0)
    status = Column(String(50), default="in_progress")
    last_practiced_at = Column(DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc))

    user = relationship("User", back_populates="learning_progress")
    path = relationship("LearningPath")
