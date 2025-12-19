from sqlalchemy import create_engine, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from datetime import datetime
from typing import Optional

# Setup Database
DATABASE_URL = "sqlite:///./emo_sense.db"

# Base Class Modern (SQLAlchemy 2.0 Style)
class Base(DeclarativeBase):
    pass

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    input_text: Mapped[str] = mapped_column(String)
    predicted_label: Mapped[str] = mapped_column(String(50))
    confidence_score: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(20))

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

def init_db():
    Base.metadata.create_all(bind=engine)

def log_prediction(text: str, label: str, confidence: float, version: str = "v2.0"):
    with Session(engine) as session:
        try:
            log_entry = PredictionLog(
                input_text=text,
                predicted_label=label,
                confidence_score=confidence,
                model_version=version
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            print(f"❌ Database Error: {e}")
            session.rollback()