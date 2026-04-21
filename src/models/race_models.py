"""
Data models for horse racing data.
Uses SQLAlchemy for persistence and DuckDB for analytics.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Date, DateTime, Text, Boolean, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class RaceDay(Base):
    """A single race meeting day."""
    __tablename__ = 'race_days'

    id = Column(Integer, primary_key=True)
    race_date = Column(Date, nullable=False, unique=True)
    venue = Column(String(4), nullable=False)  # "ST" or "HV"
    total_races = Column(Integer, default=0)
    race_type = Column(String(10))  # "day" or "night"
    scraped_at = Column(DateTime, default=datetime.utcnow)

    races = relationship("Race", back_populates="race_day", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<RaceDay {self.race_date} {self.venue}>"


class Race(Base):
    """A single race within a race day."""
    __tablename__ = 'races'

    id = Column(Integer, primary_key=True)
    race_day_id = Column(Integer, ForeignKey('race_days.id'), nullable=False)
    race_no = Column(Integer, nullable=False)

    distance = Column(Integer)  # meters
    race_class = Column(String(20))  # "Class 1", "Class 2", etc.
    track = Column(String(10))  # "AWT", "Turf", "Dirt"
    prize_money = Column(String(50))
    race_time = Column(String(10))

    # Sectional times
    sectional_1 = Column(String(10))
    sectional_2 = Column(String(10))
    sectional_3 = Column(String(10))
    sectional_4 = Column(String(10))

    scraped_at = Column(DateTime, default=datetime.utcnow)

    race_day = relationship("RaceDay", back_populates="races")
    entries = relationship("HorseEntry", back_populates="race", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_race_day_raceno', 'race_day_id', 'race_no'),
    )

    def __repr__(self):
        return f"<Race {self.race_day_id} #{self.race_no} ({self.distance}m)>"


class HorseEntry(Base):
    """A horse entered in a race."""
    __tablename__ = 'horse_entries'

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False)

    horse_no = Column(Integer, nullable=False)  # Saddle cloth number
    draw = Column(Integer)  # Barrier draw
    horse_name = Column(String(100), nullable=False)
    horse_id = Column(String(20))  # HKJC horse ID

    jockey = Column(String(100))
    trainer = Column(String(100))
    jockey_weight = Column(String(20))  # e.g., "118 lb"
    rating = Column(Integer)  # Official rating

    # Pre-race stats
    win_odds = Column(Float)
    place_odds = Column(Float)

    # Result
    finishing_position = Column(Integer)
    finishing_time = Column(String(20))
    margin = Column(String(20))
    finish_type = Column(String(20))  # "normal", "disqualified", "retired"

    # Running style
    early_pace = Column(String(10))  # "front", "mid", "hold-up"

    scraped_at = Column(DateTime, default=datetime.utcnow)

    race = relationship("Race", back_populates="entries")

    __table_args__ = (
        Index('ix_race_horse', 'race_id', 'horse_no'),
    )

    def __repr__(self):
        return f"<HorseEntry {self.horse_no} {self.horse_name}>"


class Horse(Base):
    """Horse master data with historical statistics."""
    __tablename__ = 'horses'

    id = Column(Integer, primary_key=True)
    horse_id = Column(String(20), unique=True)  # HKJC ID
    name = Column(String(100), nullable=False)

    # Pedigree
    sire = Column(String(100))
    dam = Column(String(100))
    dam_sire = Column(String(100))  # Maternal grandsire

    # Physical
    colour = Column(String(20))
    sex = Column(String(10))  # "Gelding", "Mare", "Colt", "Filly"
    age = Column(Integer)

    # Career stats (updated periodically)
    total_runs = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    second = Column(Integer, default=0)
    third = Column(Integer, default=0)
    earnings = Column(String(30))

    # Current season
    season_runs = Column(Integer, default=0)
    season_wins = Column(Integer, default=0)

    scraped_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Horse {self.horse_id} {self.name}>"


class RaceResult(Base):
    """Official race result and dividends."""
    __tablename__ = 'race_results'

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False)

    # Dividends
    win_dividend = Column(String(20))
    place_dividend_1 = Column(String(20))
    place_dividend_2 = Column(String(20))
    place_dividend_3 = Column(String(20))

    # Quartet dividend
    quartet = Column(String(30))

    scraped_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_result_race', 'race_id'),
    )


def init_db(db_path: str):
    """Initialize database and return engine + session factory."""
    engine = create_engine(
        f'sqlite:///{db_path}',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session
