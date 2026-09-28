from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from src.database.connection import Base

class ApiMapping(Base):
    """用於對齊三大 API 實體 ID 的對照表"""
    __tablename__ = "api_mappings"

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String(20), nullable=False, index=True) # 'league', 'team', 'fixture'
    standard_id = Column(String(100), nullable=False, index=True)
    provider = Column(String(30), nullable=False)                 # 'football_data', 'api_football', 'odds_api'
    external_id = Column(String(100), nullable=False)

    __table_args__ = (
        UniqueConstraint('entity_type', 'provider', 'external_id', name='_entity_provider_ext_uc'),
    )


class League(Base):
    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True, index=True)
    standard_name = Column(String(100), unique=True, nullable=False)
    country = Column(String(50), nullable=False)

    fixtures = relationship("Fixture", back_populates="league")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    standard_name = Column(String(100), unique=True, nullable=False)
    logo_url = Column(String(255), nullable=True)

    home_fixtures = relationship("Fixture", foreign_keys="Fixture.home_team_id", back_populates="home_team")
    away_fixtures = relationship("Fixture", foreign_keys="Fixture.away_team_id", back_populates="away_team")


class Fixture(Base):
    __tablename__ = "fixtures"

    id = Column(Integer, primary_key=True, index=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    match_datetime = Column(DateTime, nullable=False, index=True)
    status = Column(String(20), default="NS", index=True)

    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    ht_home_score = Column(Integer, nullable=True)
    ht_away_score = Column(Integer, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    league = relationship("League", back_populates="fixtures")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_fixtures")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_fixtures")

    odds_history = relationship("OddsHistory", back_populates="fixture", cascade="all, delete-orphan")
    live_stats = relationship("MatchStats", back_populates="fixture", uselist=False)
    predictions = relationship("Prediction", back_populates="fixture", cascade="all, delete-orphan")
    ai_reports = relationship("AiReport", back_populates="fixture", cascade="all, delete-orphan")


class OddsHistory(Base):
    __tablename__ = "odds_history"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=False, index=True)
    bookmaker = Column(String(50), nullable=False, index=True)
    market_type = Column(String(20), nullable=False)

    handicap = Column(Float, nullable=True)
    home_odds = Column(Float, nullable=True)
    draw_odds = Column(Float, nullable=True)
    away_odds = Column(Float, nullable=True)
    over_odds = Column(Float, nullable=True)
    under_odds = Column(Float, nullable=True)

    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    fixture = relationship("Fixture", back_populates="odds_history")


class MatchStats(Base):
    __tablename__ = "match_stats"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), unique=True, nullable=False)

    home_possession = Column(Integer, nullable=True)
    away_possession = Column(Integer, nullable=True)
    home_shots_on_target = Column(Integer, nullable=True)
    away_shots_on_target = Column(Integer, nullable=True)
    home_corners = Column(Integer, nullable=True)
    away_corners = Column(Integer, nullable=True)
    home_yellow_cards = Column(Integer, nullable=True)
    away_yellow_cards = Column(Integer, nullable=True)
    home_red_cards = Column(Integer, nullable=True)
    away_red_cards = Column(Integer, nullable=True)

    fixture = relationship("Fixture", back_populates="live_stats")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=False)

    model_version = Column(String(50), nullable=False)
    prob_home_win = Column(Float, nullable=False)
    prob_draw = Column(Float, nullable=False)
    prob_away_win = Column(Float, nullable=False)

    value_bet_detected = Column(Boolean, default=False)
    recommended_pick = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    fixture = relationship("Fixture", back_populates="predictions")


class AiReport(Base):
    __tablename__ = "ai_reports"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=False)

    report_title = Column(String(200), nullable=False)
    report_content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    fixture = relationship("Fixture", back_populates="ai_reports")
