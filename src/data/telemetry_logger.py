"""
Telemetry Logger for the Urban Digital Twin.

Logs simulation step data to a SQLite database.
Exports aggregated results to CSV for Power BI dashboards.
"""

import sqlite3
import os
import csv
import pandas as pd
from typing import Dict, Any


class TelemetryLogger:
    """Logs simulation metrics and exports them to CSV for analytics."""

    def __init__(self, db_path: str = "data/telemetry.db", buffer_size: int = 100):
        self.db_path = db_path
        self.buffer_size = buffer_size
        self._step_buffer = []

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()

        # Episodes Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_type TEXT NOT NULL,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_steps INTEGER,
            total_reward REAL,
            avg_wait_time REAL,
            total_throughput INTEGER
        )
        ''')

        # Step Telemetry Table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS step_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INTEGER,
            step_number INTEGER,
            sim_time_sec REAL,
            action_taken INTEGER,
            current_phase TEXT,
            queue_north INTEGER,
            queue_south INTEGER,
            queue_east INTEGER,
            queue_west INTEGER,
            total_waiting INTEGER,
            total_cleared INTEGER,
            avg_wait_time REAL,
            reward REAL,
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        )
        ''')

        self.conn.commit()

    def start_episode(self, agent_type: str) -> int:
        """Log the start of a new episode and return its ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO episodes (agent_type) VALUES (?)",
            (agent_type,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_step(self, episode_id: int, step: int, action: int, reward: float, info: Dict[str, Any]):
        """Buffer a step's telemetry."""
        queues = info.get("queue_lengths", {})
        
        row = (
            episode_id,
            step,
            info.get("sim_time", 0.0),
            action,
            info.get("phase", ""),
            queues.get("NORTH", 0),
            queues.get("SOUTH", 0),
            queues.get("EAST", 0),
            queues.get("WEST", 0),
            info.get("total_waiting", 0),
            info.get("total_cleared", 0),
            info.get("avg_wait_time", 0.0),
            reward
        )
        
        self._step_buffer.append(row)

        if len(self._step_buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffered rows to the database."""
        if not self._step_buffer:
            return

        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO step_telemetry (
                episode_id, step_number, sim_time_sec, action_taken, current_phase,
                queue_north, queue_south, queue_east, queue_west,
                total_waiting, total_cleared, avg_wait_time, reward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', self._step_buffer)
        
        self.conn.commit()
        self._step_buffer.clear()

    def end_episode(self, episode_id: int, total_steps: int, total_reward: float, avg_wait: float, throughput: int):
        """Update the episode record with final totals."""
        self.flush()  # Ensure remaining steps are saved
        
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE episodes 
            SET total_steps = ?, total_reward = ?, avg_wait_time = ?, total_throughput = ?
            WHERE episode_id = ?
        ''', (total_steps, total_reward, avg_wait, throughput, episode_id))
        self.conn.commit()

    def export_to_csv(self, export_dir: str = "data/exports"):
        """Export database tables to CSV for Power BI."""
        os.makedirs(export_dir, exist_ok=True)
        
        # Using Pandas to quickly dump to CSV
        episodes_df = pd.read_sql_query("SELECT * FROM episodes", self.conn)
        episodes_df.to_csv(os.path.join(export_dir, "episodes.csv"), index=False)
        
        steps_df = pd.read_sql_query("SELECT * FROM step_telemetry", self.conn)
        steps_df.to_csv(os.path.join(export_dir, "step_telemetry.csv"), index=False)
        
        print(f"  Exported telemetry data to {export_dir}")

    def close(self):
        self.flush()
        self.conn.close()
