import time
import sqlite3
from typing import Any, Dict, List

class PerformanceProfiler:
    def __init__(self, db_path: str = 'performance_data.db'):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self) -> None:
        with self.connection:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    response_time REAL
                )
            ''')

    def log_performance(self, cpu_usage: float, memory_usage: float, response_time: float) -> None:
        with self.connection:
            self.connection.execute('''
                INSERT INTO performance_metrics (timestamp, cpu_usage, memory_usage, response_time)
                VALUES (?, ?, ?, ?)
            ''', (time.time(), cpu_usage, memory_usage, response_time))

    def get_performance_data(self) -> List[Dict[str, Any]]:
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM performance_metrics')
        rows = cursor.fetchall()
        return [
            {
                'id': row[0],
                'timestamp': row[1],
                'cpu_usage': row[2],
                'memory_usage': row[3],
                'response_time': row[4]
            }
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()