"""
SQLite-based checkpointing for ORYX pipeline crash recovery.

Enables resumption of long-running pipelines after crashes or interruptions.
Stores intermediate results, pipeline state, and metadata in a local SQLite database.

Usage:
    >>> cp = Checkpoint("pipeline_run_001", db_path=".cache/checkpoints.db")
    >>> cp.save_stage("scrape", scraped_data, {"urls": 150})
    >>> # ... later, after crash ...
    >>> cp = Checkpoint("pipeline_run_001")
    >>> if cp.has_stage("scrape"):
    ...     scraped_data = cp.load_stage("scrape")
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_DB_PATH = ".cache/oryx_checkpoints.db"

PIPELINE_STAGES = [
    "seed_keywords",      # Initial keyword list
    "expand_keywords",    # LLM/autocomplete expansion
    "scrape_serps",       # SERP scraping results
    "extract_entities",   # Entity extraction
    "compute_metrics",    # Metrics calculation
    "cluster_keywords",   # Clustering results
    "classify_intent",    # Intent classification
    "score_opportunities", # Opportunity scoring
    "generate_output",    # Final output generation
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage_name: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    item_count: int = 0
    duration_ms: int = 0


@dataclass
class PipelineRun:
    """Metadata about a pipeline run."""
    run_id: str
    created_at: str
    updated_at: str
    status: str  # "running", "completed", "failed", "interrupted"
    config_hash: Optional[str] = None
    current_stage: Optional[str] = None
    total_keywords: int = 0
    completed_stages: List[str] = field(default_factory=list)


# ============================================================================
# Checkpoint Manager
# ============================================================================

class Checkpoint:
    """
    SQLite-based checkpoint manager for pipeline crash recovery.
    
    Stores pipeline state and intermediate results in a local SQLite database.
    Enables resumption of long-running pipelines after crashes.
    
    Attributes:
        run_id: Unique identifier for this pipeline run
        db_path: Path to SQLite database file
    
    Example:
        >>> cp = Checkpoint("run_20240115_001")
        >>> 
        >>> # Save intermediate results
        >>> cp.save_stage("scrape", scraped_data, {"urls_processed": 100})
        >>> 
        >>> # Resume after crash
        >>> if cp.has_stage("scrape"):
        ...     data = cp.load_stage("scrape")
        >>> 
        >>> # Mark completion
        >>> cp.mark_completed()
    """
    
    def __init__(
        self, 
        run_id: str, 
        db_path: str = DEFAULT_DB_PATH,
        auto_create: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            run_id: Unique identifier for this pipeline run
            db_path: Path to SQLite database file
            auto_create: Whether to create the database if it doesn't exist
        """
        self.run_id = run_id
        self.db_path = Path(db_path)
        
        if auto_create:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self._ensure_run_exists()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Pipeline runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT DEFAULT 'running',
                    config_hash TEXT,
                    current_stage TEXT,
                    total_keywords INTEGER DEFAULT 0
                )
            """)
            
            # Stage results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stage_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    metadata_json TEXT,
                    timestamp TEXT NOT NULL,
                    item_count INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    UNIQUE(run_id, stage_name),
                    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
                )
            """)
            
            # Index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stage_results_run_stage 
                ON stage_results(run_id, stage_name)
            """)
            
            conn.commit()
    
    def _connect(self) -> sqlite3.Connection:
        """Create database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def _ensure_run_exists(self) -> None:
        """Ensure run record exists in database."""
        with self._connect() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            
            cursor.execute("""
                INSERT OR IGNORE INTO pipeline_runs (run_id, created_at, updated_at)
                VALUES (?, ?, ?)
            """, (self.run_id, now, now))
            
            conn.commit()
    
    # ========================================================================
    # Stage Management
    # ========================================================================
    
    def save_stage(
        self, 
        stage_name: str, 
        data: Any, 
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0
    ) -> None:
        """
        Save stage results to checkpoint.
        
        Args:
            stage_name: Name of the pipeline stage
            data: Data to save (must be JSON-serializable)
            metadata: Optional metadata about the stage execution
            duration_ms: Execution duration in milliseconds
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            
            # Serialize data
            data_json = json.dumps(data, default=str)
            metadata_json = json.dumps(metadata or {})
            item_count = len(data) if isinstance(data, (list, dict)) else 1
            
            # Upsert stage result
            cursor.execute("""
                INSERT INTO stage_results 
                    (run_id, stage_name, data_json, metadata_json, timestamp, item_count, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, stage_name) DO UPDATE SET
                    data_json = excluded.data_json,
                    metadata_json = excluded.metadata_json,
                    timestamp = excluded.timestamp,
                    item_count = excluded.item_count,
                    duration_ms = excluded.duration_ms
            """, (self.run_id, stage_name, data_json, metadata_json, now, item_count, duration_ms))
            
            # Update run metadata
            cursor.execute("""
                UPDATE pipeline_runs 
                SET updated_at = ?, current_stage = ?
                WHERE run_id = ?
            """, (now, stage_name, self.run_id))
            
            conn.commit()
            logger.info(f"Checkpointed stage '{stage_name}' ({item_count} items)")
    
    def load_stage(self, stage_name: str) -> Optional[Any]:
        """
        Load stage results from checkpoint.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Deserialized data, or None if stage not found
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data_json FROM stage_results
                WHERE run_id = ? AND stage_name = ?
            """, (self.run_id, stage_name))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
    
    def has_stage(self, stage_name: str) -> bool:
        """Check if a stage has been checkpointed."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 1 FROM stage_results
                WHERE run_id = ? AND stage_name = ?
            """, (self.run_id, stage_name))
            
            return cursor.fetchone() is not None
    
    def get_stage_metadata(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a checkpointed stage."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT metadata_json, timestamp, item_count, duration_ms
                FROM stage_results
                WHERE run_id = ? AND stage_name = ?
            """, (self.run_id, stage_name))
            
            row = cursor.fetchone()
            if row:
                return {
                    "metadata": json.loads(row[0]) if row[0] else {},
                    "timestamp": row[1],
                    "item_count": row[2],
                    "duration_ms": row[3]
                }
            return None
    
    def get_completed_stages(self) -> List[str]:
        """Get list of completed stage names."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT stage_name FROM stage_results
                WHERE run_id = ?
                ORDER BY id
            """, (self.run_id,))
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_last_stage(self) -> Optional[str]:
        """Get the name of the last completed stage."""
        stages = self.get_completed_stages()
        return stages[-1] if stages else None
    
    # ========================================================================
    # Run Management
    # ========================================================================
    
    def get_run_info(self) -> Optional[PipelineRun]:
        """Get information about this pipeline run."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT run_id, created_at, updated_at, status, config_hash, 
                       current_stage, total_keywords
                FROM pipeline_runs
                WHERE run_id = ?
            """, (self.run_id,))
            
            row = cursor.fetchone()
            if row:
                return PipelineRun(
                    run_id=row[0],
                    created_at=row[1],
                    updated_at=row[2],
                    status=row[3],
                    config_hash=row[4],
                    current_stage=row[5],
                    total_keywords=row[6] or 0,
                    completed_stages=self.get_completed_stages()
                )
            return None
    
    def mark_completed(self) -> None:
        """Mark the pipeline run as completed."""
        self._update_status("completed")
        logger.info(f"Pipeline run '{self.run_id}' marked as completed")
    
    def mark_failed(self, error_msg: Optional[str] = None) -> None:
        """Mark the pipeline run as failed."""
        self._update_status("failed")
        if error_msg:
            # Store error in metadata
            self.save_stage("_error", {"error": error_msg})
        logger.warning(f"Pipeline run '{self.run_id}' marked as failed")
    
    def _update_status(self, status: str) -> None:
        """Update run status."""
        with self._connect() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            
            cursor.execute("""
                UPDATE pipeline_runs 
                SET status = ?, updated_at = ?
                WHERE run_id = ?
            """, (status, now, self.run_id))
            
            conn.commit()
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    
    def clear(self) -> None:
        """Clear all data for this run."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM stage_results WHERE run_id = ?", (self.run_id,))
            cursor.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (self.run_id,))
            
            conn.commit()
            logger.info(f"Cleared checkpoint data for run '{self.run_id}'")
    
    @classmethod
    def list_runs(cls, db_path: str = DEFAULT_DB_PATH) -> List[PipelineRun]:
        """List all pipeline runs in the database."""
        db_file = Path(db_path)
        if not db_file.exists():
            return []
        
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT run_id, created_at, updated_at, status, config_hash, 
                       current_stage, total_keywords
                FROM pipeline_runs
                ORDER BY updated_at DESC
            """)
            
            runs = []
            for row in cursor.fetchall():
                runs.append(PipelineRun(
                    run_id=row[0],
                    created_at=row[1],
                    updated_at=row[2],
                    status=row[3],
                    config_hash=row[4],
                    current_stage=row[5],
                    total_keywords=row[6] or 0
                ))
            return runs
        finally:
            conn.close()
    
    @classmethod
    def cleanup_old_runs(
        cls, 
        db_path: str = DEFAULT_DB_PATH,
        keep_last: int = 10,
        keep_completed: bool = True
    ) -> int:
        """
        Clean up old pipeline runs.
        
        Args:
            db_path: Path to database
            keep_last: Number of recent runs to keep
            keep_completed: Whether to keep all completed runs
            
        Returns:
            Number of runs deleted
        """
        db_file = Path(db_path)
        if not db_file.exists():
            return 0
        
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        try:
            # Get runs to delete
            if keep_completed:
                cursor.execute("""
                    SELECT run_id FROM pipeline_runs
                    WHERE status != 'completed'
                    ORDER BY updated_at DESC
                    LIMIT -1 OFFSET ?
                """, (keep_last,))
            else:
                cursor.execute("""
                    SELECT run_id FROM pipeline_runs
                    ORDER BY updated_at DESC
                    LIMIT -1 OFFSET ?
                """, (keep_last,))
            
            run_ids = [row[0] for row in cursor.fetchall()]
            
            if run_ids:
                placeholders = ",".join("?" * len(run_ids))
                cursor.execute(f"DELETE FROM stage_results WHERE run_id IN ({placeholders})", run_ids)
                cursor.execute(f"DELETE FROM pipeline_runs WHERE run_id IN ({placeholders})", run_ids)
                conn.commit()
                
            return len(run_ids)
        finally:
            conn.close()


# ============================================================================
# Context Manager for Automatic Checkpointing
# ============================================================================

class CheckpointedPipeline:
    """
    Context manager for automatic pipeline checkpointing.
    
    Wraps pipeline execution with automatic crash recovery.
    
    Example:
        >>> with CheckpointedPipeline("run_001") as cp:
        ...     if not cp.has_stage("scrape"):
        ...         data = scrape_serps(keywords)
        ...         cp.save_stage("scrape", data)
        ...     else:
        ...         data = cp.load_stage("scrape")
    """
    
    def __init__(self, run_id: str, db_path: str = DEFAULT_DB_PATH):
        self.checkpoint = Checkpoint(run_id, db_path)
        self._start_time = None
    
    def __enter__(self) -> Checkpoint:
        self._start_time = datetime.utcnow()
        logger.info(f"Starting checkpointed pipeline run: {self.checkpoint.run_id}")
        return self.checkpoint
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.checkpoint.mark_completed()
        else:
            self.checkpoint.mark_failed(str(exc_val))
        return False  # Don't suppress exceptions
