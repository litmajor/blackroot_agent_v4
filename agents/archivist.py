#!/usr/bin/env python3
"""
Blackroot Archivist Agent (BR-ARCHIVIST)
=====================================

The immutable memory and historical backbone of the Blackroot organism.
Maintains cryptographically secure, append-only records of all agent activities.

Features:
- Merkle tree-based tamper evidence
- Cryptographic integrity verification
- Historical intelligence and trend analysis
- Cross-agent interaction logging
- Audit trail preservation
"""

import hashlib
import json
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hmac
import base64
import sqlite3
from pathlib import Path
import secrets
from collections import defaultdict
import logging

class RecordType(Enum):
    """Categories of archival records"""
    AGENT_ACTION = "agent_action"
    TRANSACTION = "transaction" 
    ENFORCEMENT = "enforcement"
    COORDINATION = "coordination"
    REFLECTION = "reflection"
    REPAIR = "repair"
    SYSTEM_EVENT = "system_event"

class Priority(Enum):
    """Record priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class ArchivalRecord:
    """Immutable record structure"""
    record_id: str
    timestamp: float
    agent_code: str
    record_type: RecordType
    priority: Priority
    data: Dict[str, Any]
    previous_hash: str
    content_hash: str
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp,
            'agent_code': self.agent_code,
            'record_type': self.record_type.value,
            'priority': self.priority.value,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'content_hash': self.content_hash,
            'signature': self.signature
        }

class MerkleNode:
    """Node in Merkle tree for cryptographic integrity"""
    def __init__(self, hash_value: str, left=None, right=None, data=None):
        self.hash = hash_value
        self.left = left
        self.right = right
        self.data = data  # For leaf nodes
        self.is_leaf = data is not None

class MerkleTree:
    """Complete Merkle tree implementation for batch verification"""
    
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
        self.leaves = []
        self.root = None
        self._build_tree()
    
    def _hash_data(self, data: Union[str, bytes]) -> str:
        """Hash data consistently"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def _build_tree(self):
        """Build Merkle tree from records"""
        if not self.records:
            self.root = MerkleNode("0" * 64)  # Empty tree hash
            return
        
        # Create leaf nodes
        self.leaves = []
        for record in self.records:
            record_data = json.dumps(record, sort_keys=True)
            leaf_hash = self._hash_data(record_data)
            leaf_node = MerkleNode(leaf_hash, data=record)
            self.leaves.append(leaf_node)
        
        # Build tree bottom-up
        current_level = self.leaves[:]
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Odd number of nodes - duplicate last node
                    right = current_level[i]
                
                # Create parent node
                combined_hash = self._hash_data(left.hash + right.hash)
                parent = MerkleNode(combined_hash, left, right)
                next_level.append(parent)
            
            current_level = next_level
        
        self.root = current_level[0]
    
    def get_root_hash(self) -> str:
        """Get Merkle root hash"""
        return self.root.hash if self.root else "0" * 64
    
    def get_proof(self, record_index: int) -> List[Tuple[str, str]]:
        """
        Get Merkle proof for record at given index
        Returns list of (hash, direction) tuples where direction is 'left' or 'right'
        """
        if record_index >= len(self.leaves):
            raise IndexError("Record index out of range")
        
        proof = []
        current_node = self.leaves[record_index]
        current_level = self.leaves[:]
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                
                # Create parent
                combined_hash = self._hash_data(left.hash + right.hash)
                parent = MerkleNode(combined_hash, left, right)
                next_level.append(parent)
                
                # If current node is one of these children, add sibling to proof
                if current_node == left:
                    proof.append((right.hash, 'right'))
                    current_node = parent
                elif current_node == right:
                    proof.append((left.hash, 'left'))
                    current_node = parent
            
            current_level = next_level
        
        return proof
    
    def verify_proof(self, record: Dict[str, Any], proof: List[Tuple[str, str]], root_hash: str) -> bool:
        """Verify Merkle proof for a record"""
        record_data = json.dumps(record, sort_keys=True)
        current_hash = self._hash_data(record_data)
        
        for sibling_hash, direction in proof:
            if direction == 'left':
                current_hash = self._hash_data(sibling_hash + current_hash)
            else:
                current_hash = self._hash_data(current_hash + sibling_hash)
        
        return current_hash == root_hash

@dataclass
class Segment:
    """Archival segment containing multiple records"""
    id: str
    created_at: str
    records_count: int
    metadata: Dict[str, Any]
    content_digest: str
    previous_segment_digest: str
    records: List[Dict[str, Any]]

@dataclass 
class Epoch:
    """Epoch containing multiple segments with Merkle root"""
    id: str
    start_segment: str
    end_segment: str
    merkle_root: str
    signed_checkpoint: Dict[str, str]
    segments: List[str]

@dataclass
class SignedCheckpoint:
    """Cryptographically signed checkpoint"""
    epoch_id: str
    merkle_root: str
    timestamp: float
    signer_id: str
    signature: str
    metadata: Dict[str, Any]

class ArchivistAgent:
    """
    BR-ARCHIVIST: The immutable memory custodian
    
    Responsibilities:
    - Store append-only logs of all agent actions
    - Maintain cryptographic proof chains
    - Provide historical intelligence
    - Support audit and compliance operations
    """
    
    def __init__(self, agent_id: str, secret_key: str, governance_key: str = "", db_path: str = "blackroot_archive.db"):
        self.agent_id = agent_id
        self.code = "BR-ARCHIVIST"
        self.secret_key = secret_key.encode('utf-8')
        self.governance_key = governance_key.encode('utf-8') if governance_key else None
        self.db_path = db_path

        # Economic Layer: Wallet registration
        try:
            from agents.accountant import ACCLedger
            from decimal import Decimal
            self.accountant = ACCLedger(creator_id="SYSTEM")
            self.accountant.register_agent(str(self.agent_id), Decimal('120'))  # Initial budget
        except Exception as e:
            self.accountant = None
            print(f"Failed to initialize accountant for ArchivistAgent: {e}")

        # Thread safety
        self.lock = threading.RLock()

        # Archive state
        self.last_hash = "0" * 64  # Genesis hash
        self.record_count = 0
        self.current_segment_id = None
        self.current_epoch_id = None

        # Anchoring configuration
        self.segment_size_limit = 1000  # Records per segment
        self.epoch_size_limit = 10     # Segments per epoch
        self.auto_checkpoint = True

        # Storage
        self.segments: Dict[str, Segment] = {}
        self.epochs: Dict[str, Epoch] = {}
        self.merkle_trees: Dict[str, MerkleTree] = {}
        self.signed_checkpoints: Dict[str, SignedCheckpoint] = {}

        # Agent interaction tracking
        self.agent_states: Dict[str, Dict] = {}
        self.interaction_logs: Dict[str, List] = {}

        # Replication tracking
        self.replica_digests: Dict[str, Dict[str, str]] = {}  # replica_id -> {segment_id: digest}
        self.consensus_threshold = 0.67  # 67% consensus required

        # Initialize database
        self._init_database()
        self._load_state()

        print(f"🗄️ Archivist Agent {self.agent_id} initialized")
        print(f"📊 Archive contains {self.record_count} records")
        print(f"🔐 Governance signatures: {'ENABLED' if self.governance_key else 'DISABLED'}")

    def _deduct_archive_cost(self):
        """Deduct Root for archiving a record (0.25 Root per record)."""
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.process_agent_spend(str(self.agent_id), Decimal('0.25'), "Archive record cost")

    def _reward_archive_success(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            self.accountant.agent_wallets[str(self.agent_id)].balance += Decimal('0.5')
            self.accountant.treasury_balance -= Decimal('0.5')
            print(f"Rewarded 0.5 Root to ArchivistAgent for successful archival.")

    def _burn_on_archive_failure(self):
        if hasattr(self, 'accountant') and self.accountant:
            from decimal import Decimal
            penalty = Decimal('0.4')
            self.accountant.agent_wallets[str(self.agent_id)].balance -= penalty
            print(f"Burned 0.4 Root from ArchivistAgent for failed archival.")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archive_records (
                    record_id TEXT PRIMARY KEY,
                    segment_id TEXT,
                    timestamp REAL,
                    agent_code TEXT,
                    record_type TEXT,
                    priority INTEGER,
                    data TEXT,
                    previous_hash TEXT,
                    content_hash TEXT,
                    signature TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    records_count INTEGER,
                    metadata TEXT,
                    content_digest TEXT,
                    previous_segment_digest TEXT,
                    merkle_root TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS epochs (
                    epoch_id TEXT PRIMARY KEY,
                    start_segment TEXT,
                    end_segment TEXT,
                    merkle_root TEXT,
                    signed_checkpoint TEXT,
                    segments TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signed_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    epoch_id TEXT,
                    merkle_root TEXT,
                    timestamp REAL,
                    signer_id TEXT,
                    signature TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archive_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_records_timestamp ON archive_records(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_records_agent_code ON archive_records(agent_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_records_record_type ON archive_records(record_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_records_segment_id ON archive_records(segment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_created_at ON segments(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_epochs_created_at ON epochs(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signed_checkpoints_timestamp ON signed_checkpoints(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signed_checkpoints_epoch_id ON signed_checkpoints(epoch_id)")
            conn.commit()
    
    def _load_state(self):
        """Load archive state from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load last hash
            cursor.execute("SELECT value FROM archive_state WHERE key = 'last_hash'")
            result = cursor.fetchone()
            if result:
                self.last_hash = result[0]
            
            # Load current segment and epoch IDs
            cursor.execute("SELECT value FROM archive_state WHERE key = 'current_segment_id'")
            result = cursor.fetchone()
            if result:
                self.current_segment_id = result[0]
                
            cursor.execute("SELECT value FROM archive_state WHERE key = 'current_epoch_id'")
            result = cursor.fetchone()
            if result:
                self.current_epoch_id = result[0]
            
            # Load record count
            cursor.execute("SELECT COUNT(*) FROM archive_records")
            self.record_count = cursor.fetchone()[0]
            
            # Load segments
            cursor.execute("SELECT * FROM segments")
            for row in cursor.fetchall():
                segment_id, created_at, records_count, metadata_json, content_digest, previous_segment_digest, merkle_root = row
                segment = Segment(
                    id=segment_id,
                    created_at=created_at,
                    records_count=records_count,
                    metadata=json.loads(metadata_json),
                    content_digest=content_digest,
                    previous_segment_digest=previous_segment_digest,
                    records=[]  # Records loaded on demand
                )
                self.segments[segment_id] = segment
            
            # Load epochs
            cursor.execute("SELECT * FROM epochs")
            for row in cursor.fetchall():
                epoch_id, start_segment, end_segment, merkle_root, signed_checkpoint_json, segments_json, created_at = row
                epoch = Epoch(
                    id=epoch_id,
                    start_segment=start_segment,
                    end_segment=end_segment,
                    merkle_root=merkle_root,
                    signed_checkpoint=json.loads(signed_checkpoint_json) if signed_checkpoint_json else {},
                    segments=json.loads(segments_json)
                )
                self.epochs[epoch_id] = epoch
    
    def _save_state(self):
        """Save archive state to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO archive_state (key, value) VALUES (?, ?)",
                ('last_hash', self.last_hash)
            )
            conn.execute(
                "INSERT OR REPLACE INTO archive_state (key, value) VALUES (?, ?)",
                ('current_segment_id', self.current_segment_id)
            )
            conn.execute(
                "INSERT OR REPLACE INTO archive_state (key, value) VALUES (?, ?)",
                ('current_epoch_id', self.current_epoch_id)
            )
            conn.commit()
    
    def _generate_segment_id(self) -> str:
        """Generate unique segment identifier"""
        return f"SEG-{int(time.time() * 1000000)}-{secrets.token_hex(4).upper()}"
    
    def _generate_epoch_id(self) -> str:
        """Generate unique epoch identifier"""  
        return f"EPOCH-{int(time.time() * 1000000)}-{secrets.token_hex(4).upper()}"
    
    def _create_new_segment(self) -> str:
        """Create new segment for records"""
        segment_id = self._generate_segment_id()
        
        # Get previous segment digest
        previous_digest = "0" * 64  # Genesis
        if self.current_segment_id and self.current_segment_id in self.segments:
            previous_digest = self.segments[self.current_segment_id].content_digest
        
        segment = Segment(
            id=segment_id,
            created_at=datetime.utcnow().isoformat(),
            records_count=0,
            metadata={
                'provenance': self.agent_id,
                'tags': ['audit', 'compliance'],
                'version': '1.0'
            },
            content_digest="",  # Will be computed when sealed
            previous_segment_digest=previous_digest,
            records=[]
        )
        
        self.segments[segment_id] = segment
        self.current_segment_id = segment_id
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO segments 
                (segment_id, created_at, records_count, metadata, content_digest, previous_segment_digest, merkle_root)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                segment.id,
                segment.created_at, 
                segment.records_count,
                json.dumps(segment.metadata),
                segment.content_digest,
                segment.previous_segment_digest,
                ""  # Merkle root computed when sealed
            ))
            conn.commit()
        
        print(f"📦 Created new segment: {segment_id}")
        return segment_id
    
    def _seal_segment(self, segment_id: str) -> str:
        """Seal segment and compute final digest and Merkle root"""
        segment = self.segments[segment_id]
        
        # Load all records for this segment
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT record_id, timestamp, agent_code, record_type, priority, data, content_hash, signature
                FROM archive_records WHERE segment_id = ? ORDER BY timestamp
            """, (segment_id,))
            
            records = []
            for row in cursor.fetchall():
                record_id, timestamp, agent_code, record_type, priority, data_json, content_hash, signature = row
                records.append({
                    'record_id': record_id,
                    'timestamp': timestamp,
                    'agent_code': agent_code,
                    'record_type': record_type,
                    'priority': priority,
                    'data': json.loads(data_json),
                    'content_hash': content_hash,
                    'signature': signature
                })
        
        # Create Merkle tree for segment
        if records:
            merkle_tree = MerkleTree(records)
            merkle_root = merkle_tree.get_root_hash()
            self.merkle_trees[segment_id] = merkle_tree
        else:
            merkle_root = "0" * 64
        
        # Compute segment content digest
        segment_data = {
            'segment_id': segment.id,
            'records_count': len(records),
            'merkle_root': merkle_root,
            'previous_segment_digest': segment.previous_segment_digest
        }
        content_digest = hashlib.sha256(json.dumps(segment_data, sort_keys=True).encode()).hexdigest()
        
        # Update segment
        segment.content_digest = content_digest
        segment.records_count = len(records)
        segment.records = records
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE segments SET records_count = ?, content_digest = ?, merkle_root = ?
                WHERE segment_id = ?
            """, (len(records), content_digest, merkle_root, segment_id))
            conn.commit()
        
        print(f"🔒 Sealed segment {segment_id}: {len(records)} records, root {merkle_root[:16]}...")
        return content_digest
    
    def _generate_record_id(self) -> str:
        """Generate unique record identifier"""
        return f"AR-{int(time.time() * 1000000)}-{self.record_count:06d}"
    
    def _compute_content_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of record content"""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _sign_record(self, record_data: str) -> str:
        """Create HMAC signature for record authenticity"""
        signature = hmac.new(
            self.secret_key,
            record_data.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _verify_signature(self, record_data: str, signature: str) -> bool:
        """Verify record signature"""
        expected_sig = self._sign_record(record_data)
        return hmac.compare_digest(signature, expected_sig)
    
    def archive_record(self, 
                      agent_code: str,
                      record_type: RecordType,
                      data: Dict[str, Any],
                      priority: Priority = Priority.NORMAL) -> str:
        """
        Archive an immutable record with cryptographic integrity
        
        Returns: record_id of the archived record
        """
        with self.lock:
            # Economic: Deduct archive cost
            self._deduct_archive_cost()
            # Ensure we have a current segment
            if not self.current_segment_id:
                self._create_new_segment()
            # Check if current segment is full
            if not isinstance(self.current_segment_id, str) or self.current_segment_id not in self.segments:
                self._create_new_segment()
            # After creation, ensure current_segment_id is valid
            if not isinstance(self.current_segment_id, str) or self.current_segment_id not in self.segments:
                raise RuntimeError("Current segment ID is invalid after creation.")
            current_segment = self.segments[self.current_segment_id]
            if current_segment.records_count >= self.segment_size_limit:
                # Seal current segment and create new one
                self._seal_segment(self.current_segment_id)
                # Epoch completion logic can be added here if needed
                self._create_new_segment()
            # Generate record components
            record_id = self._generate_record_id()
            timestamp = time.time()
            content_hash = self._compute_content_hash(data)
            # Create record for signing
            record_content = f"{record_id}|{timestamp}|{agent_code}|{record_type.value}|{content_hash}|{self.last_hash}"
            signature = self._sign_record(record_content)
            # Create archival record
            record = ArchivalRecord(
                record_id=record_id,
                timestamp=timestamp,
                agent_code=agent_code,
                record_type=record_type,
                priority=priority,
                data=data,
                previous_hash=self.last_hash,
                content_hash=content_hash,
                signature=signature
            )
            try:
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO archive_records 
                        (record_id, segment_id, timestamp, agent_code, record_type, priority, 
                         data, previous_hash, content_hash, signature)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.record_id,
                        self.current_segment_id,
                        record.timestamp,
                        record.agent_code,
                        record.record_type.value,
                        record.priority.value,
                        json.dumps(record.data),
                        record.previous_hash,
                        record.content_hash,
                        record.signature
                    ))
                    conn.commit()
                # Economic: Reward for successful archival
                self._reward_archive_success()
            except Exception as e:
                # Economic: Penalty for failed archival
                self._burn_on_archive_failure()
                print(f"Archival failed: {e}")
                raise
            # Update segment record count
            current_segment.records_count += 1
            # Update state
            self.last_hash = content_hash
            self.record_count += 1
            self._save_state()
            # Log interaction
            self._log_interaction(agent_code, record_type, record_id)
            print(f"📚 Archived {record_type.value} from {agent_code}: {record_id}")
            return record_id
    
    def _log_interaction(self, agent_code: str, record_type: RecordType, record_id: str):
        """Track agent interactions for analysis"""
        if agent_code not in self.interaction_logs:
            self.interaction_logs[agent_code] = []
        
        self.interaction_logs[agent_code].append({
            'timestamp': time.time(),
            'record_type': record_type.value,
            'record_id': record_id
        })
        
        # Keep only last 1000 interactions per agent
        if len(self.interaction_logs[agent_code]) > 1000:
            self.interaction_logs[agent_code] = self.interaction_logs[agent_code][-1000:]
    
    # OVERSEER INTERACTIONS
    def archive_agent_status(self, agent_statuses: Dict[str, Dict]):
        """Archive periodic status updates from Overseer"""
        return self.archive_record(
            agent_code="BR-OVERSEER",
            record_type=RecordType.COORDINATION,
            data={
                'event': 'agent_status_update',
                'statuses': agent_statuses,
                'active_agents': len(agent_statuses)
            },
            priority=Priority.NORMAL
        )
    
    def archive_mission_plan(self, mission_id: str, plan: Dict[str, Any]):
        """Archive mission plans and assignments"""
        return self.archive_record(
            agent_code="BR-OVERSEER",
            record_type=RecordType.COORDINATION,
            data={
                'event': 'mission_planned',
                'mission_id': mission_id,
                'plan': plan
            },
            priority=Priority.HIGH
        )
    
    # ACCOUNTANT INTERACTIONS  
    def archive_transaction(self, transaction_data: Dict[str, Any]):
        """Archive BEL (Blackroot Economic Ledger) transactions"""
        return self.archive_record(
            agent_code="BR-ACCOUNTANT",
            record_type=RecordType.TRANSACTION,
            data={
                'event': 'bel_transaction',
                'transaction': transaction_data
            },
            priority=Priority.HIGH
        )
    
    def archive_economic_audit(self, audit_results: Dict[str, Any]):
        """Archive economic audit trails"""
        return self.archive_record(
            agent_code="BR-ACCOUNTANT", 
            record_type=RecordType.TRANSACTION,
            data={
                'event': 'economic_audit',
                'audit': audit_results
            },
            priority=Priority.CRITICAL
        )
    
    # ENFORCEMENT ENGINE INTERACTIONS
    def archive_judgment(self, judgment_data: Dict[str, Any]):
        """Archive immutable enforcement judgments"""
        return self.archive_record(
            agent_code="BR-ENFORCEMENT-ENGINE",
            record_type=RecordType.ENFORCEMENT,
            data={
                'event': 'judgment_rendered',
                'judgment': judgment_data
            },
            priority=Priority.CRITICAL
        )
    
    def archive_enforcement_action(self, action_data: Dict[str, Any]):
        """Archive enforcement actions and outcomes"""
        return self.archive_record(
            agent_code="BR-ENFORCER",
            record_type=RecordType.ENFORCEMENT,
            data={
                'event': 'enforcement_executed',
                'action': action_data
            },
            priority=Priority.HIGH
        )
    
    # REFLECTION INTERACTIONS
    def archive_reflection(self, reflection_data: Dict[str, Any]):
        """Archive introspection and anomaly reports"""
        return self.archive_record(
            agent_code="BR-REFLECTION",
            record_type=RecordType.REFLECTION,
            data={
                'event': 'reflection_completed',
                'reflection': reflection_data
            },
            priority=Priority.NORMAL
        )
    
    # REPAIR INTERACTIONS
    def archive_incident(self, incident_data: Dict[str, Any]):
        """Archive system incidents and repairs"""
        return self.archive_record(
            agent_code="BR-REPAIR",
            record_type=RecordType.REPAIR,
            data={
                'event': 'incident_logged',
                'incident': incident_data
            },
            priority=Priority.HIGH
        )
    
    def query_records(self, 
                     agent_code: Optional[str] = None,
                     record_type: Optional[RecordType] = None,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Query archived records with filters"""
        
        query = "SELECT * FROM archive_records WHERE 1=1"
        params = []
        
        if agent_code:
            query += " AND agent_code = ?"
            params.append(agent_code)
        
        if record_type:
            query += " AND record_type = ?"
            params.append(record_type.value)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            records = []
            for row in cursor.fetchall():
                record = dict(row)
                record['data'] = json.loads(record['data'])
                records.append(record)
            
            return records
    
    def get_historical_trends(self, agent_code: str, days: int = 30) -> Dict[str, Any]:
        """Provide historical trend analysis for coordination"""
        
        cutoff_time = time.time() - (days * 24 * 3600)
        records = self.query_records(
            agent_code=agent_code,
            start_time=cutoff_time,
            limit=10000
        )
        
        # Analyze trends
        record_types = {}
        daily_activity = {}
        
        for record in records:
            # Count by type
            rtype = record['record_type']
            record_types[rtype] = record_types.get(rtype, 0) + 1
            
            # Daily activity
            day = datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d')
            daily_activity[day] = daily_activity.get(day, 0) + 1
        
        return {
            'agent_code': agent_code,
            'analysis_period_days': days,
            'total_records': len(records),
            'record_types': record_types,
            'daily_activity': daily_activity,
            'avg_daily_activity': sum(daily_activity.values()) / max(len(daily_activity), 1)
        }
    
    def verify_archive_integrity(self) -> Dict[str, Any]:
        """Verify cryptographic integrity of the archive"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM archive_records ORDER BY timestamp")
            
            verified_count = 0
            error_count = 0
            errors = []
            
            previous_hash = "0" * 64
            
            for row in cursor.fetchall():
                record_id, timestamp, agent_code, record_type, priority, data_json, prev_hash, content_hash, signature = row
                
                # Verify hash chain
                if prev_hash != previous_hash:
                    errors.append(f"Hash chain broken at {record_id}")
                    error_count += 1
                    continue
                
                # Verify content hash
                data = json.loads(data_json)
                expected_content_hash = self._compute_content_hash(data)
                if content_hash != expected_content_hash:
                    errors.append(f"Content hash mismatch at {record_id}")
                    error_count += 1
                    continue
                
                # Verify signature
                record_content = f"{record_id}|{timestamp}|{agent_code}|{record_type}|{content_hash}|{prev_hash}"
                if not self._verify_signature(record_content, signature):
                    errors.append(f"Signature invalid at {record_id}")
                    error_count += 1
                    continue
                
                verified_count += 1
                previous_hash = content_hash
        
        return {
            'total_records': self.record_count,
            'verified_records': verified_count,
            'errors': error_count,
            'error_details': errors[:10],  # First 10 errors
            'integrity_percentage': (verified_count / max(self.record_count, 1)) * 100
        }
    
    # REPLICATION AND CONSENSUS
    
    def register_replica(self, replica_id: str, replica_digests: Dict[str, str]):
        """Register replica and its segment digests for consensus checking"""
        self.replica_digests[replica_id] = replica_digests
        print(f"🔗 Registered replica: {replica_id} with {len(replica_digests)} segments")
    
    def check_consensus(self) -> Dict[str, Any]:
        """Check consensus across replicas"""
        if not self.replica_digests:
            return {'status': 'no_replicas', 'consensus': True}
        
        consensus_results = {}
        conflicts = []
        
        # Get all unique segment IDs across replicas
        all_segments = set()
        for replica_digests in self.replica_digests.values():
            all_segments.update(replica_digests.keys())
        
        # Check each segment for consensus
        for segment_id in all_segments:
            digests = []
            for replica_id, replica_digests in self.replica_digests.items():
                if segment_id in replica_digests:
                    digests.append(replica_digests[segment_id])
            
            # Count digest occurrences
            digest_counts = defaultdict(int)
            for digest in digests:
                digest_counts[digest] += 1
            
            # Determine consensus
            total_replicas = len([r for r in self.replica_digests.values() if segment_id in r])
            required_consensus = max(1, int(total_replicas * self.consensus_threshold))
            
            consensus_digest = None
            consensus_count = 0
            
            for digest, count in digest_counts.items():
                if count >= required_consensus and count > consensus_count:
                    consensus_digest = digest
                    consensus_count = count
            
            if consensus_digest:
                consensus_results[segment_id] = {
                    'consensus_digest': consensus_digest,
                    'consensus_count': consensus_count,
                    'total_replicas': total_replicas,
                    'consensus_achieved': True
                }
            else:
                conflicts.append({
                    'segment_id': segment_id,
                    'digests': dict(digest_counts),
                    'total_replicas': total_replicas,
                    'required_consensus': required_consensus
                })
                consensus_results[segment_id] = {
                    'consensus_achieved': False,
                    'conflict': True
                }
        
        return {
            'status': 'consensus_checked',
            'total_segments': len(all_segments),
            'consensus_segments': len([r for r in consensus_results.values() if r.get('consensus_achieved')]),
            'conflict_segments': len(conflicts),
            'conflicts': conflicts,
            'results': consensus_results
        }
    
    def handle_tamper_detection(self, segment_id: str, expected_digest: str, actual_digest: str):
        """Handle detected tampering with high-severity alert"""
        tamper_alert = {
            'alert_type': 'TAMPER_DETECTED',
            'severity': 'CRITICAL',
            'segment_id': segment_id,
            'expected_digest': expected_digest,
            'actual_digest': actual_digest,
            'detection_time': time.time(),
            'replica_consensus': self.check_consensus()
        }
        
        # Archive the tamper detection as a critical record
        self.archive_record(
            agent_code=self.code,
            record_type=RecordType.SYSTEM_EVENT,
            data=tamper_alert,
            priority=Priority.CRITICAL
        )
        
        print(f"🚨 CRITICAL: Tampering detected in segment {segment_id}")
        print(f"Expected: {expected_digest[:16]}...")
        print(f"Actual:   {actual_digest[:16]}...")
        
        # In production, this would trigger alerts to Enforcement Engine and Governance
        return tamper_alert
    def get_agent_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of all agent interactions"""
        
        summary = {
            'total_agents': len(self.interaction_logs),
            'agent_activity': {},
            'most_active_agents': [],
            'record_type_distribution': {}
        }
        
        # Analyze each agent
        for agent_code, interactions in self.interaction_logs.items():
            recent_interactions = [
                i for i in interactions 
                if i['timestamp'] > time.time() - (24 * 3600)  # Last 24 hours
            ]
            
            summary['agent_activity'][agent_code] = {
                'total_interactions': len(interactions),
                'recent_interactions': len(recent_interactions),
                'last_interaction': max([i['timestamp'] for i in interactions]) if interactions else None
            }
        
        # Most active agents (last 24 hours)
        activity_pairs = [(code, data['recent_interactions']) 
                         for code, data in summary['agent_activity'].items()]
        summary['most_active_agents'] = sorted(activity_pairs, key=lambda x: x[1], reverse=True)[:5]
        
        # Record type distribution
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT record_type, COUNT(*) FROM archive_records GROUP BY record_type")
            summary['record_type_distribution'] = dict(cursor.fetchall())
        
        return summary
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive archivist status with anchoring info"""
        
        integrity_check = self.verify_archive_integrity()
        interaction_summary = self.get_agent_interaction_summary()
        consensus_check = self.check_consensus()
        
        return {
            'agent_id': self.agent_id,
            'code': self.code,
            'status': 'OPERATIONAL',
            'archive_stats': {
                'total_records': self.record_count,
                'total_segments': len(self.segments),
                'total_epochs': len(self.epochs),
                'signed_checkpoints': len(self.signed_checkpoints),
                'current_segment': self.current_segment_id,
                'current_epoch': self.current_epoch_id,
                'last_hash': self.last_hash[:16] + "...",
                'integrity_percentage': integrity_check['integrity_percentage'],
                'database_size': Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            },
            'anchoring_status': {
                'merkle_trees_active': len(self.merkle_trees),
                'governance_key_configured': self.governance_key is not None,
                'auto_checkpoint_enabled': self.auto_checkpoint,
                'segment_size_limit': self.segment_size_limit,
                'epoch_size_limit': self.epoch_size_limit
            },
            'consensus_status': {
                'registered_replicas': len(self.replica_digests),
                'consensus_threshold': self.consensus_threshold,
                'last_consensus_check': consensus_check['status']
            },
            'agent_interactions': interaction_summary,
            'capabilities': [
                "Immutable record storage",
                "Merkle tree verification", 
                "Cryptographic integrity chains",
                "Signed checkpoint creation",
                "Cross-replica consensus checking",
                "Tamper detection and alerting",
                "Compliance audit exports",
                "Historical trend analysis"
            ]
        }

"""
Integration points:
- Cross-agent communication via archive_record and query_records
- Monitoring, recovery, audit, and consensus via verify_archive_integrity and check_consensus
- Anchoring and Merkle verification for compliance
"""