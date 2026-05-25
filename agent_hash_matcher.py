import hashlib


# List of known agent names/IDs (auto-extracted and extended; add more as needed)
agent_names = [
    # Explicit IDs from constructors and assignments
    "SENS-DAEMON",
    "FAL-SCOURGE-001",
    "12345",
    "PROP-SEED",
    "INF-VENOM",
    "ALPHA",
    "DEF-GLOOM",
    # Agent class names (if any are used as IDs)
    "SensorAgent",
    "ScoutAgent",
    "FallmanAgent",
    "ReplicatorAgent",
    "RepairAgent",
    "ReflectionAgent",
    "PropagationAgent",
    "PeerLinkerAgent",
    "EnhancedPeerLinkerAgent",
    "MissionAgent",
    "LearningAgent",
    "InfiltratorAgent",
    "ExfiltratorAgent",
    "EnforcerAgent",
    "DefilerAgent",
    "ArchivistAgent",
    # Add more agent names/IDs from your system here
]

dashboard_hashes = [
    "9301e28acb51cbc4b4e1f52ebda0c7ba2a1a8601",
    "0f35a6b1d85df4411c832b4b971d9053c63bf60a",
    "88327ca765203a7c145c5fe6cf92b6e422a2ad0c",
    "f33717919f2ae10dc1f81db2cb1d5c3ef74e9507",
    "11fe833a2455268ca60f58c8cecd1fde3cac9932"
]

print("Agent Name/ID to SHA-1 Hash Mapping:")
for name in agent_names:
    sha1 = hashlib.sha1(name.encode()).hexdigest()
    match = "MATCH" if sha1 in dashboard_hashes else ""
    print(f"{name} → {sha1} {match}")

print("\nUnmatched dashboard hashes:")
for h in dashboard_hashes:
    if not any(hashlib.sha1(name.encode()).hexdigest() == h for name in agent_names):
        print(h)
