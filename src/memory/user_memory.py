"""User profile memory — JSON file persistence for cross-session memory."""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict

from src.config import USER_DATA_PATH


@dataclass
class UserProfile:
    user_id: str = "default"
    age: int = 0
    gender: str = ""
    occupation: str = ""
    budget: int = 0
    family_history: str = ""
    chronic_conditions: str = ""
    preferences: dict = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if critical fields are populated for accurate recommendations."""
        return self.age > 0 and bool(self.gender) and self.budget > 0

    def missing_fields(self) -> list[str]:
        """Return list of critical missing fields (same criteria as is_complete)."""
        missing = []
        if self.age <= 0:
            missing.append("年龄")
        if not self.gender:
            missing.append("性别")
        if self.budget <= 0:
            missing.append("预算")
        if not self.occupation:
            missing.append("职业")
        return missing


class UserMemory:
    """Manages user profile persistence as JSON files."""

    def __init__(self, storage_dir: str = USER_DATA_PATH):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, user_id: str) -> Path:
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_id}.json"

    def load(self, user_id: str = "default") -> UserProfile:
        path = self._file_path(user_id)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserProfile(**data)
        return UserProfile(user_id=user_id)

    def save(self, profile: UserProfile) -> None:
        path = self._file_path(profile.user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

    def delete(self, user_id: str) -> bool:
        path = self._file_path(user_id)
        if path.exists():
            os.remove(path)
            return True
        return False
