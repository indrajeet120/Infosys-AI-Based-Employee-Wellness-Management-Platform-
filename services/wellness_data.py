"""
Wellness Content Loader and Repository Service.
Loads, validates, and manages wellness activity content for recommendation candidate generation.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from services.config import WELLNESS_CONTENT_PATH


@dataclass
class WellnessContent:
    """Represents a single wellness activity / content item."""
    content_id: str
    title: str
    description: str
    content_type: str
    target_emotions: List[str]
    activity_type: str
    difficulty: str
    duration: str
    duration_minutes: int
    language: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_embedding_text(self) -> str:
        """Constructs rich semantic text representation for embedding matching."""
        target_emo_str = ", ".join(self.target_emotions)
        tag_str = ", ".join(self.tags)
        return (
            f"Title: {self.title}. "
            f"Description: {self.description}. "
            f"Activity: {self.activity_type} ({self.content_type}). "
            f"Target Emotions: {target_emo_str}. "
            f"Tags: {tag_str}."
        )


class WellnessContentRepository:
    """Manages loading and querying wellness content."""
    def __init__(self, data_path: Path = WELLNESS_CONTENT_PATH):
        self.data_path = data_path
        self._items: Dict[str, WellnessContent] = {}
        self.load()

    def load(self) -> None:
        """Loads wellness content from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Wellness content file not found at: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        self._items = {}
        for _, row in df.iterrows():
            target_emotions = [
                e.strip().lower()
                for e in str(row["target_emotions"]).split(",")
                if e.strip()
            ]
            tags = [
                t.strip().lower()
                for t in str(row["tags"]).split(",")
                if t.strip()
            ]
            
            content = WellnessContent(
                content_id=str(row["content_id"]).strip(),
                title=str(row["title"]).strip(),
                description=str(row["description"]).strip(),
                content_type=str(row["content_type"]).strip().lower(),
                target_emotions=target_emotions,
                activity_type=str(row["activity_type"]).strip().lower(),
                difficulty=str(row.get("difficulty", "Beginner")).strip(),
                duration=str(row.get("duration", "5 mins")).strip(),
                duration_minutes=int(row.get("duration_minutes", 5)),
                language=str(row.get("language", "English")).strip(),
                tags=tags,
            )
            self._items[content.content_id] = content

    def get_all(self) -> List[WellnessContent]:
        """Returns all wellness content items."""
        return list(self._items.values())

    def get_by_id(self, content_id: str) -> Optional[WellnessContent]:
        """Retrieves a specific content item by ID."""
        return self._items.get(content_id)

    def count(self) -> int:
        return len(self._items)


# Module-level cached instance
_REPO_INSTANCE: Optional[WellnessContentRepository] = None


def get_wellness_repository() -> WellnessContentRepository:
    """Returns a singleton wellness repository instance."""
    global _REPO_INSTANCE
    if _REPO_INSTANCE is None:
        _REPO_INSTANCE = WellnessContentRepository()
    return _REPO_INSTANCE
