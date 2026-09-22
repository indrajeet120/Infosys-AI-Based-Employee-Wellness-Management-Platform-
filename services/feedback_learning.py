"""
Recommendation Feedback Learning and Dynamic Ranking Adaptation Service (Task 7).

Captures, persists, and analyzes:
1. Recommendation viewed
2. Recommendation accepted
3. Recommendation rejected
4. User ratings (1 to 5 stars)
5. User preference changes
6. Full recommendation interaction history

Uses real feedback signals to dynamically re-weight and re-rank future recommendations:
- Positive interactions (accepted, liked, rated >= 4.0) increase relevance of similar content.
- Negative interactions (rejected, disliked, rated <= 2.0) reduce relevance of similar content.
- Avoids repeatedly recommending rejected content.
- Adapts ranking signals based on configurable weights without immediate transformer retraining.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import uuid
import numpy as np

from services.config import FEEDBACK_EVENTS_PATH
from services.user_profile import get_user_profile_manager, UserProfile


@dataclass
class FeedbackRecord:
    """Structured feedback record capturing comprehensive user interaction context."""
    id: str = field(default_factory=lambda: f"fb_{uuid.uuid4().hex[:10]}")
    user_id: str = "default_user"
    recommendation_id: str = ""
    recommendation_type: str = ""
    viewed: bool = True
    accepted: bool = False
    rejected: bool = False
    rating: Optional[float] = None  # 1.0 to 5.0
    feedback_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    preference_snapshot: Dict[str, Any] = field(default_factory=dict)
    emotion_at_recommendation: str = "Joy"
    emotion_intensity: float = 0.50
    interaction_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackRecord":
        return cls(
            id=data.get("id", f"fb_{uuid.uuid4().hex[:10]}"),
            user_id=data.get("user_id", "default_user"),
            recommendation_id=data.get("recommendation_id", ""),
            recommendation_type=data.get("recommendation_type", ""),
            viewed=data.get("viewed", True),
            accepted=data.get("accepted", False),
            rejected=data.get("rejected", False),
            rating=float(data["rating"]) if data.get("rating") is not None else None,
            feedback_timestamp=data.get("feedback_timestamp", datetime.now(timezone.utc).isoformat()),
            preference_snapshot=data.get("preference_snapshot", {}),
            emotion_at_recommendation=data.get("emotion_at_recommendation", "Joy"),
            emotion_intensity=float(data.get("emotion_intensity", 0.50)),
            interaction_metadata=data.get("interaction_metadata", {}),
        )


class FeedbackManager:
    """
    Manages persistent storage, analysis, and ranking signals derived from user feedback.
    """
    def __init__(self, storage_path: Path = FEEDBACK_EVENTS_PATH):
        self.storage_path = storage_path
        self._events: List[FeedbackRecord] = []
        self.load()

    def load(self) -> None:
        """Loads feedback events from persistent disk storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._events = [FeedbackRecord.from_dict(item) for item in data]
            except Exception:
                self._events = []
        else:
            self._events = []

    def save(self) -> None:
        """Persists all feedback events to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [e.to_dict() for e in self._events]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def _get_current_preference_snapshot(self, user_id: str) -> Dict[str, Any]:
        """Captures a snapshot of user's current preferences."""
        user_mgr = get_user_profile_manager()
        profile = user_mgr.get_or_create_profile(user_id)
        return {
            "preferred_content_types": list(profile.preferred_content_types),
            "preferred_activities": list(profile.preferred_activities),
            "preferred_language": profile.preferred_language,
            "interaction_count": profile.interaction_count,
        }

    def record_view(
        self,
        user_id: str,
        recommendation_id: str,
        recommendation_type: str = "",
        emotion: str = "Joy",
        intensity: float = 0.50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """1. Records a viewed recommendation event."""
        pref_snap = self._get_current_preference_snapshot(user_id)
        rec = FeedbackRecord(
            user_id=user_id,
            recommendation_id=recommendation_id,
            recommendation_type=recommendation_type,
            viewed=True,
            accepted=False,
            rejected=False,
            rating=None,
            preference_snapshot=pref_snap,
            emotion_at_recommendation=emotion,
            emotion_intensity=round(float(intensity), 4),
            interaction_metadata=metadata or {},
        )
        self._events.append(rec)
        self.save()
        return rec

    def record_acceptance(
        self,
        user_id: str,
        recommendation_id: str,
        recommendation_type: str = "",
        emotion: str = "Joy",
        intensity: float = 0.50,
        rating: Optional[float] = 5.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """
        2. Records an accepted recommendation event.
        Updates user profile and triggers dynamic preference synergy.
        """
        user_mgr = get_user_profile_manager()
        user_mgr.record_feedback(
            user_id=user_id,
            content_id=recommendation_id,
            feedback_type="like",
            recommendation_score=1.0,
            dominant_emotion=emotion,
            intensity=intensity,
            activity_type=recommendation_type,
            tags=metadata.get("tags", []) if metadata else [],
        )

        pref_snap = self._get_current_preference_snapshot(user_id)
        rec = FeedbackRecord(
            user_id=user_id,
            recommendation_id=recommendation_id,
            recommendation_type=recommendation_type,
            viewed=True,
            accepted=True,
            rejected=False,
            rating=rating,
            preference_snapshot=pref_snap,
            emotion_at_recommendation=emotion,
            emotion_intensity=round(float(intensity), 4),
            interaction_metadata=metadata or {},
        )
        self._events.append(rec)
        self.save()
        return rec

    def record_rejection(
        self,
        user_id: str,
        recommendation_id: str,
        recommendation_type: str = "",
        emotion: str = "Joy",
        intensity: float = 0.50,
        reason: str = "not_relevant",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """
        3. Records a rejected recommendation event.
        Demotes item and similar items from being repeatedly recommended.
        """
        user_mgr = get_user_profile_manager()
        user_mgr.record_feedback(
            user_id=user_id,
            content_id=recommendation_id,
            feedback_type="dislike",
            recommendation_score=0.0,
            dominant_emotion=emotion,
            intensity=intensity,
            activity_type=recommendation_type,
            tags=metadata.get("tags", []) if metadata else [],
        )

        meta = dict(metadata or {})
        meta["rejection_reason"] = reason

        pref_snap = self._get_current_preference_snapshot(user_id)
        rec = FeedbackRecord(
            user_id=user_id,
            recommendation_id=recommendation_id,
            recommendation_type=recommendation_type,
            viewed=True,
            accepted=False,
            rejected=True,
            rating=1.0,
            preference_snapshot=pref_snap,
            emotion_at_recommendation=emotion,
            emotion_intensity=round(float(intensity), 4),
            interaction_metadata=meta,
        )
        self._events.append(rec)
        self.save()
        return rec

    def record_rating(
        self,
        user_id: str,
        recommendation_id: str,
        rating: float,
        recommendation_type: str = "",
        emotion: str = "Joy",
        intensity: float = 0.50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """
        4. Records a direct user rating (1.0 to 5.0 stars).
        """
        clamped_rating = float(np.clip(rating, 1.0, 5.0))
        accepted = clamped_rating >= 4.0
        rejected = clamped_rating <= 2.0

        user_mgr = get_user_profile_manager()
        feedback_type = "like" if accepted else ("dislike" if rejected else "select")
        user_mgr.record_feedback(
            user_id=user_id,
            content_id=recommendation_id,
            feedback_type=feedback_type,
            recommendation_score=round(clamped_rating / 5.0, 4),
            dominant_emotion=emotion,
            intensity=intensity,
            activity_type=recommendation_type,
            tags=metadata.get("tags", []) if metadata else [],
        )

        pref_snap = self._get_current_preference_snapshot(user_id)
        rec = FeedbackRecord(
            user_id=user_id,
            recommendation_id=recommendation_id,
            recommendation_type=recommendation_type,
            viewed=True,
            accepted=accepted,
            rejected=rejected,
            rating=clamped_rating,
            preference_snapshot=pref_snap,
            emotion_at_recommendation=emotion,
            emotion_intensity=round(float(intensity), 4),
            interaction_metadata=metadata or {},
        )
        self._events.append(rec)
        self.save()
        return rec

    def record_preference_change(
        self,
        user_id: str,
        preferred_content_types: Optional[List[str]] = None,
        preferred_activities: Optional[List[str]] = None,
        preferred_language: Optional[str] = None,
    ) -> FeedbackRecord:
        """
        5. Records explicit user preference changes.
        """
        user_mgr = get_user_profile_manager()
        user_mgr.update_preferences(
            user_id=user_id,
            preferred_content_types=preferred_content_types,
            preferred_activities=preferred_activities,
            preferred_language=preferred_language,
        )

        pref_snap = self._get_current_preference_snapshot(user_id)
        rec = FeedbackRecord(
            user_id=user_id,
            recommendation_id="preference_update_event",
            recommendation_type="user_preference",
            viewed=False,
            accepted=True,
            rejected=False,
            rating=None,
            preference_snapshot=pref_snap,
            emotion_at_recommendation="N/A",
            emotion_intensity=0.0,
            interaction_metadata={
                "action": "preferences_updated",
                "new_types": preferred_content_types,
                "new_activities": preferred_activities,
                "new_language": preferred_language,
            },
        )
        self._events.append(rec)
        self.save()
        return rec

    def get_user_feedback(self, user_id: str, limit: Optional[int] = None) -> List[FeedbackRecord]:
        """Returns all feedback records for a specific user."""
        user_events = [e for e in self._events if e.user_id == user_id]
        if limit is None:
            return user_events
        return user_events[-limit:]

    def get_rejected_content_ids(self, user_id: str) -> Set[str]:
        """Returns the set of all rejected content IDs for a user."""
        return set(e.recommendation_id for e in self._events if e.user_id == user_id and e.rejected)

    def get_accepted_content_ids(self, user_id: str) -> Set[str]:
        """Returns the set of all accepted content IDs for a user."""
        return set(e.recommendation_id for e in self._events if e.user_id == user_id and e.accepted)

    def calculate_historical_acceptance_score(
        self,
        content_id: str,
        activity_type: str,
        tags: List[str],
        user_id: str,
        current_emotion: str = "Joy",
    ) -> float:
        """
        Calculates acceptance boost based on:
        1. Direct past acceptances of this item.
        2. Synergy with accepted activity types and tags under matching emotional context.
        """
        user_events = [e for e in self._events if e.user_id == user_id]
        if not user_events:
            return 0.50

        act_norm = activity_type.strip().lower()
        tag_set = set(t.strip().lower() for t in tags)
        curr_emo_norm = current_emotion.strip().lower()

        direct_accepts = sum(1 for e in user_events if e.recommendation_id == content_id and e.accepted)
        activity_accepts = sum(
            1 for e in user_events
            if e.accepted and e.recommendation_type.strip().lower() == act_norm
        )
        emotion_matched_accepts = sum(
            1 for e in user_events
            if e.accepted and e.emotion_at_recommendation.strip().lower() == curr_emo_norm
            and (e.recommendation_id == content_id or e.recommendation_type.strip().lower() == act_norm)
        )

        raw_score = 0.50 + 0.20 * min(direct_accepts, 2) + 0.15 * min(activity_accepts, 3) / 3.0 + 0.15 * min(emotion_matched_accepts, 2) / 2.0
        return float(np.clip(round(raw_score, 4), 0.0, 1.0))

    def calculate_historical_rejection_penalty(
        self,
        content_id: str,
        activity_type: str,
        tags: List[str],
        user_id: str,
        current_emotion: str = "Joy",
    ) -> float:
        """
        Calculates rejection penalty based on:
        1. Direct rejections of this item (strong penalty to prevent repeating rejected content).
        2. Rejections of the same activity type.
        """
        user_events = [e for e in self._events if e.user_id == user_id]
        if not user_events:
            return 0.0

        act_norm = activity_type.strip().lower()
        direct_rejects = sum(1 for e in user_events if e.recommendation_id == content_id and e.rejected)
        activity_rejects = sum(
            1 for e in user_events
            if e.rejected and e.recommendation_type.strip().lower() == act_norm
        )

        # High direct penalty for rejected items
        if direct_rejects >= 1:
            penalty = 0.60 + 0.20 * min(direct_rejects - 1, 2)
        elif activity_rejects >= 2:
            penalty = 0.25 * min(activity_rejects, 3) / 3.0
        else:
            penalty = 0.0

        return float(np.clip(round(penalty, 4), 0.0, 1.0))

    def calculate_rating_score(
        self,
        content_id: str,
        activity_type: str,
        user_id: str,
    ) -> float:
        """
        Calculates normalized rating score (0.0 to 1.0) from explicit 1-5 star ratings.
        Unrated content defaults to 0.50 neutral baseline.
        """
        user_events = [e for e in self._events if e.user_id == user_id and e.rating is not None]
        if not user_events:
            return 0.50

        item_ratings = [e.rating for e in user_events if e.recommendation_id == content_id]
        if item_ratings:
            avg_rating = float(np.mean(item_ratings))
            return float(np.clip(round((avg_rating - 1.0) / 4.0, 4), 0.0, 1.0))

        act_norm = activity_type.strip().lower()
        act_ratings = [e.rating for e in user_events if e.recommendation_type.strip().lower() == act_norm]
        if act_ratings:
            avg_act_rating = float(np.mean(act_ratings))
            # Blended toward 0.50 baseline
            norm_act = (avg_act_rating - 1.0) / 4.0
            return float(np.clip(round(0.50 * 0.50 + 0.50 * norm_act, 4), 0.0, 1.0))

        return 0.50

    def calculate_item_rating_score(
        self,
        content_id: str,
        activity_type: str,
        user_id: str,
    ) -> float:
        """Alias for calculate_rating_score."""
        return self.calculate_rating_score(content_id=content_id, activity_type=activity_type, user_id=user_id)

    def calculate_diversity_score(
        self,
        candidate_activity: str,
        selected_activities: List[str],
    ) -> float:
        """
        Calculates recommendation diversity score: penalizes duplicate activity types among top items.
        """
        count = selected_activities.count(candidate_activity.strip().lower())
        if count == 0:
            return 1.0
        elif count == 1:
            return 0.60
        else:
            return float(max(0.20, 1.0 - 0.35 * count))


# Singleton instance
_FEEDBACK_MANAGER_INSTANCE: Optional[FeedbackManager] = None


def get_feedback_manager() -> FeedbackManager:
    """Returns singleton instance of FeedbackManager."""
    global _FEEDBACK_MANAGER_INSTANCE
    if _FEEDBACK_MANAGER_INSTANCE is None:
        _FEEDBACK_MANAGER_INSTANCE = FeedbackManager()
    return _FEEDBACK_MANAGER_INSTANCE
