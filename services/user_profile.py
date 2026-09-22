"""
User Profile and Interaction History Management Service.
Handles user preferences, emotional history tracking, feedback recording,
and persistence in local storage.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from services.config import USER_PROFILES_PATH, INTERACTION_HISTORY_PATH


@dataclass
class UserProfile:
    """Represents a user's profile, preferences, and interaction history."""
    user_id: str = "default_user"
    preferred_content_types: List[str] = field(default_factory=lambda: ["exercise", "audio", "text", "interactive_guide"])
    preferred_activities: List[str] = field(default_factory=lambda: ["breathing exercise", "meditation", "journaling", "relaxation activity", "motivational content"])
    preferred_language: str = "English"
    liked_content: List[str] = field(default_factory=list)
    disliked_content: List[str] = field(default_factory=list)
    previously_selected_content: List[str] = field(default_factory=list)
    interaction_count: int = 0
    recent_emotions: List[str] = field(default_factory=list)
    emotion_history: List[Dict[str, Any]] = field(default_factory=list)
    recommendation_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data.get("user_id", "default_user"),
            preferred_content_types=data.get("preferred_content_types", ["exercise", "audio", "text"]),
            preferred_activities=data.get("preferred_activities", ["breathing exercise", "meditation"]),
            preferred_language=data.get("preferred_language", "English"),
            liked_content=data.get("liked_content", []),
            disliked_content=data.get("disliked_content", []),
            previously_selected_content=data.get("previously_selected_content", []),
            interaction_count=int(data.get("interaction_count", 0)),
            recent_emotions=data.get("recent_emotions", []),
            emotion_history=data.get("emotion_history", []),
            recommendation_history=data.get("recommendation_history", []),
        )


class UserProfileManager:
    """Manages loading, updating, and saving user profiles."""
    def __init__(self, profiles_path: Path = USER_PROFILES_PATH, interactions_path: Path = INTERACTION_HISTORY_PATH):
        self.profiles_path = profiles_path
        self.interactions_path = interactions_path
        self._profiles: Dict[str, UserProfile] = {}
        self.load()

    def load(self) -> None:
        """Loads user profiles from disk."""
        if self.profiles_path.exists():
            try:
                with open(self.profiles_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._profiles = {
                        uid: UserProfile.from_dict(p_data)
                        for uid, p_data in data.items()
                    }
            except Exception:
                self._profiles = {}
        else:
            self._profiles = {}

    def save(self) -> None:
        """Saves user profiles to disk."""
        self.profiles_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {uid: p.to_dict() for uid, p in self._profiles.items()}
        with open(self.profiles_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def get_or_create_profile(self, user_id: str = "default_user") -> UserProfile:
        """Retrieves existing profile or initializes a default profile for the user."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
            self.save()
        return self._profiles[user_id]

    def update_preferences(
        self,
        user_id: str,
        preferred_content_types: Optional[List[str]] = None,
        preferred_activities: Optional[List[str]] = None,
        preferred_language: Optional[str] = None,
    ) -> UserProfile:
        """Updates user preferences."""
        profile = self.get_or_create_profile(user_id)
        if preferred_content_types is not None:
            profile.preferred_content_types = [c.lower() for c in preferred_content_types]
        if preferred_activities is not None:
            profile.preferred_activities = [a.lower() for a in preferred_activities]
        if preferred_language is not None:
            profile.preferred_language = preferred_language
        self.save()
        return profile

    def record_emotion_entry(
        self,
        user_id: str,
        dominant_emotion: str,
        intensity: float,
        positive_polarity: float,
        negative_polarity: float,
        final_emotional_state: str,
        raw_text: str = "",
        probabilities: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None,
        model_used: str = "bert",
    ) -> None:
        """Records a new emotional state entry in user's history."""
        profile = self.get_or_create_profile(user_id)
        
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": timestamp,
            "dominant_emotion": dominant_emotion,
            "intensity": round(float(intensity), 4),
            "positive_polarity": round(float(positive_polarity), 4),
            "negative_polarity": round(float(negative_polarity), 4),
            "final_emotional_state": final_emotional_state,
            "text_snippet": raw_text[:80] if raw_text else "",
            "probabilities": probabilities or {},
            "confidence": round(float(confidence), 4) if confidence is not None else round(float(intensity), 4),
            "model_used": model_used,
        }
        profile.emotion_history.append(entry)
        
        # Update recent emotions list (keep last 5)
        profile.recent_emotions.append(dominant_emotion)
        if len(profile.recent_emotions) > 5:
            profile.recent_emotions = profile.recent_emotions[-5:]
            
        self.save()

    def get_emotion_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns the user's emotion history, optionally limited to the most recent N items."""
        profile = self.get_or_create_profile(user_id)
        if limit is None:
            return profile.emotion_history
        return profile.emotion_history[-limit:]

    def record_feedback(
        self,
        user_id: str,
        content_id: str,
        feedback_type: str,  # 'like', 'dislike', 'select', 'skip', 'viewed'
        recommendation_score: float = 0.0,
        dominant_emotion: Optional[str] = None,
        intensity: Optional[float] = None,
        activity_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Records user interaction and feedback on a recommendation.
        Feedback directly influences subsequent recommendations.
        """
        profile = self.get_or_create_profile(user_id)
        profile.interaction_count += 1

        was_liked = (feedback_type == "like")
        was_disliked = (feedback_type == "dislike")
        was_selected = (feedback_type in ["select", "like"])

        if was_liked and content_id not in profile.liked_content:
            profile.liked_content.append(content_id)
            if content_id in profile.disliked_content:
                profile.disliked_content.remove(content_id)

        elif was_disliked and content_id not in profile.disliked_content:
            profile.disliked_content.append(content_id)
            if content_id in profile.liked_content:
                profile.liked_content.remove(content_id)

        if was_selected and content_id not in profile.previously_selected_content:
            profile.previously_selected_content.append(content_id)

        # Infer emotional context if not provided
        if dominant_emotion is None and profile.emotion_history:
            dominant_emotion = profile.emotion_history[-1].get("dominant_emotion")
        if intensity is None and profile.emotion_history:
            intensity = profile.emotion_history[-1].get("intensity")

        timestamp = datetime.now(timezone.utc).isoformat()
        history_entry = {
            "user_id": user_id,
            "content_id": content_id,
            "timestamp": timestamp,
            "recommendation_score": round(recommendation_score, 4),
            "was_selected": was_selected,
            "was_liked": was_liked,
            "was_disliked": was_disliked,
            "feedback": feedback_type,
            "dominant_emotion": dominant_emotion or "Joy",
            "intensity": round(float(intensity), 4) if intensity is not None else 0.50,
            "activity_type": activity_type or "",
            "tags": tags or [],
        }
        profile.recommendation_history.append(history_entry)
        self.save()


_PROFILE_MANAGER_INSTANCE: Optional[UserProfileManager] = None


def get_user_profile_manager() -> UserProfileManager:
    """Returns singleton instance of UserProfileManager."""
    global _PROFILE_MANAGER_INSTANCE
    if _PROFILE_MANAGER_INSTANCE is None:
        _PROFILE_MANAGER_INSTANCE = UserProfileManager()
    return _PROFILE_MANAGER_INSTANCE


def check_collaborative_filtering_availability(min_users: int = 5, min_interactions: int = 20) -> Dict[str, Any]:
    """
    Checks if there is sufficient multi-user interaction data to train/run Collaborative Filtering.
    When data is insufficient, returns clear status and activates documented hybrid fallback.
    """
    manager = get_user_profile_manager()
    total_users = len(manager._profiles)
    total_interactions = sum(p.interaction_count for p in manager._profiles.values())

    if total_users >= min_users and total_interactions >= min_interactions:
        return {
            "available": True,
            "total_users": total_users,
            "total_interactions": total_interactions,
            "strategy": "Matrix Factorization / Item-Item Collaborative Filtering",
            "reason": f"Sufficient interaction data available ({total_users} users, {total_interactions} interactions).",
            "fallback_strategy": "N/A - Collaborative Filtering Active",
        }
    else:
        return {
            "available": False,
            "total_users": total_users,
            "total_interactions": total_interactions,
            "reason": (
                f"Insufficient interaction matrix (Requires >= {min_users} users and >= {min_interactions} ratings; "
                f"Current: {total_users} users, {total_interactions} interactions)."
            ),
            "fallback_strategy": "Content-Based, Preference-Based, and Emotion-Similarity Hybrid Matching",
        }
