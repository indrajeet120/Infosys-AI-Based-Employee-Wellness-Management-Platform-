"""
Task 4: Advanced Search and Filtering Service.
Filters actual stored emotional records, recommendation history, feedback events, and wellness content.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from services.wellness_data import WellnessContent, get_wellness_repository


@dataclass
class SearchFilterCriteria:
    """Dataclass defining multi-criteria search and filter parameters."""
    query: Optional[str] = None
    emotions: Optional[List[str]] = field(default_factory=list)
    min_intensity: Optional[float] = None
    max_intensity: Optional[float] = None
    activity_types: Optional[List[str]] = field(default_factory=list)
    difficulty: Optional[str] = None
    start_date: Optional[Any] = None
    end_date: Optional[Any] = None


class FilterEngine:
    """Provides dynamic filtering and search capabilities across stored records."""

    @staticmethod
    def _parse_date(dt_val: Any) -> Optional[datetime]:
        """Safely parses ISO date string or datetime object."""
        if not dt_val:
            return None
        if isinstance(dt_val, datetime):
            return dt_val
        try:
            return datetime.fromisoformat(str(dt_val).replace("Z", "+00:00"))
        except Exception:
            return None

    def filter_emotional_records(
        self,
        records: List[Dict[str, Any]],
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        emotion: Optional[str] = None,
        min_intensity: Optional[float] = None,
        max_intensity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filters user emotional history check-ins by date range, emotion, and intensity range."""
        filtered = []
        dt_start = self._parse_date(start_date)
        dt_end = self._parse_date(end_date)
        emo_clean = emotion.strip().lower() if emotion and emotion.strip() != "All" else None

        for rec in records:
            # Date filter
            rec_dt = self._parse_date(rec.get("timestamp") or rec.get("date"))
            if dt_start and rec_dt and rec_dt < dt_start:
                continue
            if dt_end and rec_dt and rec_dt > dt_end:
                continue

            # Emotion filter
            dom = str(rec.get("dominant_emotion", "")).strip().lower()
            if emo_clean and dom != emo_clean:
                continue

            # Intensity range filter
            intensity = float(rec.get("intensity", 0.5))
            if min_intensity is not None and intensity < min_intensity:
                continue
            if max_intensity is not None and intensity > max_intensity:
                continue

            filtered.append(rec)
        return filtered

    def filter_recommendation_history(
        self,
        history: List[Dict[str, Any]],
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        activity_type: Optional[str] = None,
        emotion: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filters past recommendation interactions by date range, activity, emotion, and score."""
        filtered = []
        dt_start = self._parse_date(start_date)
        dt_end = self._parse_date(end_date)
        act_clean = activity_type.strip().lower() if activity_type and activity_type.strip() != "All" else None
        emo_clean = emotion.strip().lower() if emotion and emotion.strip() != "All" else None

        for rec in history:
            rec_dt = self._parse_date(rec.get("timestamp"))
            if dt_start and rec_dt and rec_dt < dt_start:
                continue
            if dt_end and rec_dt and rec_dt > dt_end:
                continue

            act = str(rec.get("activity_type", "")).strip().lower()
            if act_clean and act != act_clean:
                continue

            dom = str(rec.get("dominant_emotion", "")).strip().lower()
            if emo_clean and dom != emo_clean:
                continue

            score = float(rec.get("recommendation_score", 0.0))
            if min_score is not None and score < min_score:
                continue

            filtered.append(rec)
        return filtered

    def filter_feedback_events(
        self,
        events: List[Dict[str, Any]],
        status: Optional[str] = None,  # "accepted", "rejected", "rated"
        min_rating: Optional[float] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Filters user feedback events by status (accepted/rejected/rated), rating, and date."""
        filtered = []
        dt_start = self._parse_date(start_date)
        dt_end = self._parse_date(end_date)
        status_clean = status.strip().lower() if status and status.strip() != "All" else None

        for ev in events:
            ev_dt = self._parse_date(ev.get("feedback_timestamp") or ev.get("timestamp"))
            if dt_start and ev_dt and ev_dt < dt_start:
                continue
            if dt_end and ev_dt and ev_dt > dt_end:
                continue

            if status_clean == "accepted" and not ev.get("accepted", False):
                continue
            if status_clean == "rejected" and not ev.get("rejected", False):
                continue
            if status_clean == "rated" and ev.get("rating") is None:
                continue

            rating = ev.get("rating")
            if min_rating is not None:
                if rating is None or float(rating) < min_rating:
                    continue

            filtered.append(ev)
        return filtered

    def filter_wellness_content(
        self,
        query_text: Optional[str] = None,
        content_type: Optional[str] = None,
        activity_type: Optional[str] = None,
        target_emotion: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[WellnessContent]:
        """Filters wellness repository items by content type, activity type, target emotion, and query."""
        repo = get_wellness_repository()
        items = repo.get_all()
        filtered = []

        q_clean = query_text.strip().lower() if query_text and query_text.strip() else None
        c_clean = content_type.strip().lower() if content_type and content_type.strip() != "All" else None
        a_clean = activity_type.strip().lower() if activity_type and activity_type.strip() != "All" else None
        e_clean = target_emotion.strip().lower() if target_emotion and target_emotion.strip() != "All" else None
        d_clean = difficulty.strip().lower() if difficulty and difficulty.strip() != "All" else None

        for item in items:
            if c_clean and item.content_type.lower() != c_clean:
                continue
            if a_clean and item.activity_type.lower() != a_clean:
                continue
            if e_clean and e_clean not in [t.lower() for t in item.target_emotions]:
                continue
            if d_clean and item.difficulty.lower() != d_clean:
                continue
            if q_clean:
                haystack = f"{item.title} {item.description} {' '.join(item.tags)}".lower()
                if q_clean not in haystack:
                    continue

            filtered.append(item)
        return filtered

    @staticmethod
    def filter_records(
        records: List[Dict[str, Any]],
        criteria: SearchFilterCriteria,
        text_key: str = "text",
        emotion_key: str = "dominant_emotion",
        intensity_key: str = "intensity",
    ) -> List[Dict[str, Any]]:
        """Static helper to filter generic emotion/text records by SearchFilterCriteria."""
        filtered = []
        q_clean = criteria.query.strip().lower() if criteria.query and criteria.query.strip() else None
        emos_clean = [e.strip().lower() for e in criteria.emotions] if criteria.emotions else []

        for rec in records:
            # Query match
            if q_clean:
                text_val = f"{rec.get('text', '')} {rec.get('raw_text', '')} {rec.get('cleaned_text', '')} {rec.get('text_snippet', '')} {rec.get(text_key, '')}".lower()
                if q_clean not in text_val:
                    continue

            # Emotion match
            if emos_clean:
                dom_emo = str(rec.get(emotion_key, "")).strip().lower()
                if dom_emo not in emos_clean:
                    continue

            # Intensity match
            intensity_val = float(rec.get(intensity_key, 0.5))
            if criteria.min_intensity is not None and intensity_val < criteria.min_intensity:
                continue
            if criteria.max_intensity is not None and intensity_val > criteria.max_intensity:
                continue

            filtered.append(rec)
        return filtered

    @staticmethod
    def filter_recommendations(
        recommendations: List[Dict[str, Any]],
        criteria: SearchFilterCriteria,
    ) -> List[Dict[str, Any]]:
        """Static helper to filter recommendation item dicts by difficulty, activity_type, and query."""
        filtered = []
        q_clean = criteria.query.strip().lower() if criteria.query and criteria.query.strip() else None
        acts_clean = [a.strip().lower() for a in criteria.activity_types] if criteria.activity_types else []
        diff_clean = criteria.difficulty.strip().lower() if criteria.difficulty and criteria.difficulty.strip() != "All" else None

        for rec in recommendations:
            if diff_clean:
                rec_diff = str(rec.get("difficulty", "")).strip().lower()
                if rec_diff != diff_clean:
                    continue

            if acts_clean:
                rec_act = str(rec.get("activity_type", "")).strip().lower()
                if rec_act not in acts_clean:
                    continue

            if q_clean:
                rec_title = str(rec.get("title", "") or rec.get("recommendation", "")).lower()
                rec_desc = str(rec.get("description", "")).lower()
                if q_clean not in rec_title and q_clean not in rec_desc:
                    continue

            filtered.append(rec)
        return filtered

    @staticmethod
    def get_filter_summary(
        input_list: List[Any],
        filtered_list: List[Any],
        criteria: SearchFilterCriteria,
    ) -> Dict[str, Any]:
        """Returns summary statistics for a filtering operation."""
        total_in = len(input_list)
        matched = len(filtered_list)
        return {
            "total_input": total_in,
            "matched_count": matched,
            "filtered_out_count": total_in - matched,
            "match_percentage": round((matched / total_in) * 100.0, 2) if total_in > 0 else 0.0,
        }


# Singleton instance
_FILTER_ENGINE_INSTANCE: Optional[FilterEngine] = None


def get_filter_engine() -> FilterEngine:
    """Returns singleton FilterEngine instance."""
    global _FILTER_ENGINE_INSTANCE
    if _FILTER_ENGINE_INSTANCE is None:
        _FILTER_ENGINE_INSTANCE = FilterEngine()
    return _FILTER_ENGINE_INSTANCE
