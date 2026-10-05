"""
Reporting service for Text Sentiment & Deep Emotion Analysis.
Generates comprehensive tabular reports and aggregated summary statistics combining
Milestone 1 (VADER Sentiment) and Milestone 2 (Transformer Multi-label Emotion Analysis).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from services.config import EMOTIONS, EMOTION_DISPLAY_NAMES, DEFAULT_THRESHOLD, MEDICAL_DISCLAIMER
from services.emotion import analyze_emotion, get_emotion_classifier
from services.ingestion import IngestedItem
from services.preprocessing import preprocess_text
from services.sentiment import analyze_sentiment


def process_pipeline_items(
    items: List[IngestedItem],
    analyze_source_text: str = "original",
    include_emotion: bool = False,
    emotion_model_type: str = "bert",
    emotion_threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the end-to-end sentiment (and optional emotion) pipeline:
    Ingestion validation -> Preprocessing -> VADER Sentiment Scoring -> Transformer Emotion -> Reporting.

    Args:
        items: List of IngestedItem instances.
        analyze_source_text: Whether VADER should score 'original' or 'processed' text.
                             Default is 'original'. Transformers always use the original input text.
        include_emotion: Whether to also run BERT / DistilBERT multi-label emotion classification.
        emotion_model_type: 'bert' or 'distilbert'.
        emotion_threshold: Multi-label probability cutoff (default 0.50).

    Returns:
        (report_df, summary_stats_dict)
    """
    records = []
    total_samples = len(items)
    valid_samples = 0
    invalid_samples = 0
    analyzed_samples = 0
    pos_count = 0
    neg_count = 0
    neu_count = 0
    emotion_counts = {EMOTION_DISPLAY_NAMES[e]: 0 for e in EMOTIONS}

    classifier = None
    if include_emotion:
        try:
            classifier = get_emotion_classifier(emotion_model_type)
        except Exception:
            classifier = None

    for item in items:
        if not item.is_valid:
            invalid_samples += 1
            row_dict = {
                "ID": item.id,
                "Input Text": item.text,
                "Processed Text": "",
                "Sentiment": "Invalid",
                "Positive": None,
                "Negative": None,
                "Neutral": None,
                "Compound": None,
                "Status": "Invalid",
                "Error": item.error_message or "Validation failed",
            }
            if include_emotion:
                row_dict.update({
                    "Emotion Model": emotion_model_type.upper(),
                    "Primary Emotion": "N/A",
                    "Primary Confidence": None,
                    "Detected Emotions": "N/A",
                    "Combined Analysis": "Invalid Input",
                })
                for emo in EMOTIONS:
                    row_dict[f"Prob_{EMOTION_DISPLAY_NAMES[emo]}"] = None
            records.append(row_dict)
            continue

        valid_samples += 1
        # Preprocess text (Milestone 1)
        processed = preprocess_text(item.text)

        # Sentiment Analysis (Milestone 1)
        text_to_score = item.text if analyze_source_text == "original" else processed
        scores = analyze_sentiment(text_to_score)

        analyzed_samples += 1
        sentiment = scores["sentiment"]
        if sentiment == "Positive":
            pos_count += 1
        elif sentiment == "Negative":
            neg_count += 1
        else:
            neu_count += 1

        row_dict = {
            "ID": item.id,
            "Input Text": item.text,
            "Processed Text": processed,
            "Sentiment": sentiment,
            "Positive": scores["pos"],
            "Negative": scores["neg"],
            "Neutral": scores["neu"],
            "Compound": scores["compound"],
            "Status": "Valid",
            "Error": None,
        }

        # Emotion Analysis (Milestone 2)
        if include_emotion and classifier is not None and classifier.is_loaded:
            # Transformer uses the preserved original text for rich context
            emotion_pred = classifier.predict(item.text, threshold=emotion_threshold)
            primary_emo = emotion_pred.primary_emotion
            primary_conf = emotion_pred.primary_confidence
            
            detected_list = [f"{d['emotion']} ({d['confidence_pct']})" for d in emotion_pred.detected_emotions]
            detected_str = ", ".join(detected_list) if detected_list else f"None (Primary: {primary_emo})"

            if primary_emo in emotion_counts:
                emotion_counts[primary_emo] += 1

            combined_analysis = f"{sentiment} Sentiment | Primary Emotion: {primary_emo} ({primary_conf * 100:.1f}%)"
            if detected_list:
                combined_analysis += f" | Detected: {', '.join([d['emotion'] for d in emotion_pred.detected_emotions])}"

            row_dict.update({
                "Emotion Model": emotion_model_type.upper(),
                "Primary Emotion": primary_emo,
                "Primary Confidence": round(primary_conf, 4),
                "Detected Emotions": detected_str,
                "Combined Analysis": combined_analysis,
            })

            # Add all 6 individual probabilities
            for emo in EMOTIONS:
                display_name = EMOTION_DISPLAY_NAMES[emo]
                row_dict[f"Prob_{display_name}"] = round(emotion_pred.probabilities.get(display_name, 0.0), 4)

        elif include_emotion:
            row_dict.update({
                "Emotion Model": emotion_model_type.upper(),
                "Primary Emotion": "Model Not Loaded",
                "Primary Confidence": None,
                "Detected Emotions": "N/A",
                "Combined Analysis": f"{sentiment} Sentiment (Emotion Model Not Loaded)",
            })
            for emo in EMOTIONS:
                row_dict[f"Prob_{EMOTION_DISPLAY_NAMES[emo]}"] = None

        records.append(row_dict)

    report_df = pd.DataFrame(records)

    summary_stats = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "analyzed_samples": analyzed_samples,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "emotion_counts": emotion_counts if include_emotion else None,
    }

    return report_df, summary_stats


def generate_sentiment_report(
    items: List[IngestedItem],
    analyze_source_text: str = "original",
) -> pd.DataFrame:
    """Milestone 1 convenience function returning only the DataFrame report."""
    df, _ = process_pipeline_items(items, analyze_source_text=analyze_source_text, include_emotion=False)
    return df


def generate_complete_report(
    items: List[IngestedItem],
    emotion_model_type: str = "bert",
    emotion_threshold: float = DEFAULT_THRESHOLD,
    analyze_source_text: str = "original",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Milestone 1 + 2 integrated report function."""
    return process_pipeline_items(
        items,
        analyze_source_text=analyze_source_text,
        include_emotion=True,
        emotion_model_type=emotion_model_type,
        emotion_threshold=emotion_threshold,
    )


def calculate_summary_stats(items: List[IngestedItem]) -> Dict[str, Any]:
    """Convenience function returning only the summary statistics dictionary."""
    _, stats = process_pipeline_items(items)
    return stats


def export_records_to_csv(
    records: Any,
    output_filepath: Optional[str] = None,
) -> Any:
    """Task 5: Exports a pandas DataFrame or list of dicts to CSV file path or bytes."""
    if isinstance(records, pd.DataFrame):
        df = records
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame()

    if df.empty:
        csv_bytes = b"Status\nNo records available for the selected range\n"
    else:
        csv_bytes = df.to_csv(index=False).encode("utf-8")

    if output_filepath:
        out_p = Path(output_filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(csv_bytes)
        return str(output_filepath)
    return csv_bytes


def export_analytics_pdf(
    report_title: str = "Employee Wellness & Emotion Analytics Summary Report",
    user_id: str = "default_user",
    date_range_str: str = "All Time",
    summary_kpis: Optional[Dict[str, Any]] = None,
    emotional_records: Optional[List[Dict[str, Any]]] = None,
    recommendation_history: Optional[List[Dict[str, Any]]] = None,
    output_filepath: Optional[str] = None,
    analytics_summary: Optional[Dict[str, Any]] = None,
    emotion_records: Optional[List[Dict[str, Any]]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """
    Task 5: Generates a structured PDF report containing date ranges, emotional summary KPIs,
    trend analysis, recommendation history, and disclaimer using ReportLab.
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    kpis = summary_kpis or analytics_summary or {}
    e_records = emotional_records or emotion_records or []
    r_history = recommendation_history or recommendations or []

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    # Title & Metadata
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, textColor=colors.HexColor("#065F46"))
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#4B5563"))
    disc_style = ParagraphStyle("DiscStyle", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#92400E"), backColor=colors.HexColor("#FEF3C7"), borderPadding=6)

    story.append(Paragraph(f"🌿 MoodMentor — {report_title}", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>User ID:</b> {user_id} &nbsp;|&nbsp; <b>Date Range:</b> {date_range_str} &nbsp;|&nbsp; <b>Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Medical Disclaimer:</b> {MEDICAL_DISCLAIMER}", disc_style))
    story.append(Spacer(1, 14))

    # Summary KPIs Table
    story.append(Paragraph("<b>📊 Summary Analytics & Key Performance Indicators</b>", styles["Heading2"]))
    kpi_data = [
        ["Metric", "Value"],
        ["Total Historical Check-ins", str(kpis.get("total_checkins", len(e_records)))],
        ["Dominant Emotion", str(kpis.get("dominant_emotion", "N/A"))],
        ["Average Emotion Intensity", f"{float(kpis.get('avg_intensity', kpis.get('mean_intensity', 0.5))):.2f}"],
        ["Total Feedback Events", str(kpis.get("total_feedback", kpis.get("total_feedback_events", 0)))],
        ["Accepted Recommendations", str(kpis.get("accepted_count", 0))],
        ["Average Feedback Star Rating", str(kpis.get("avg_rating", "N/A"))],
    ]
    kpi_table = Table(kpi_data, colWidths=[250, 250])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#065F46")),
        ("TEXTCOLOR", (0, 0), (1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 16))

    # Recent Emotional Check-ins Table
    story.append(Paragraph("<b>📈 Recent Emotional Records Log</b>", styles["Heading2"]))
    if not e_records:
        story.append(Paragraph("<i>No emotional check-in records available for this date range.</i>", styles["Normal"]))
    else:
        rec_data = [["Timestamp", "Dominant Emotion", "Intensity", "Severity", "Raw Input Text"]]
        for r in e_records[:10]:
            ts = str(r.get("timestamp") or r.get("date") or "N/A")[:16]
            emo = str(r.get("dominant_emotion", "N/A"))
            inten = f"{float(r.get('intensity', 0.5)):.2f}"
            sev = str(r.get("severity", r.get("severity_level", "Moderate")))
            raw_txt = str(r.get("raw_text") or r.get("text") or r.get("text_snippet") or "")[:40]
            rec_data.append([ts, emo, inten, sev, raw_txt])

        rec_table = Table(rec_data, colWidths=[100, 90, 60, 70, 180])
        rec_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#10B981")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(rec_table)

    story.append(Spacer(1, 16))

    # Recent Recommendation History Table
    story.append(Paragraph("<b>🌟 Recommendation History & Feedback Events</b>", styles["Heading2"]))
    if not r_history:
        story.append(Paragraph("<i>No recommendation history recorded for this date range.</i>", styles["Normal"]))
    else:
        rec_hist_data = [["Timestamp", "Activity Type", "Match Score", "Feedback Status", "Rating"]]
        for rh in r_history[:10]:
            ts = str(rh.get("timestamp") or rh.get("feedback_timestamp") or "N/A")[:16]
            act = str(rh.get("activity_type") or rh.get("recommendation_type") or "N/A")
            score = f"{float(rh.get('recommendation_score', rh.get('score', 0.0))):.3f}"
            status = "Accepted" if rh.get("accepted") or rh.get("was_liked") else ("Rejected" if rh.get("rejected") or rh.get("was_disliked") else "Viewed")
            rating = f"{rh.get('rating')} Stars" if rh.get("rating") is not None else "N/A"
            rec_hist_data.append([ts, act, score, status, rating])

        rh_table = Table(rec_hist_data, colWidths=[110, 130, 80, 90, 90])
        rh_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(rh_table)

    doc.build(story)
    pdf_bytes = buffer.getvalue()

    if output_filepath:
        out_p = Path(output_filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(pdf_bytes)
        return str(output_filepath)
    return pdf_bytes

