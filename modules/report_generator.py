from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

def format_value(value):
    """Format values for a readable PDF."""

    if value is None:
        return "-"

    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def format_recommendations(recommendations):
    """Convert recommendation dictionary into readable rows."""

    rows = []

    if not isinstance(recommendations, dict):
        return rows

    for feature, result in recommendations.items():

        if not isinstance(result, dict):
            continue

        rows.append([
            str(feature),
            "Yes" if result.get("bias_detected", False) else "No",
            str(result.get("strategy", "None")),
            str(result.get("reason", ""))
        ])

    return rows


def extract_fairness_rows(fairness_metrics):
    """Flatten FairLens fairness results into table rows."""

    rows = []

    if not isinstance(fairness_metrics, dict):
        return rows

    for stage, feature_data in fairness_metrics.items():

        if not isinstance(feature_data, dict):
            continue

        for feature, result in feature_data.items():

            if not isinstance(result, dict):
                continue

            metrics = result.get("metrics", {})

            if not isinstance(metrics, dict):
                continue

            for metric_name, value in metrics.items():

                rows.append([
                    str(stage),
                    str(feature),
                    str(metric_name),
                    format_value(value)
                ])

    return rows


def extract_shap_importance(explainability_summary):
    """Extract only useful SHAP feature importance information."""

    if not isinstance(explainability_summary, dict):
        return []

    importance = explainability_summary.get(
        "feature_importance"
    )

    if importance is None:
        return []

    rows = []

    try:

        if hasattr(importance, "iterrows"):

            for _, row in importance.iterrows():

                feature = row.get("Feature", "Unknown")
                value = row.get("SHAP Importance", 0)

                rows.append([
                    str(feature),
                    format_value(value)
                ])

    except Exception:
        pass

    return rows

def pdf_cell(value, font_size=7):
    """Create a wrapped PDF table cell."""

    if value is None:
        value = "-"

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return Paragraph(
        str(value),
        ParagraphStyle(
            "TableCell",
            fontSize=font_size,
            leading=font_size + 2,
            wordWrap="CJK"
        )
    )


def format_number(value):
    """Format fairness metric values for readable PDF tables."""

    if value is None:
        return "-"

    # Convert NumPy scalar values
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    # Single numeric value
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"

    # Dictionary values such as:
    # {'Male': 0.7813, 'Female': 0.9052}
    if isinstance(value, dict):

        formatted_items = []

        for group, group_value in value.items():

            if hasattr(group_value, "item"):
                try:
                    group_value = group_value.item()
                except Exception:
                    pass

            if isinstance(group_value, (int, float)):
                formatted_items.append(
                    f"{group}: {float(group_value):.4f}"
                )
            else:
                formatted_items.append(
                    f"{group}: {group_value}"
                )

        return "<br/>".join(formatted_items)

    return str(value)

def generate_pdf_report(
    file_path,
    dataset_summary,
    model_name,
    task_type,
    performance_metrics,
    sensitive_features,
    fairness_metrics,
    explainability_summary,
    bias_severity,
    recommended_mitigation,
    applied_mitigation,
    before_after_comparison,
    final_recommendation
):

    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=10,
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13
    )

    story = []

    # ============================================================
    # TITLE
    # ============================================================

    story.append(
        Paragraph(
            "FairLens - Bias Audit Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Fairness, Performance and Explainability Analysis",
            subtitle_style
        )
    )

    # ============================================================
    # DATASET SUMMARY
    # ============================================================

    story.append(
        Paragraph(
            "1. Dataset Summary",
            section_style
        )
    )

    dataset_data = [
        ["Property", "Value"]
    ]

    for key, value in dataset_summary.items():

        if isinstance(value, list):
            value = ", ".join(map(str, value))

        dataset_data.append([
            str(key),
            str(value)
        ])

    dataset_table = Table(
        dataset_data,
        colWidths=[2.2 * 72, 4.5 * 72]
    )

    dataset_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4F4F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F5F5F5")])
        ])
    )

    story.append(dataset_table)
    story.append(Spacer(1, 12))

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    story.append(
        Paragraph(
            "2. Model Information",
            section_style
        )
    )

    model_data = [
        ["Property", "Value"],
        ["Model", str(model_name)],
        ["Task Type", str(task_type)],
        [
            "Sensitive Attributes",
            ", ".join(map(str, sensitive_features))
            if sensitive_features
            else "None"
        ]
    ]

    model_table = Table(
        model_data,
        colWidths=[2.2 * 72, 4.5 * 72]
    )

    model_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4F4F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ])
    )

    story.append(model_table)
    story.append(Spacer(1, 12))

    # ============================================================
    # PERFORMANCE METRICS
    # ============================================================

    story.append(
        Paragraph(
            "3. Model Performance",
            section_style
        )
    )

    if performance_metrics:

        if isinstance(performance_metrics, list):

            keys = list(performance_metrics[0].keys())

            performance_data = [
                [str(k) for k in keys]
            ]

            for row in performance_metrics:

                performance_data.append([
                    str(row.get(k, "-"))
                    for k in keys
                ])

            performance_table = Table(
                performance_data,
                repeatRows=1
            )

            performance_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0),
                     colors.HexColor("#2F4F4F")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP")
                ])
            )

            story.append(performance_table)

        else:

            story.append(
                Paragraph(
                    str(performance_metrics),
                    body_style
                )
            )

    else:

        story.append(
            Paragraph(
                "No performance metrics available.",
                body_style
            )
        )

    story.append(Spacer(1, 12))

    # ============================================================
    # FAIRNESS EVALUATION
    # ============================================================

    story.append(
        Paragraph(
            "4. Fairness Evaluation",
            section_style
        )
    )


    def format_fairness_value(value):
        """
        Safely format fairness metric values for PDF tables.
        Handles normal numbers and dictionary-based metrics
        such as Group Accuracy.
        """

        if value is None:
            return "-"

        # Convert numpy values to Python values
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass

        # Handle dictionary values
        if isinstance(value, dict):

            formatted_items = []

            for group, score in value.items():

                if hasattr(score, "item"):
                    try:
                        score = score.item()
                    except Exception:
                        pass

                if isinstance(score, (int, float)):
                    formatted_items.append(
                        f"{group}: {score:.4f}"
                    )
                else:
                    formatted_items.append(
                        f"{group}: {score}"
                    )

            # <br/> forces ReportLab Paragraph wrapping
            return "<br/>".join(formatted_items)

        # Handle numeric values
        if isinstance(value, (int, float)):
            return f"{value:.4f}"

        return str(value)


    # ------------------------------------------------------------
    # Build fairness table
    # ------------------------------------------------------------

    fairness_data = [
        [
            Paragraph("Stage", ParagraphStyle(
                "FairnessHeader1",
                fontSize=7,
                leading=8,
                textColor=colors.white
            )),

            Paragraph("Sensitive Attribute", ParagraphStyle(
                "FairnessHeader2",
                fontSize=7,
                leading=8,
                textColor=colors.white
            )),

            Paragraph("Metric", ParagraphStyle(
                "FairnessHeader3",
                fontSize=7,
                leading=8,
                textColor=colors.white
            )),

            Paragraph("Value", ParagraphStyle(
                "FairnessHeader4",
                fontSize=7,
                leading=8,
                textColor=colors.white
            ))
        ]
    ]


    if isinstance(fairness_metrics, dict):

        # --------------------------------------------------------
        # Case 1:
        # Before/After structure
        # --------------------------------------------------------

        for stage, attributes in fairness_metrics.items():

            if not isinstance(attributes, dict):
                continue

            for attribute, result in attributes.items():

                if not isinstance(result, dict):
                    continue

                metrics = result.get("metrics", {})

                if not isinstance(metrics, dict):
                    continue

                for metric_name, value in metrics.items():

                    formatted_value = format_fairness_value(value)

                    fairness_data.append([
                        Paragraph(
                            str(stage),
                            ParagraphStyle(
                                "FairnessCellStage",
                                fontSize=6.5,
                                leading=8,
                                wordWrap="CJK"
                            )
                        ),

                        Paragraph(
                            str(attribute),
                            ParagraphStyle(
                                "FairnessCellAttribute",
                                fontSize=6.5,
                                leading=8,
                                wordWrap="CJK"
                            )
                        ),

                        Paragraph(
                            str(metric_name),
                            ParagraphStyle(
                                "FairnessCellMetric",
                                fontSize=6.5,
                                leading=8,
                                wordWrap="CJK"
                            )
                        ),

                        Paragraph(
                            formatted_value,
                            ParagraphStyle(
                                "FairnessCellValue",
                                fontSize=6.5,
                                leading=8,
                                wordWrap="CJK"
                            )
                        )
                    ])


    if len(fairness_data) > 1:

        fairness_table = Table(
            fairness_data,

            # IMPORTANT:
            # Total = 468 points.
            # This fits inside the letter page width
            # after 40pt left/right margins.
            colWidths=[
                65,
                95,
                180,
                128
            ],

            repeatRows=1
        )

        fairness_table.setStyle(
            TableStyle([

                # Header
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#2F4F4F")
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                # Border
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                # Alignment
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                # Padding
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),

                # Alternating rows
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F5F5F5")
                    ]
                )
            ])
        )

        story.append(fairness_table)

    else:

        story.append(
            Paragraph(
                "No fairness metrics available.",
                body_style
            )
        )

    story.append(Spacer(1, 12))
    # ============================================================
    # BIAS SEVERITY
    # ============================================================

    story.append(
        Paragraph(
            "5. Bias Assessment",
            section_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Overall Assessment:</b> {bias_severity}",
            body_style
        )
    )

    story.append(Spacer(1, 12))

    # ============================================================
    # MITIGATION ANALYSIS
    # ============================================================

    story.append(
        Paragraph(
            "6. Mitigation Analysis",
            section_style
        )
    )

    story.append(
        Paragraph(
            "<b>Recommended Mitigation</b>",
            body_style
        )
    )

    recommendation_data = [
        [
            pdf_cell("Sensitive Attribute"),
            pdf_cell("Bias Detected"),
            pdf_cell("Recommended Strategy"),
            pdf_cell("Reason")
        ]
    ]

    if isinstance(recommended_mitigation, dict):

        for attribute, result in recommended_mitigation.items():

            if not isinstance(result, dict):
                continue

            bias_detected = result.get(
                "bias_detected",
                False
            )

            strategy = result.get(
                "strategy",
                None
            )

            reason = result.get(
                "reason",
                "No significant bias was detected."
            )

            if strategy is None:
                strategy = "No mitigation required"

            recommendation_data.append([
                pdf_cell(attribute),
                pdf_cell(
                    "Yes" if bool(bias_detected) else "No"
                ),
                pdf_cell(strategy),
                pdf_cell(reason)
            ])


    if len(recommendation_data) > 1:

        recommendation_table = Table(
            recommendation_data,
            colWidths=[
                1.1 * 72,
                0.9 * 72,
                1.6 * 72,
                3.0 * 72
            ],
            repeatRows=1
        )

        recommendation_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                colors.HexColor("#2F4F4F")),

                ("TEXTCOLOR", (0, 0), (-1, 0),
                colors.white),

                ("FONTNAME", (0, 0), (-1, 0),
                "Helvetica-Bold"),

                ("GRID", (0, 0), (-1, -1),
                0.5, colors.grey),

                ("VALIGN", (0, 0), (-1, -1),
                "TOP"),

                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
            ])
        )

        story.append(recommendation_table)

    else:

        story.append(
            Paragraph(
                "No specific mitigation recommendation was generated.",
                body_style
            )
        )


    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "<b>Applied Mitigation</b>",
            body_style
        )
    )

    story.append(
        Paragraph(
            str(applied_mitigation),
            body_style
        )
    )

    story.append(Spacer(1, 12))

    # ============================================================
    # BEFORE VS AFTER
    # ============================================================

    story.append(
        Paragraph(
            "7. Before vs After Comparison",
            section_style
        )
    )

    if before_after_comparison:

        for category, values in before_after_comparison.items():

            story.append(
                Paragraph(
                    f"<b>{category}</b>",
                    body_style
                )
            )

            if isinstance(values, list) and values:

                keys = list(values[0].keys())

                comparison_data = [
                    [str(k) for k in keys]
                ]

                for row in values:

                    comparison_data.append([
                        pdf_cell(row.get(k, "-"))
                        for k in keys
                    ])

                comparison_table = Table(
                    comparison_data,
                    repeatRows=1
                )

                comparison_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0),
                         colors.HexColor("#2F4F4F")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0),
                         "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1),
                         0.5, colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP")
                    ])
                )

                story.append(comparison_table)

            elif isinstance(values, dict):

                # ========================================================
                # FAIRNESS BEFORE VS AFTER
                # ========================================================

                if category == "Fairness":

                    fairness_comparison_data = [
                        [
                            pdf_cell("Stage", 7),
                            pdf_cell("Sensitive Attribute", 7),
                            pdf_cell("Metric", 7),
                            pdf_cell("Value", 7)
                        ]
                    ]

                    # values structure:
                    #
                    # {
                    #     "original": {
                    #         "sex": {
                    #             "metrics": {
                    #                 "Demographic Parity Difference": ...,
                    #                 ...
                    #             }
                    #         }
                    #     },
                    #
                    #     "mitigated": {
                    #         "sex": {
                    #             "metrics": {
                    #                 ...
                    #             }
                    #         }
                    #     }
                    # }

                    for stage, feature_data in values.items():

                        if not isinstance(feature_data, dict):
                            continue

                        for feature, result in feature_data.items():

                            if not isinstance(result, dict):
                                continue

                            metrics = result.get("metrics", {})

                            if not isinstance(metrics, dict):
                                continue

                            for metric_name, value in metrics.items():

                                # Handle dictionary-valued metrics
                                if isinstance(value, dict):

                                    for group, group_value in value.items():

                                        if hasattr(group_value, "item"):
                                            try:
                                                group_value = group_value.item()
                                            except Exception:
                                                pass

                                        if isinstance(
                                            group_value,
                                            (int, float)
                                        ):
                                            formatted_value = (
                                                f"{group}: "
                                                f"{group_value:.4f}"
                                            )
                                        else:
                                            formatted_value = (
                                                f"{group}: "
                                                f"{group_value}"
                                            )

                                        fairness_comparison_data.append([
                                            pdf_cell(stage, 6.5),
                                            pdf_cell(feature, 6.5),
                                            pdf_cell(metric_name, 6.5),
                                            pdf_cell(formatted_value, 6.5)
                                        ])

                                else:

                                    if hasattr(value, "item"):
                                        try:
                                            value = value.item()
                                        except Exception:
                                            pass

                                    if isinstance(value, (int, float)):
                                        formatted_value = f"{value:.4f}"
                                    else:
                                        formatted_value = str(value)

                                    fairness_comparison_data.append([
                                        pdf_cell(stage, 6.5),
                                        pdf_cell(feature, 6.5),
                                        pdf_cell(metric_name, 6.5),
                                        pdf_cell(formatted_value, 6.5)
                                    ])


                    if len(fairness_comparison_data) > 1:

                        comparison_table = Table(
                            fairness_comparison_data,

                            colWidths=[
                                65,
                                95,
                                180,
                                128
                            ],

                            repeatRows=1
                        )

                        comparison_table.setStyle(
                            TableStyle([

                                # Header
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#2F4F4F")
                                ),

                                (
                                    "TEXTCOLOR",
                                    (0, 0),
                                    (-1, 0),
                                    colors.white
                                ),

                                (
                                    "FONTNAME",
                                    (0, 0),
                                    (-1, 0),
                                    "Helvetica-Bold"
                                ),

                                # Border
                                (
                                    "GRID",
                                    (0, 0),
                                    (-1, -1),
                                    0.5,
                                    colors.grey
                                ),

                                # Wrapping / alignment
                                (
                                    "VALIGN",
                                    (0, 0),
                                    (-1, -1),
                                    "TOP"
                                ),

                                (
                                    "LEFTPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    4
                                ),

                                (
                                    "RIGHTPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    4
                                ),

                                (
                                    "TOPPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    4
                                ),

                                (
                                    "BOTTOMPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    4
                                ),

                                (
                                    "ROWBACKGROUNDS",
                                    (0, 1),
                                    (-1, -1),
                                    [
                                        colors.white,
                                        colors.HexColor("#F5F5F5")
                                    ]
                                )
                            ])
                        )

                        story.append(comparison_table)

                    else:

                        story.append(
                            Paragraph(
                                "No fairness comparison data available.",
                                body_style
                            )
                        )

                else:

                    # ========================================================
                    # OTHER DICTIONARY DATA
                    # ========================================================

                    comparison_data = [
                        [
                            pdf_cell("Metric", 7),
                            pdf_cell("Value", 7)
                        ]
                    ]

                    for key, value in values.items():

                        comparison_data.append([
                            pdf_cell(key, 7),
                            pdf_cell(value, 7)
                        ])

                    comparison_table = Table(
                        comparison_data,
                        colWidths=[
                            3.5 * 72,
                            3.2 * 72
                        ],
                        repeatRows=1
                    )

                    comparison_table.setStyle(
                        TableStyle([
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor("#2F4F4F")
                            ),
                            (
                                "TEXTCOLOR",
                                (0, 0),
                                (-1, 0),
                                colors.white
                            ),
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, 0),
                                "Helvetica-Bold"
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.5,
                                colors.grey
                            ),
                            (
                                "FONTSIZE",
                                (0, 0),
                                (-1, -1),
                                8
                            ),
                            (
                                "VALIGN",
                                (0, 0),
                                (-1, -1),
                                "TOP"
                            )
                        ])
                    )

                    story.append(comparison_table)

            else:

                story.append(
                    Paragraph(
                        str(values),
                        body_style
                    )
                )

            story.append(Spacer(1, 8))

    else:

        story.append(
            Paragraph(
                "No before vs after comparison is available.",
                body_style
            )
        )

    story.append(Spacer(1, 8))

    # ============================================================
    # EXPLAINABILITY
    # ============================================================

    story.append(
        Paragraph(
            "8. Explainability",
            section_style
        )
    )

    shap_rows = extract_shap_importance(
        explainability_summary
    )

    if shap_rows:

        shap_data = [
            ["Feature", "SHAP Importance"]
        ]

        # Show top 10 features only
        shap_data.extend(
            shap_rows[:10]
        )

        shap_table = Table(
            shap_data,
            colWidths=[
                4.5 * 72,
                1.5 * 72
            ],
            repeatRows=1
        )

        shap_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                colors.HexColor("#2F4F4F")),
                ("TEXTCOLOR", (0, 0), (-1, 0),
                colors.white),
                ("FONTNAME", (0, 0), (-1, 0),
                "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1),
                0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP")
            ])
        )

        story.append(shap_table)

        story.append(
            Spacer(1, 6)
        )

        story.append(
            Paragraph(
                "SHAP importance indicates the relative contribution "
                "of each feature to the model's predictions. Higher "
                "values indicate greater influence on model behavior.",
                body_style
            )
        )

    else:

        story.append(
            Paragraph(
                "SHAP feature importance is not available for this model.",
                body_style
            )
        )

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    story.append(
        Paragraph(
            "9. Recommendations",
            section_style
        )
    )

    recommendation_rows = format_recommendations(
        recommended_mitigation
    )

    if recommendation_rows:

        recommendation_data = [
            [
                "Sensitive Attribute",
                "Bias Detected",
                "Recommended Strategy",
                "Reason"
            ]
        ]

        recommendation_data.extend(
            recommendation_rows
        )

        recommendation_table = Table(
            recommendation_data,
            colWidths=[
                1.0 * 72,
                0.8 * 72,
                1.5 * 72,
                3.0 * 72
            ],
            repeatRows=1
        )

        recommendation_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                colors.HexColor("#2F4F4F")),
                ("TEXTCOLOR", (0, 0), (-1, 0),
                colors.white),
                ("FONTNAME", (0, 0), (-1, 0),
                "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1),
                0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP")
            ])
        )

        story.append(
            recommendation_table
        )

    else:

        story.append(
            Paragraph(
                "No specific mitigation recommendation was generated.",
                body_style
            )
        )

    story.append(Spacer(1, 12))

    # ============================================================
    # FINAL CONCLUSION
    # ============================================================

    story.append(
        Paragraph(
            "10. Final Conclusion",
            section_style
        )
    )

    story.append(
        Paragraph(
            str(final_recommendation),
            body_style
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by FairLens - Bias Auditing Toolkit",
            subtitle_style
        )
    )

    # ============================================================
    # BUILD PDF
    # ============================================================

    doc.build(story)