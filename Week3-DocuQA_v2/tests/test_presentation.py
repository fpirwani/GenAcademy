from comparison.presentation import ReportOrganization, organize_report


def row(name, a="10", b="20"):
    return {
        "topic": name,
        "one": {"status": "found", "value": a, "sources": []},
        "two": {"status": "found", "value": b, "sources": []},
    }


def test_related_changes_group_and_unchanged_go_to_additional():
    class Model:
        def structured(self, schema, instruction, payload):
            return ReportOrganization(
                groups=[{"topic": "Payment terms", "members": ["R0", "R1"]}], minor=[]
            )

    rows = [row("Amount"), row("Rate"), row("Notice", "30 days", "30 days")]
    result = organize_report(Model(), rows)
    assert [r["topic"] for r in result] == ["Payment terms"]
    assert result[0]["details"] == rows[:2]
    assert len(result) == 1


def test_five_group_cap_retains_unassigned_and_one_sided_changes():
    class Model:
        def structured(self, schema, instruction, payload):
            return ReportOrganization(
                groups=[{"topic": f"Group {i}", "members": [f"R{i}"]} for i in range(9)], minor=[]
            )

    rows = [row(str(i)) for i in range(12)]
    rows[0]["two"] = {
        "status": "not_found",
        "value": "No corresponding section found",
        "sources": [],
    }
    result = organize_report(Model(), rows)
    assert len(result) <= 5
    assert sorted(r["topic"] for g in result for r in g["details"]) == sorted(
        r["topic"] for r in rows
    )


def test_failed_organization_keeps_uncertain_changes_visible():
    class Model:
        def structured(self, *args):
            raise RuntimeError("offline")

    result = organize_report(Model(), [row("Amount"), row("Same", "a", "a")])
    assert result[0]["topic"] == "Substantive differences"
    assert len(result) == 1
    assert (
        result[0]["difference_summaries"]["Amount"]
        == "The reported details differ between the documents."
    )
    assert "uncertain" not in result[0]["difference_summaries"]["Amount"]


def test_grouped_fallback_summarizes_changed_fields_only():
    from comparison.presentation import factual_difference

    value = row("Course", "Name: Databases\nCredits: 3", "Name: Design\nCredits: 3")
    assert factual_difference(value) == "Differences in Name."


def test_summary_does_not_repeat_legacy_source_dump():
    from comparison.presentation import concise_difference

    value = row("Terms", "Long source text " * 50, "Other source text " * 50)
    summary = concise_difference(value, "Compared values: " + value["one"]["value"])
    assert len(summary) < 100
    assert "Long source text" not in summary
    assert (
        concise_difference(value, "The repayment periods differ.")
        == "The repayment periods differ."
    )


def test_five_substantive_groups_plus_additional():
    class Model:
        def structured(self, schema, instruction, payload):
            return schema(groups=[{"topic": f"Topic {i}", "members": [f"R{i}"]} for i in range(5)])

    result = organize_report(Model(), [row(str(i)) for i in range(5)] + [row("Same", "x", "x")])
    assert len(result) == 5
    assert all(g["topic"] != "Additional provisions" for g in result)


def test_minor_wording_but_not_numbers_moves_to_additional():
    class Model:
        def structured(self, schema, instruction, payload):
            return schema(
                groups=[],
                decisions=[
                    {"member": "R0", "importance": "minor", "reason": "Equivalent wording."},
                    {"member": "R1", "importance": "minor", "reason": "Minor amount change."},
                ],
            )

    result = organize_report(
        Model(), [row("Wording", "pay monthly", "monthly payment"), row("Price")]
    )
    assert result[0]["details"][0]["topic"] == "Price"
    assert len(result) == 1


def test_topics_ranked_by_importance():
    class Model:
        def structured(self, schema, instruction, payload):
            return schema(
                groups=[
                    {"topic": "Lower", "members": ["R0"]},
                    {"topic": "Higher", "members": ["R1"]},
                ],
                decisions=[
                    {
                        "member": "R0",
                        "importance": "substantive",
                        "priority": 2,
                        "reason": "Lower impact.",
                    },
                    {
                        "member": "R1",
                        "importance": "substantive",
                        "priority": 5,
                        "reason": "Higher impact.",
                    },
                ],
            )

    result = organize_report(Model(), [row("A"), row("B")])
    assert [g["topic"] for g in result] == ["Higher", "Lower"]


def test_contextual_difference_keeps_names_values_and_conditions():
    from comparison.presentation import concise_difference

    summary = (
        "Alpha estimates monthly principal and interest at $1,970.30. "
        "Beta estimates $2,246.67. Both exclude escrow and other charges."
    )
    value = row("Payments", "estimated $1,970.30", "estimated $2,246.67")
    assert concise_difference(value, summary) == summary


def test_organizer_receives_document_names_and_verified_identity_context():
    captured = {}

    class Model:
        def structured(self, schema, instruction, payload):
            captured.update(payload)
            return schema(groups=[])

    organize_report(
        Model(),
        [row("Provider", "Alpha", "Beta"), row("Amount")],
        document_names={"one": "alpha.pdf", "two": "beta.docx"},
    )
    assert captured["document_names"]["one"] == "alpha.pdf"
    assert captured["verified_context"][0]["one"] == "Alpha"
