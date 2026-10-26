from comparison.concepts import (
    ConceptMatches,
    ConceptReview,
    explicit_fields,
    field_name,
    reconcile,
)


def test_field_names_do_not_include_values_for_any_domain():
    for text, expected in [
        ("COURSE NAME: Database Fundamentals", "COURSE NAME"),
        ("Beneficiary: Alex Chen", "Beneficiary"),
        ("Tomato soup price: $8", "Tomato soup price"),
        ("Principal: $320,000", "Principal"),
    ]:
        assert field_name(text) == expected
    assert explicit_fields(
        [{"text": "Course name: A\nCourse credits: 3\nSome ordinary prose."}]
    ) == ["Course name", "Course credits"]


def test_semantic_guesses_cannot_merge_distinct_source_labels():
    class Model:
        def structured(self, schema, instruction, payload):
            if schema is ConceptReview:
                return ConceptReview(decisions=[{"source": "Course title", "equivalent": True}])
            return ConceptMatches(
                matches=[
                    {"source": "Course title", "canonical": "Course name"},
                    {"source": "Course name", "canonical": "Course name"},
                    {"source": "Contact hours", "canonical": "Contact hours"},
                    {"source": "Credit hours", "canonical": "Credit hours"},
                ]
            )

    result = reconcile(
        Model(),
        ["Course title", "Course name", "Contact hours", "Credit hours", "Unmapped provision"],
        lambda _: None,
    )
    assert result["Course name"] == ["Course name"]
    assert result["Course title"] == ["Course title"]
    assert "Contact hours" in result and "Credit hours" in result
    assert "Unmapped provision" in result


def test_broad_or_unreviewed_matches_preserve_original_field():
    class Model:
        def structured(self, schema, instruction, payload):
            if schema is ConceptMatches:
                return ConceptMatches(
                    matches=[{"source": "Outcome 6", "canonical": "Learning outcomes"}]
                )
            return ConceptReview(decisions=[{"source": "Outcome 6", "equivalent": False}])

    assert list(reconcile(Model(), ["Outcome 6"], lambda _: None)) == ["Outcome 6"]


def test_course_samples_have_matching_labels_not_course_values():
    from pathlib import Path

    from documents import extract, make_chunks

    root = Path(__file__).parents[1] / "samples"
    fields = []
    for name in ("Course Outline SDEV1201 - 2025.pdf", "Course Outline SDEV1701 - 2025.pdf"):
        path = root / name
        fields.append(explicit_fields(make_chunks(extract(name, path.read_bytes()))))
    for labels in fields:
        assert "COURSE NAME" in labels
        assert "COURSE DESCRIPTION" in labels
        assert "LEARNING OUTCOMES" in labels
        assert not any(
            "Database Fundamentals" in label or "UI/UX for Developers" in label for label in labels
        )


def test_model_matching_contract_restricts_names_to_selected_fields():
    from comparison.agents import AgentModel

    captured = {}

    class Chat:
        def with_structured_output(self, schema, method):
            captured["schema"] = schema
            return self

        def invoke(self, messages):
            return {"matches": [{"source": "Course title", "canonical": "Course name"}]}

    model = AgentModel.__new__(AgentModel)
    model.chat = Chat()
    model.structured(
        ConceptMatches,
        "Match fields",
        {"source_labels": ["Course title"], "known_concepts": ["Course name"]},
    )
    props = captured["schema"]["$defs"]["ConceptMatch"]["properties"]
    assert props["source"]["enum"] == ["Course title"]
    assert props["canonical"]["enum"] == ["Course name", "Course title"]


def test_literal_fields_keep_values_separate_and_reject_conflicts():
    from comparison.concepts import literal_value

    passages = [{"ref": "earlier:c1", "text": "COURSE NAME: Database Fundamentals", "location": 1}]
    value = literal_value("COURSE NAME", [], passages)
    assert value["value"] == "Database Fundamentals"
    assert value["sources"][0]["quote"] == passages[0]["text"]
    assert (
        literal_value(
            "COURSE NAME",
            [],
            passages + [{"ref": "earlier:c2", "text": "COURSE NAME: Another course"}],
        )
        is None
    )
    assert literal_value("Tuition fee", [], passages) is None


def test_alias_values_require_semantic_extraction_and_review():
    from comparison.concepts import literal_value

    assert (
        literal_value(
            "Executor",
            ["Personal representative"],
            [{"text": "Personal representative: Jamie Lee", "ref": "later:c1"}],
        )
        is None
    )


def test_long_provisions_require_extraction_instead_of_truncating_to_one_line():
    from comparison.concepts import literal_value

    assert (
        literal_value(
            "Conditions",
            [],
            [
                {
                    "ref": "earlier:c1",
                    "text": "Conditions: This provision has many words and continues with qualifications that need to be read together\nwith this continuation and its exceptions.",
                }
            ],
        )
        is None
    )


def test_table_labels_and_numbered_headings_exclude_blank_forms():
    assert explicit_fields(
        [{"text": "Capacity | 800 units\n4. Storage and handling\nOWNER: __________ Date: ______"}]
    ) == ["Capacity", "Storage and handling"]


def test_literal_alias_cannot_certify_unrelated_canonical_field():
    from comparison.concepts import literal_value
    passages = [{"ref": "earlier:c1", "text": "Review date: October 29, 2024"}]
    assert literal_value("Learning outcomes", ["Review date"], passages) is None
    assert literal_value("Review date", [], passages)["value"] == "October 29, 2024"
