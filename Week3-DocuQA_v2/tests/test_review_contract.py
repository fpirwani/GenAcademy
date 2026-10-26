from comparison.agents import AgentModel
from comparison.holistic_schemas import FieldReview


def test_group_review_contract_uses_group_names_not_members():
    captured = {}

    class Chat:
        def with_structured_output(self, schema, method):
            captured.update(schema)
            return self

        def invoke(self, messages):
            return {"decisions": [{"topic": "Identity", "supported": True}]}

    model = AgentModel.__new__(AgentModel)
    model.chat = Chat()
    verdict = model.structured(
        FieldReview, "Review", {"fields": [{"topic": "Identity", "members": ["Name", "Code"]}]}
    )
    assert verdict.decisions[0].topic == "Identity"
    assert captured["properties"]["decisions"]["minItems"] == 1
    assert captured["properties"]["decisions"]["maxItems"] == 1
    assert captured["$defs"]["FieldDecision"]["properties"]["topic"]["enum"] == ["Identity"]


def test_review_isolates_groups_and_combines_decisions():
    seen = []

    class Chat:
        def with_structured_output(self, schema, method):
            self.topic = schema["$defs"]["FieldDecision"]["properties"]["topic"]["enum"]
            seen.append(self.topic)
            return self

        def invoke(self, messages):
            return {"decisions": [{"topic": self.topic[0], "supported": True}]}

    model = AgentModel.__new__(AgentModel)
    model.chat = Chat()
    result = model.structured(
        FieldReview, "Review", {"fields": [{"topic": "Identity"}, {"topic": "Conditions"}]}
    )
    assert seen == [["Identity"], ["Conditions"]]
    assert [d.topic for d in result.decisions] == ["Identity", "Conditions"]
