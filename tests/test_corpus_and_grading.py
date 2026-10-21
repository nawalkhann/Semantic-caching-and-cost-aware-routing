from ragroute.corpus import build_intents, build_workload
from ragroute.presets import IntentOracle, grade


def test_reference_answers_pass_and_distractors_fail():
    for intent in build_intents():
        assert grade(intent.answer, intent), intent.intent_id
        for wrong in intent.distractors:
            assert not grade(wrong, intent), (intent.intent_id, wrong)


def test_grading_matches_whole_numbers_only():
    intent = next(i for i in build_intents() if i.intent_id == "easy-refund-annual")
    assert not grade("Refunds within 300 days.", intent)


def test_workload_is_deterministic_and_resolvable():
    a, b = build_workload(300, seed=5), build_workload(300, seed=5)
    assert [x.query for x in a] == [x.query for x in b]
    oracle = IntentOracle()
    for item in a:
        oracle.register(item.query, item.intent)
        assert oracle.intent(item.query) is item.intent
