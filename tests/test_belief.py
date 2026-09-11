import pytest

from grounded_vla.belief import Atom, BeliefStore, Truth


def test_missing_is_unknown_and_dirty_absence_does_not_prove_clean():
    store = BeliefStore()
    assert store.query(Atom("dirty", "cup"), 0).truth == Truth.UNKNOWN
    assert store.query(Atom("clean", "cup"), 0).truth == Truth.UNKNOWN


def test_expiry_is_half_open_and_future_evidence_is_not_used():
    store, atom = BeliefStore(), Atom("clean", "cup")
    store.add(atom, True, at=2, ttl=3, source="sensor")
    assert store.query(atom, 1).truth == Truth.UNKNOWN
    assert store.query(atom, 4.999).truth == Truth.SUPPORTED
    assert store.query(atom, 5).truth == Truth.UNKNOWN


def test_conflicts_preserve_both_sources_until_explicit_replacement():
    store, atom = BeliefStore(), Atom("clean", "cup")
    first = store.add(atom, True, at=0, source="camera")
    second = store.add(atom, False, at=0, source="contact")
    answer = store.query(atom, 0)
    assert answer.truth == Truth.UNKNOWN
    assert set(answer.evidence_ids) == {first.id, second.id}
    store.add(atom, False, at=1, source="inspection", replace=True)
    assert store.query(atom, 1).truth == Truth.REFUTED
    assert len(store.archive) == 3


@pytest.mark.parametrize("kind,confidence", [("predicted", 1.0), ("observed", 0.1)])
def test_predictions_and_weak_evidence_do_not_satisfy_preconditions(kind, confidence):
    store, atom = BeliefStore(), Atom("held", "cup")
    store.add(atom, True, at=0, source="model", kind=kind, confidence=confidence)
    assert store.query(atom, 0).truth == Truth.UNKNOWN


def test_retraction_and_counterfactual_clone_do_not_modify_original():
    store, atom = BeliefStore(), Atom("clean", "cup")
    store.add(atom, True, at=0, source="sensor")
    counterfactual = store.clone()
    counterfactual.retract(atom, at=1, reason="contact")
    assert store.query(atom, 1).truth == Truth.SUPPORTED
    assert counterfactual.query(atom, 1).truth == Truth.UNKNOWN


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_truth_is_not_python_truthiness(value):
    with pytest.raises(TypeError):
        BeliefStore().add(Atom("clean", "cup"), value, at=0, source="sensor")
