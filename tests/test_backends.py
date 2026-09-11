from types import SimpleNamespace

import pytest

from grounded_vla.backends.knowrob import (
    KnowledgeBackendError,
    KnowRobView,
    _positive,
    atom_name,
    resolve_explicit_status,
)
from grounded_vla.backends.openpi import OpenPiPolicy, PolicyBackendError
from grounded_vla.belief import Atom, Truth
from grounded_vla.contracts import Contract
from grounded_vla.interventions import fixture
from grounded_vla.runner import RunConfig, run


class MirrorDouble:
    """Transport double only; deliberately not named or presented as KnowRob."""

    def __init__(self):
        self.facts, self.queries = {}, []

    def sync(self, snapshot):
        self.facts = {q["fact"]: Truth(q["truth"]) for q in snapshot["facts"]}

    def query(self, atom):
        self.queries.append(atom)
        return self.facts.get(atom.key, Truth.UNKNOWN)


def test_explicit_refutation_is_required():
    assert resolve_explicit_status(False, False) == Truth.UNKNOWN
    assert resolve_explicit_status(False, True) == Truth.REFUTED
    assert resolve_explicit_status(True, True) == Truth.UNKNOWN


@pytest.mark.parametrize(
    "answers,expected",
    [([(True, False)], True), ([(True, False), (False, True)], False), ([(False, True)], False)],
)
def test_native_ground_query_conflicts_do_not_establish_support(answers, expected):
    class Token:
        def __init__(self, positive=False, negative=False, control=False):
            self.positive, self.negative, self.control = positive, negative, control

        def tokenType(self):
            return 0 if self.control else 1

        def isPositive(self):
            return self.positive

        def isNegative(self):
            return self.negative

        def isUncertain(self):
            return False

    tokens = iter([Token(*answer) for answer in answers] + [Token(control=True)])
    queue = SimpleNamespace(pop_front=lambda: next(tokens))
    stream = SimpleNamespace(createQueue=lambda: queue)
    kb = SimpleNamespace(submitQuery=lambda *_: stream)
    api = SimpleNamespace(
        QueryParser=SimpleNamespace(parse=lambda value: value),
        QueryContext=lambda value: value,
        QueryFlag=SimpleNamespace(QUERY_FLAG_ALL_SOLUTIONS=1),
        TokenType=SimpleNamespace(ANSWER_TOKEN=1),
    )
    assert _positive(kb, api, "test") is expected


def test_read_through_mirror_participates_in_execution():
    mirror = MirrorDouble()
    result = run(RunConfig(scenario="disturbance"), knowrob_session=mirror)
    assert result.metrics["constrained_success"]
    assert Atom("clean", "cup_a") in mirror.queries


def test_backend_mismatch_and_errors_are_not_false():
    mirror = MirrorDouble()
    view = KnowRobView(fixture(), mirror, 0)
    mirror.facts["clean(cup_a)"] = Truth.UNKNOWN
    with pytest.raises(KnowledgeBackendError, match="Snapshot mismatch"):
        view.query(Atom("clean", "cup_a"), 0)
    with pytest.raises(KnowledgeBackendError, match="fresh"):
        view.query(Atom("clean", "cup_a"), 1)


@pytest.mark.parametrize("subject", ["cup_a);delete(x)", "cup__a", "a'b", "../cup", ""])
def test_identifiers_cannot_inject_a_query(subject):
    with pytest.raises(ValueError):
        atom_name(Atom("clean", subject))


class PolicyDouble:
    def __init__(self, actions):
        self.actions, self.calls = actions, []

    def infer(self, payload):
        self.calls.append(payload)
        return {"actions": self.actions}


def libero_inputs():
    np = pytest.importorskip("numpy")
    return {
        "model_inputs": {
            "observation/image": np.zeros((224, 224, 3), np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
            "observation/state": np.zeros(8),
        }
    }


def test_openpi_validates_inputs_and_preserves_actual_action_dimension():
    np = pytest.importorskip("numpy")
    client = PolicyDouble(np.zeros((50, 7)))
    policy = OpenPiPolicy(client, embodiment="libero", action_dim=7)
    chunks = policy.sample(Contract("place", "cup_a", "the blue cup"), libero_inputs(), 2)
    assert len(chunks) == len(client.calls) == 2
    assert len(chunks[0].values) == 50 and len(chunks[0].values[0]) == 7
    assert client.calls[0]["prompt"] == "Place the blue cup on the serving tray."
    assert chunks[0].space == "libero_action"


@pytest.mark.parametrize("shape", [(50, 32), (0, 7), (2, 50, 7), (101, 7)])
def test_openpi_rejects_wrong_shapes_instead_of_slicing(shape):
    np = pytest.importorskip("numpy")
    policy = OpenPiPolicy(PolicyDouble(np.zeros(shape)), embodiment="libero", action_dim=7)
    with pytest.raises(PolicyBackendError):
        policy.sample(Contract("pick", "cup_a", "the blue cup"), libero_inputs())


def test_serialized_token_guard_fails_before_request():
    np = pytest.importorskip("numpy")
    client = PolicyDouble(np.zeros((50, 7)))

    def reject(_):
        raise ValueError("serialized budget exceeded")

    policy = OpenPiPolicy(client, embodiment="libero", action_dim=7, serialized_input_guard=reject)
    with pytest.raises(ValueError, match="serialized budget"):
        policy.sample(Contract("pick", "cup_a", "the blue cup"), libero_inputs())
    assert not client.calls
