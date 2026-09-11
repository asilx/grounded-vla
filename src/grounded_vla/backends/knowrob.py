"""Native KnowRob bindings behind a bounded, isolated worker process.

The evidence ledger owns timestamps/provenance; only its resolved snapshot is
mirrored. Explicit truth objects preserve open-world semantics. A negative answer
to the 'supported' edge is NOT treated as evidence for 'refuted'.
"""

from __future__ import annotations

import multiprocessing as mp
import re
from pathlib import Path

from grounded_vla.belief import Atom, BeliefStore, QueryResult, Truth

NAMESPACE = "https://example.org/grounded-vla#"


class KnowledgeBackendError(RuntimeError):
    pass


def atom_name(atom: Atom) -> str:
    for value in (atom.predicate, atom.subject):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", value) or "__" in value:
            raise ValueError(
                "KnowRob identifiers must be lowercase names without double underscores"
            )
    return f"{atom.predicate}__{atom.subject}"


def resolve_explicit_status(supported: bool, refuted: bool) -> Truth:
    if supported == refuted:
        return Truth.UNKNOWN
    return Truth.SUPPORTED if supported else Truth.REFUTED


def _positive(kb, api, query: str) -> bool:
    expression = api.QueryParser.parse(query)
    stream = kb.submitQuery(expression, api.QueryContext(api.QueryFlag.QUERY_FLAG_ALL_SOLUTIONS))
    queue = stream.createQueue()
    positive = negative = False
    for _ in range(64):
        token = queue.pop_front()
        if token.tokenType() != api.TokenType.ANSWER_TOKEN:
            return positive and not negative
        if token.isPositive() and not token.isUncertain():
            positive = True
        if token.isNegative() and not token.isUncertain():
            negative = True
    raise KnowledgeBackendError("Unexpectedly many answers to a ground query")


def _worker(connection, config_path: str) -> None:
    try:
        import knowrob as api

        kb = api.KnowledgeBase(config_path)
        mirrored: set[tuple[str, str, str]] = set()
        connection.send({"ok": True})
        while True:
            request = connection.recv()
            if request["op"] == "close":
                return
            if request["op"] == "sync":
                desired = set()
                for fact in request["snapshot"]["facts"]:
                    match = re.fullmatch(r"([a-z][a-z0-9_]*)\(([a-z][a-z0-9_]*)\)", fact["fact"])
                    if not match:
                        raise ValueError("Unsupported fact syntax")
                    name = atom_name(Atom(*match.groups()))
                    status = Truth(fact["truth"]).value
                    desired.add((NAMESPACE + name, NAMESPACE + "hasTruth", NAMESPACE + status))
                for triple in sorted(mirrored - desired):
                    if kb.removeOne(api.TripleCopy(*triple)) is False:
                        raise KnowledgeBackendError("KnowRob triple retraction failed")
                for triple in sorted(desired - mirrored):
                    if kb.insertOne(api.TripleCopy(*triple)) is False:
                        raise KnowledgeBackendError("KnowRob triple insertion failed")
                mirrored = desired
                connection.send({"ok": True})
            elif request["op"] == "query":
                name = atom_name(Atom(request["predicate"], request["subject"]))
                yes = _positive(kb, api, f"gv:hasTruth(gv:{name}, gv:supported)")
                no = _positive(kb, api, f"gv:hasTruth(gv:{name}, gv:refuted)")
                connection.send({"ok": True, "truth": resolve_explicit_status(yes, no).value})
            else:
                raise ValueError("Unknown KnowRob worker operation")
    except EOFError:
        return
    except Exception as exc:
        try:
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class KnowRobSession:
    def __init__(self, config_path: str | Path, *, timeout: float = 10.0) -> None:
        import math

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        path = Path(config_path).resolve(strict=True)
        self.timeout, self._closed = timeout, False
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, str(path)), daemon=True)
        self._process.start()
        child.close()
        self._receive()

    def _receive(self) -> dict:
        try:
            if not self._connection.poll(self.timeout):
                raise KnowledgeBackendError("KnowRob operation timed out")
            result = self._connection.recv()
            if not result.get("ok"):
                raise KnowledgeBackendError(result.get("error", "KnowRob worker failed"))
            return result
        except (EOFError, OSError, KnowledgeBackendError) as exc:
            self.close()
            raise KnowledgeBackendError(
                "KnowRob is unavailable; no truth value was inferred"
            ) from exc

    def _call(self, request: dict) -> dict:
        if self._closed:
            raise KnowledgeBackendError("KnowRob session is closed")
        try:
            self._connection.send(request)
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise KnowledgeBackendError("KnowRob transport failed") from exc
        return self._receive()

    def sync(self, snapshot: dict) -> None:
        self._call({"op": "sync", "snapshot": snapshot})

    def query(self, atom: Atom) -> Truth:
        atom_name(atom)
        return Truth(
            self._call({"op": "query", "predicate": atom.predicate, "subject": atom.subject})[
                "truth"
            ]
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=1)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class KnowRobView:
    """Queries reach KnowRob; evidence citations still refer to the local ledger.

    A mismatch fails explicitly: this reference mirror does not fabricate
    provenance for additional external inferences. Extend that boundary with
    proof/evidence imports before adding domain-specific KnowRob rules.
    """

    def __init__(self, ledger: BeliefStore, session, now: float) -> None:
        self.ledger, self.session, self.now = ledger, session, now
        self._cache: dict[Atom, QueryResult] = {}
        self.session.sync(self.ledger.snapshot(now))

    def query(self, atom: Atom, now: float) -> QueryResult:
        if now != self.now:
            raise KnowledgeBackendError("Create a fresh KnowRob view after every observation")
        if atom not in self._cache:
            local = self.ledger.query(atom, now)
            external = self.session.query(atom)
            if external != local.truth:
                raise KnowledgeBackendError(
                    f"Snapshot mismatch for {atom.key}; provenance required"
                )
            self._cache[atom] = local
        return self._cache[atom]

    def snapshot(self, now: float) -> dict:
        return self.ledger.snapshot(now)
