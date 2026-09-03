"""A predeclared, NON-ANALYTIC VOID trace: schema, validator and writer.

A VOID leaves no report, which is right -- a partial cohort is not a result. But
D1's VOID also left NO RECORD OF HOW FAR IT GOT, and the cause could not be
settled afterwards and still cannot be. This is the narrow fix: a trace that says
how far a run reached and NOTHING ELSE.

THE STANDARD IT HAS TO MEET, and the correction that produced it
-----------------------------------------------------------------
v1 of D1's trace allowlisted FIELD NAMES and left their values arbitrary, so
`stage="move=(11,11)"` passed and the "structurally incapable of carrying a
measurement" claim was FALSE. Review caught it. Every rule below exists because
of that:

  * `event`, and every enum field, are CLOSED ENUMS -- no free text anywhere;
  * every counter is a PLAIN int (`type(x) is int`, so BOOLS are rejected)
    bounded by its OWN REAL ceiling, not merely "an int", so it cannot hold a
    payload;
  * `schema` must equal the declared constant exactly;
  * each event carries EXACTLY its declared fields -- not a superset, because an
    event that may omit a field has a shape no reader can rely on;
  * `ts` is INJECTED here, never accepted from a caller: one fewer channel.

WHY THIS IS SHARED RATHER THAN COPIED
Because it was WRONG ONCE AND WAS FIXED ONCE. A second copy for H1 would be a
second chance to reintroduce v1's name-only check, and nothing would notice until
another review. The SCHEMA differs per design; the VALIDATOR does not.

The schema is a REQUIRED argument at every call. A default would let a caller
validate H1 lines against D1's tables by omission -- the defaultable-switch
failure this workstream keeps finding.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any, Dict, Mapping, Tuple


@dataclasses.dataclass(frozen=True)
class TraceSchema:
    """WHICH trace is being written, and the complete shape it permits."""

    #: The exact value a `schema` field must carry.
    schema: str
    #: The closed set of permitted `event` values.
    events: Tuple[str, ...]
    #: event -> EXACTLY the fields it carries (a frozenset).
    event_fields: Mapping[str, frozenset]
    #: counter name -> its own inclusive ceiling.
    counter_max: Mapping[str, int]
    #: field name -> the closed enum of values it may take.
    enums: Mapping[str, Tuple[str, ...]]
    #: The exception type raised on refusal, so each design keeps its own.
    error: type
    #: Design-specific prose for the extra-field refusal, which explains what the
    #: trace would BECOME if a free-form field were admitted.
    extra_field_note: str

    @property
    def fields(self) -> frozenset:
        """The union, for readers. The per-event schema is the gate."""
        return frozenset({"event", "ts"}) | frozenset(
            f for fs in self.event_fields.values() for f in fs)


def check(schema: TraceSchema, event: Any, fields: Dict[str, Any], /) -> None:
    """Validate one trace line COMPLETELY, before anything is written.

    Names AND values.
    """
    if event not in schema.events:
        raise schema.error(f"{event!r} is not a permitted trace event; "
                           f"permitted: {list(schema.events)}")
    allowed = schema.event_fields[event]
    extra = sorted(set(fields) - allowed)
    if extra:
        raise schema.error(
            f"{extra} is not permitted on a {event!r} line. {schema.extra_field_note} "
            f"Permitted: {sorted(allowed)}")
    missing = sorted(allowed - set(fields))
    if missing:
        raise schema.error(f"a {event!r} line is missing {missing}")

    for name, value in fields.items():
        if name == "schema":
            if value != schema.schema:
                raise schema.error(
                    f"schema must be exactly {schema.schema!r}, got {value!r}")
        elif name in schema.enums:
            if value not in schema.enums[name]:
                raise schema.error(
                    f"{name} {value!r} is not one of {list(schema.enums[name])}")
        else:
            # `type(...) is int` rejects bool, which IS an int and would otherwise
            # slip a two-valued channel through a "counter".
            cap = schema.counter_max[name]
            if type(value) is not int or not 0 <= value <= cap:
                raise schema.error(
                    f"{name} must be a plain integer in [0, {cap}], got {value!r}")


def write(schema: TraceSchema, fh, /, *, event: Any = None, **fields: Any) -> None:
    """Append one validated trace line and fsync it.

    POSITIONAL-ONLY (`/`): a trace line carries a field literally named `schema`,
    which collides with this parameter's name unless it cannot be passed by
    keyword. D1's own trace tests found that on the first run.

    VALIDATED FIRST: a refusal that has already appended has not refused.

    FSYNCED PER LINE on purpose -- a trace lost when the run is terminated
    mid-stage answers nothing, and surviving exactly that is the whole point.
    """
    check(schema, event, fields)
    fh.write(json.dumps(dict(fields, event=event, ts=time.time()),
                        sort_keys=True) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
