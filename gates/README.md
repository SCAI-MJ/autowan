# Contributed checks

A check placed here is added to the gate; it cannot replace or relax one already there.
Acceptance is the conjunction over every check, so a contribution can only ever make a
gate stricter — which is what keeps a certificate meaning the same thing between two
installations, and is the property any shared corpus of results depends on.

A check states what it measures, what it resolves against (the parent calculation, the
crystal structure, or the construction itself — never another localized representation),
and, if it is a scalar, a dimensionless fraction of a scale the material supplies
together with a record of where that fraction was calibrated. A bare constant is refused
at construction. See `autowan.gate.Check`.
