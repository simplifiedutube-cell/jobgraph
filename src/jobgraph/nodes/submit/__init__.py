"""ATS submission: vendor routing, per-vendor adapters, verification.

Safety model, in precedence order:

  1. kill switch, already-applied, daily cap  — checked before a browser opens
  2. review_mode                              — optional interrupt before the click
  3. settings.dry_run                         — fill and screenshot, click nothing
  4. manual queue                             — anything not confidently understood

Every adapter records the exact values it entered. When something goes wrong
unattended, that record plus the screenshots are how you find out what was
actually sent.
"""

from __future__ import annotations

from jobgraph.nodes.submit.router import route_and_submit, verify_submission

__all__ = ["route_and_submit", "verify_submission"]
