# Projection and reference alignment review

The largest observed road-length difference was 27.84 m on training trip `00000028`, arc `125478` (nodes `59746 → 59741`). Its projected length was 9,299.03 m and its spherical length was 9,326.87 m, a difference of about 0.30%. The closest GPS position was 5,384.15 m from the projected segment. This arc never appeared in the saved candidates or the reference route for that trip. The audit compares arc lengths. Candidate filtering uses point-to-segment distance.

I visually inspected projected GPS points, nearby roads, the reference route, and the assigned arcs in 41-point windows around representative training cases. The gap case used 21 points because it ends at the final observation.

| Trip | Case | What the plot showed |
|---|---|---|
| `00000028` | GPS point closest to the projection outlier's midpoint | The assigned roads followed the local GPS trace through a crossing. All 41 observations passed the alignment-distance filter. |
| `00000090` | 42.5-hour gap before observation 535 | The observations on either side can be assigned nearby roads. The gap leaves the journey between them unobserved. |
| `00000042` | Disconnected reference transition near observation 6784 | GPS points follow the main road near a reference branch, while the supplied transition is disconnected. A small point-to-segment residual does not verify connectivity. |
| `00000054` | Large residual near observation 1585 | A cluster of GPS positions is assigned to a distant segment. The largest residual in the window was 259.12 m. All 41 observations in the window were excluded by the 100 m filter. |

Visual checks can reveal poor alignments but cannot establish independently verified per-point labels. In particular, nearby parallel roads, loops and repeated arcs can admit more than one plausible assignment.
