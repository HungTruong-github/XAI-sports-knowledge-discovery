"""Package tính toán các chỉ số tham chiếu bóng đá (Reference Metrics).

Bao gồm:
- player_stats: Số phút thi đấu, đường kiến tạo kỳ vọng (xA), key passes, assists.
- expected_threat: Mô hình mối đe dọa kỳ vọng (Expected Threat - xT) trên lưới Markov 16x12.
"""

from __future__ import annotations

from xai_football.reference.expected_threat import ExpectedThreatModel
from xai_football.reference.player_stats import (
    compute_player_minutes,
    compute_player_reference_stats,
)

__all__ = [
    "ExpectedThreatModel",
    "compute_player_minutes",
    "compute_player_reference_stats",
]
