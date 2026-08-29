'''Unit tests for reranker score normalization and recency decay utilities.

Tests cover:
- normalize_reranker_scores: Batch min-max normalization to [0, 1]
- apply_threshold_with_safety_net: Threshold filtering with optional safety net
- calculate_recency_factor: Exponential decay factor from timestamp
- apply_recency_decay: Apply decay to scored results
'''

from datetime import datetime, timedelta, timezone
import math

import pytest

from reflectlog.utility.scoring import (
    apply_recency_decay,
    apply_threshold_with_safety_net,
    calculate_recency_factor,
    normalize_reranker_scores,
)


class TestNormalizeRerankerScores:
    '''Tests for normalize_reranker_scores function.'''

    def test_empty_input(self) -> None:
        '''Empty input should return empty list.'''
        result = normalize_reranker_scores([])
        assert result == []

    def test_single_result(self) -> None:
        '''Single result should get score 1.0 (best by definition).'''
        result = normalize_reranker_scores([("doc1", 0.17)])
        assert len(result) == 1
        assert result[0][0] == "doc1"
        assert result[0][1] == 1.0

    def test_two_results(self) -> None:
        '''With two results, best=1.0, worst=0.0.'''
        result = normalize_reranker_scores([("best", 0.9), ("worst", 0.1)])
        assert len(result) == 2

        # Find results by document
        best_result = next(r for r in result if r[0] == "best")
        worst_result = next(r for r in result if r[0] == "worst")

        assert best_result[1] == 1.0
        assert worst_result[1] == 0.0

    def test_cross_encoder_range(self) -> None:
        '''CrossEncoder typical range (0.001-0.17) should normalize to 0-1.'''
        scored = [
            ("doc1", 0.17),  # best
            ("doc2", 0.05),  # middle
            ("doc3", 0.001),  # worst
        ]
        result = normalize_reranker_scores(scored)

        assert len(result) == 3

        # Best should be 1.0, worst should be 0.0
        doc1_result = next(r for r in result if r[0] == "doc1")
        doc3_result = next(r for r in result if r[0] == "doc3")

        assert doc1_result[1] == 1.0
        assert doc3_result[1] == 0.0

        # Middle should be in between
        doc2_result = next(r for r in result if r[0] == "doc2")
        assert 0.0 < doc2_result[1] < 1.0

    def test_llm_range(self) -> None:
        '''LLM typical range (0.7-0.9) should normalize to 0-1.'''
        scored = [
            ("doc1", 0.90),  # best
            ("doc2", 0.85),  # middle
            ("doc3", 0.70),  # worst
        ]
        result = normalize_reranker_scores(scored)

        assert len(result) == 3

        # Best should be 1.0, worst should be 0.0
        doc1_result = next(r for r in result if r[0] == "doc1")
        doc3_result = next(r for r in result if r[0] == "doc3")

        assert doc1_result[1] == 1.0
        assert doc3_result[1] == 0.0

        # Middle should be 0.75 ((0.85 - 0.70) / (0.90 - 0.70))
        doc2_result = next(r for r in result if r[0] == "doc2")
        assert doc2_result[1] == pytest.approx(0.75)

    def test_all_equal_scores(self) -> None:
        '''All equal scores should become 0.5 (neutral midpoint with no range).'''
        scored = [
            ("doc1", 0.5),
            ("doc2", 0.5),
            ("doc3", 0.5),
        ]
        result = normalize_reranker_scores(scored)

        assert len(result) == 3
        for _doc, score in result:
            # Zero range means no differentiation, so normalized to 0.5 (neutral midpoint)
            assert score == 0.5

    def test_preserves_document_order(self) -> None:
        '''Document order should be preserved.'''
        scored = [
            ("first", 0.3),
            ("second", 0.5),
            ("third", 0.4),
        ]
        result = normalize_reranker_scores(scored)

        assert result[0][0] == "first"
        assert result[1][0] == "second"
        assert result[2][0] == "third"

    def test_negative_scores(self) -> None:
        '''Negative scores should be handled correctly.'''
        scored = [
            ("doc1", 0.5),
            ("doc2", -0.5),
        ]
        result = normalize_reranker_scores(scored)

        doc1_result = next(r for r in result if r[0] == "doc1")
        doc2_result = next(r for r in result if r[0] == "doc2")

        assert doc1_result[1] == 1.0
        assert doc2_result[1] == 0.0


class TestApplyThresholdWithSafetyNet:
    '''Tests for apply_threshold_with_safety_net function.'''

    def test_empty_input(self) -> None:
        '''Empty input should return empty list.'''
        result = apply_threshold_with_safety_net([], 0.5)
        assert result == []

    def test_all_above_threshold(self) -> None:
        '''All scores above threshold should be kept.'''
        scored = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.6)]
        result = apply_threshold_with_safety_net(scored, 0.5)
        assert len(result) == 3

    def test_some_below_threshold(self) -> None:
        '''Scores below threshold should be filtered out.'''
        scored = [("doc1", 0.9), ("doc2", 0.4), ("doc3", 0.3)]
        result = apply_threshold_with_safety_net(scored, 0.5)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_all_below_threshold_no_safety_net(self) -> None:
        '''All below threshold with min_results=0 should return empty.'''
        scored = [("doc1", 0.4), ("doc2", 0.3)]
        result = apply_threshold_with_safety_net(scored, 0.5, min_results=0)
        assert result == []

    def test_safety_net_when_enabled(self) -> None:
        '''Safety net should return min_results when all below threshold.'''
        # Results must be sorted by score descending for safety net to work
        scored = [("doc1", 0.4), ("doc2", 0.3), ("doc3", 0.2)]
        result = apply_threshold_with_safety_net(scored, 0.5, min_results=2)

        # Should return top 2 even though all below threshold
        assert len(result) == 2
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"

    def test_safety_net_partial_filter(self) -> None:
        '''Safety net should return min_results when filtered too much.'''
        # 1 passes threshold, but min_results=2
        scored = [("doc1", 0.6), ("doc2", 0.4), ("doc3", 0.3)]
        result = apply_threshold_with_safety_net(scored, 0.5, min_results=2)

        # Should return top 2 (safety net kicks in)
        assert len(result) == 2
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"

    def test_safety_net_not_enough_candidates(self) -> None:
        '''Safety net should return all available if less than min_results.'''
        scored = [("doc1", 0.4)]  # Only 1 candidate, min_results=2
        result = apply_threshold_with_safety_net(scored, 0.5, min_results=2)

        # Should return all available (1)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_threshold_zero(self) -> None:
        '''Threshold of 0.0 should keep all results.'''
        scored = [("doc1", 0.9), ("doc2", 0.1), ("doc3", 0.0)]
        result = apply_threshold_with_safety_net(scored, 0.0)
        assert len(result) == 3

    def test_threshold_one(self) -> None:
        '''Threshold of 1.0 should only keep perfect scores.'''
        scored = [("doc1", 1.0), ("doc2", 0.9)]
        result = apply_threshold_with_safety_net(scored, 1.0)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_threshold_at_boundary(self) -> None:
        '''Score equal to threshold should be kept (inclusive).'''
        scored = [("doc1", 0.5), ("doc2", 0.4)]
        result = apply_threshold_with_safety_net(scored, 0.5)
        assert len(result) == 1
        assert result[0][0] == "doc1"

    def test_min_results_zero_default(self) -> None:
        '''Default min_results=0 allows empty results.'''
        scored = [("doc1", 0.4), ("doc2", 0.3)]
        result = apply_threshold_with_safety_net(scored, 0.5)  # default min_results=0
        assert result == []


class TestIntegration:
    '''Integration tests combining normalization and threshold filtering.'''

    def test_cross_encoder_workflow(self) -> None:
        '''Simulate CrossEncoder reranking workflow.

        CrossEncoder produces low scores (0.001-0.17), which after
        normalization become 0-1, making threshold 0.5 useful.
        '''
        # Raw CrossEncoder scores (typical range)
        raw_scored = [
            ("doc1", 0.17),  # Best
            ("doc2", 0.05),  # Should be filtered
            ("doc3", 0.003),  # Should be filtered
        ]

        # Step 1: Normalize
        normalized = normalize_reranker_scores(raw_scored)

        # Step 2: Sort by score descending
        normalized.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Apply threshold
        result = apply_threshold_with_safety_net(normalized, 0.5, min_results=0)

        # Only doc1 (score=1.0) should pass threshold 0.5
        assert len(result) == 1
        assert result[0][0] == "doc1"
        assert result[0][1] == pytest.approx(1.0)

    def test_calibrated_reranker_workflow(self) -> None:
        """Simulate reranking workflow with calibrated scores.

        Calibrated scores (0.7-0.9) after normalization can filter more
        aggressively.
        """
        # Raw calibrated scores (typical range)
        raw_scored = [
            ("doc1", 0.90),  # Best -> 1.0
            ("doc2", 0.85),  # Middle -> 0.75
            ("doc3", 0.70),  # Worst -> 0.0
        ]

        # Step 1: Normalize
        normalized = normalize_reranker_scores(raw_scored)

        # Step 2: Sort by score descending
        normalized.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Apply threshold 0.5
        result = apply_threshold_with_safety_net(normalized, 0.5, min_results=0)

        # doc1 (1.0) and doc2 (0.75) pass threshold 0.5
        assert len(result) == 2
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"


class TestCalculateRecencyFactor:
    '''Tests for calculate_recency_factor function.'''

    def test_just_created_memory(self) -> None:
        '''Memory created just now should have factor ≈ 1.0.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        timestamp = now.isoformat()

        factor = calculate_recency_factor(timestamp, decay_rate=0.01, now=now)

        assert factor == pytest.approx(1.0)

    def test_two_hours_old(self) -> None:
        '''Memory 2 hours old with decay 0.01 should have factor ≈ 0.98.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(hours=2)
        timestamp = created_at.isoformat()

        factor = calculate_recency_factor(timestamp, decay_rate=0.01, now=now)

        # exp(-0.01 * 2) ≈ 0.9802
        expected = math.exp(-0.01 * 2)
        assert factor == pytest.approx(expected)

    def test_half_life_at_69_hours(self) -> None:
        '''Memory at ~69 hours old with decay 0.01 should have factor ≈ 0.5.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        # Half-life: ln(2) / 0.01 ≈ 69.3 hours
        half_life_hours = math.log(2) / 0.01
        created_at = now - timedelta(hours=half_life_hours)
        timestamp = created_at.isoformat()

        factor = calculate_recency_factor(timestamp, decay_rate=0.01, now=now)

        assert factor == pytest.approx(0.5, rel=0.01)

    def test_one_week_old(self) -> None:
        '''Memory 1 week old (168 hours) should have significant decay.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(days=7)
        timestamp = created_at.isoformat()

        factor = calculate_recency_factor(timestamp, decay_rate=0.01, now=now)

        # exp(-0.01 * 168) ≈ 0.186
        expected = math.exp(-0.01 * 168)
        assert factor == pytest.approx(expected)

    def test_higher_decay_rate(self) -> None:
        '''Higher decay rate should produce faster decay.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(hours=10)
        timestamp = created_at.isoformat()

        # With decay rate 0.05 (5x faster)
        factor = calculate_recency_factor(timestamp, decay_rate=0.05, now=now)

        # exp(-0.05 * 10) ≈ 0.606
        expected = math.exp(-0.05 * 10)
        assert factor == pytest.approx(expected)

    def test_zero_decay_rate(self) -> None:
        '''Zero decay rate should always return 1.0 (no decay).'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(days=30)
        timestamp = created_at.isoformat()

        factor = calculate_recency_factor(timestamp, decay_rate=0.0, now=now)

        assert factor == pytest.approx(1.0)

    def test_invalid_timestamp_returns_one(self) -> None:
        '''Invalid timestamp should return factor 1.0 (no decay).'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        factor = calculate_recency_factor("invalid-timestamp", decay_rate=0.01, now=now)

        assert factor == 1.0

    def test_empty_timestamp_returns_one(self) -> None:
        '''Empty timestamp should return factor 1.0 (no decay).'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        factor = calculate_recency_factor("", decay_rate=0.01, now=now)

        assert factor == 1.0

    def test_future_timestamp(self) -> None:
        '''Future timestamp should return factor 1.0 (no negative decay).'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        future = now + timedelta(hours=5)
        timestamp = future.isoformat()

        factor = calculate_recency_factor(timestamp, decay_rate=0.01, now=now)

        # max(0.0, ...) should clamp negative hours to 0
        assert factor == pytest.approx(1.0)

    def test_naive_timestamp_treated_as_utc(self) -> None:
        '''Naive timestamp (no timezone) should be treated as UTC.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(hours=2)
        # Create naive timestamp (no timezone info)
        timestamp = created_at.replace(tzinfo=None).isoformat()

        factor = calculate_recency_factor(timestamp, decay_rate=0.01, now=now)

        expected = math.exp(-0.01 * 2)
        assert factor == pytest.approx(expected)

    def test_different_timezone(self) -> None:
        '''Timestamp in different timezone should be correctly converted.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        # Create timestamp in UTC+5 (same actual time as 3:00 UTC = 8:00 UTC+5)
        # So 2 hours before now (10:00 UTC) would be 8:00 UTC which is 13:00 UTC+5
        utc_plus_5 = timezone(timedelta(hours=5))
        created_at_utc = now - timedelta(hours=2)
        created_at_local = created_at_utc.astimezone(utc_plus_5)
        timestamp = created_at_local.isoformat()

        factor = calculate_recency_factor(timestamp, decay_rate=0.01, now=now)

        # Should still be 2 hours difference
        expected = math.exp(-0.01 * 2)
        assert factor == pytest.approx(expected)


class TestApplyRecencyDecay:
    '''Tests for apply_recency_decay function.'''

    def test_empty_input(self) -> None:
        '''Empty input should return empty list.'''
        result = apply_recency_decay([], {}, decay_rate=0.01)
        assert result == []

    def test_no_timestamps_keeps_original_scores(self) -> None:
        '''Documents without timestamps should keep original scores.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        scored = [("doc1", 0.9), ("doc2", 0.8)]
        timestamp_map: dict[str, str] = {}  # Empty map

        result = apply_recency_decay(scored, timestamp_map, decay_rate=0.01, now=now)

        # Original scores preserved, sorted by score descending
        assert result[0] == ("doc1", 0.9)
        assert result[1] == ("doc2", 0.8)

    def test_decay_applied_to_old_document(self) -> None:
        '''Older document should get decayed score.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        scored = [("old_doc", 0.9), ("new_doc", 0.8)]

        # old_doc is 100 hours old, new_doc is just created
        old_time = (now - timedelta(hours=100)).isoformat()
        new_time = now.isoformat()

        timestamp_map = {
            "old_doc": old_time,
            "new_doc": new_time,
        }

        result = apply_recency_decay(scored, timestamp_map, decay_rate=0.01, now=now)

        # old_doc: 0.9 * exp(-0.01 * 100) ≈ 0.9 * 0.368 ≈ 0.331
        # new_doc: 0.8 * exp(-0.01 * 0) = 0.8 * 1.0 = 0.8
        old_factor = math.exp(-0.01 * 100)
        expected_old = 0.9 * old_factor
        expected_new = 0.8

        # Results should be re-sorted by decayed score
        # new_doc (0.8) should now rank higher than old_doc (~0.331)
        assert result[0][0] == "new_doc"
        assert result[0][1] == pytest.approx(expected_new)
        assert result[1][0] == "old_doc"
        assert result[1][1] == pytest.approx(expected_old)

    def test_reranking_after_decay(self) -> None:
        '''Decay should cause re-ranking when old doc was initially higher.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Old doc has higher initial score but will decay
        scored = [
            ("old_high_score", 0.95),  # High score but old
            ("new_lower_score", 0.7),  # Lower score but new
        ]

        # old is 200 hours old (significant decay)
        old_time = (now - timedelta(hours=200)).isoformat()
        new_time = now.isoformat()

        timestamp_map = {
            "old_high_score": old_time,
            "new_lower_score": new_time,
        }

        result = apply_recency_decay(scored, timestamp_map, decay_rate=0.01, now=now)

        # old: 0.95 * exp(-0.01 * 200) ≈ 0.95 * 0.135 ≈ 0.128
        # new: 0.7 * 1.0 = 0.7

        # New doc should now rank higher
        assert result[0][0] == "new_lower_score"
        assert result[1][0] == "old_high_score"

    def test_partial_timestamps(self) -> None:
        '''Documents with missing timestamps should keep original scores.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        scored = [
            ("doc_with_timestamp", 0.8),
            ("doc_without_timestamp", 0.7),
        ]

        timestamp = (now - timedelta(hours=50)).isoformat()
        timestamp_map = {
            "doc_with_timestamp": timestamp,
            # doc_without_timestamp not in map
        }

        result = apply_recency_decay(scored, timestamp_map, decay_rate=0.01, now=now)

        # doc_with_timestamp: 0.8 * exp(-0.01 * 50) ≈ 0.8 * 0.606 ≈ 0.485
        # doc_without_timestamp: 0.7 (unchanged)
        expected_with = 0.8 * math.exp(-0.01 * 50)

        # doc_without_timestamp should rank higher (0.7 > ~0.485)
        assert result[0][0] == "doc_without_timestamp"
        assert result[0][1] == pytest.approx(0.7)
        assert result[1][0] == "doc_with_timestamp"
        assert result[1][1] == pytest.approx(expected_with)

    def test_zero_decay_preserves_order(self) -> None:
        '''Zero decay rate should preserve original scores (but re-sort).'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        scored = [("doc1", 0.6), ("doc2", 0.9), ("doc3", 0.75)]

        old_time = (now - timedelta(days=30)).isoformat()
        timestamp_map = {
            "doc1": old_time,
            "doc2": old_time,
            "doc3": old_time,
        }

        result = apply_recency_decay(scored, timestamp_map, decay_rate=0.0, now=now)

        # No decay applied, but results are sorted by score descending
        assert result[0][0] == "doc2"
        assert result[0][1] == pytest.approx(0.9)
        assert result[1][0] == "doc3"
        assert result[1][1] == pytest.approx(0.75)
        assert result[2][0] == "doc1"
        assert result[2][1] == pytest.approx(0.6)

    def test_all_same_age(self) -> None:
        '''All documents with same age should have same decay factor.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        scored = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.8)]

        # All 10 hours old
        same_time = (now - timedelta(hours=10)).isoformat()
        timestamp_map = {
            "doc1": same_time,
            "doc2": same_time,
            "doc3": same_time,
        }

        result = apply_recency_decay(scored, timestamp_map, decay_rate=0.01, now=now)

        decay_factor = math.exp(-0.01 * 10)

        # All scores multiplied by same factor, order preserved by relative score
        assert result[0][0] == "doc1"  # Highest original
        assert result[0][1] == pytest.approx(0.9 * decay_factor)
        assert result[1][0] == "doc3"  # Second highest
        assert result[1][1] == pytest.approx(0.8 * decay_factor)
        assert result[2][0] == "doc2"  # Lowest
        assert result[2][1] == pytest.approx(0.7 * decay_factor)


class TestRecencyDecayIntegration:
    '''Integration tests combining recency decay with normalization.'''

    def test_full_workflow_with_normalization_and_decay(self) -> None:
        '''Test complete workflow: normalize -> apply decay -> threshold.'''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Raw CrossEncoder scores
        raw_scored = [
            ("recent_relevant", 0.15),  # Good relevance, recent
            ("old_best_match", 0.17),  # Best relevance, but old
            ("recent_weak", 0.05),  # Weak relevance, recent
        ]

        # Step 1: Normalize scores to [0, 1]
        normalized = normalize_reranker_scores(raw_scored)

        # Step 2: Apply recency decay
        # old_best_match is 100 hours old, others are recent
        old_time = (now - timedelta(hours=100)).isoformat()
        new_time = now.isoformat()
        timestamp_map = {
            "recent_relevant": new_time,
            "old_best_match": old_time,
            "recent_weak": new_time,
        }

        decayed = apply_recency_decay(
            normalized, timestamp_map, decay_rate=0.01, now=now
        )

        # Step 3: Apply threshold
        result = apply_threshold_with_safety_net(decayed, 0.3, min_results=0)

        # After normalization:
        # old_best_match: 1.0 (best)
        # recent_relevant: ~0.83
        # recent_weak: 0.0 (worst)

        # After decay with 100 hours (factor ~0.368):
        # old_best_match: 1.0 * 0.368 ≈ 0.368
        # recent_relevant: ~0.83 * 1.0 ≈ 0.83
        # recent_weak: 0.0 * 1.0 = 0.0

        # recent_relevant should now be highest
        assert result[0][0] == "recent_relevant"
        # old_best_match should still pass threshold 0.3
        assert len(result) == 2  # recent_weak (0.0) filtered out
        assert result[1][0] == "old_best_match"

    def test_temporal_conflict_resolution(self) -> None:
        '''Test that newer memory wins in contradictory scenario.

        This simulates the original problem: two contradictory memories
        ("I like cats" vs "I don't like cats anymore") should rank the
        newer one higher when both have similar relevance scores.
        '''
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Both have identical LLM relevance scores (the original problem)
        scored = [
            ("I like to have a blone cat.", 1.0),
            ("I don't like cat anymore, I want to have a golden retriever dog.", 1.0),
        ]

        # Old preference is 30 days old, new preference is 1 hour old
        old_time = (now - timedelta(days=30)).isoformat()
        new_time = (now - timedelta(hours=1)).isoformat()

        timestamp_map = {
            "I like to have a blone cat.": old_time,
            "I don't like cat anymore, I want to have a golden retriever dog.": new_time,
        }

        result = apply_recency_decay(scored, timestamp_map, decay_rate=0.01, now=now)

        # Newer memory should rank higher
        # Old: 1.0 * exp(-0.01 * 720) ≈ 1.0 * 0.00074 ≈ 0.00074
        # New: 1.0 * exp(-0.01 * 1) ≈ 1.0 * 0.99 ≈ 0.99
        assert "I don't like cat anymore" in result[0][0]
        assert "I like to have a blone cat" in result[1][0]

        # Significant score difference
        assert result[0][1] > result[1][1] * 10  # Newer is much higher
