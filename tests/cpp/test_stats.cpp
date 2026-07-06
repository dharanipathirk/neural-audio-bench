// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
//
// Unit tests for TimingStats::compute — the statistics that every published
// number flows through. Hand-computed expectations; any change here is a
// results-affecting change and requires re-baselining.

#include "TimingLogger.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static bool approx(double a, double b, double eps = 1e-9)
{
    return std::fabs(a - b) <= eps;
}

static void test_empty()
{
    std::vector<double> v;
    auto s = TimingStats::compute(v, 1000.0);
    assert(s.total_samples == 0);
    assert(s.median_ns == 0 && s.rtf == 0 && s.dropout_count == 0);
}

static void test_odd_n_median_exact_middle()
{
    std::vector<double> v = {30, 10, 20};  // unsorted on purpose
    auto s = TimingStats::compute(v, 100.0);
    assert(s.total_samples == 3);
    assert(approx(s.median_ns, 20.0));
    assert(approx(s.min_ns, 10.0) && approx(s.max_ns, 30.0));
    assert(approx(s.mean_ns, 20.0));
    assert(approx(s.rtf, 0.2));  // median / deadline
}

static void test_even_n_median_interpolated()
{
    std::vector<double> v = {10, 20, 30, 40};
    auto s = TimingStats::compute(v, 100.0);
    assert(approx(s.median_ns, 25.0));  // (20+30)/2
}

static void test_nearest_rank_percentiles()
{
    // 100 samples: 1..100. Nearest-rank ceil(n*p)-1 (0-indexed):
    // p95 -> idx 94 -> value 95; p99 -> idx 98 -> 99; p999 -> idx 99 -> 100.
    std::vector<double> v;
    for (int i = 1; i <= 100; i++) v.push_back(i);
    auto s = TimingStats::compute(v, 1000.0);
    assert(approx(s.p95_ns, 95.0));
    assert(approx(s.p99_ns, 99.0));
    assert(approx(s.p999_ns, 100.0));
    // Utilization percentiles: value/deadline*100
    assert(approx(s.util_p95, 9.5));
    assert(approx(s.util_max, 10.0));
}

static void test_small_n_percentiles_clamped()
{
    std::vector<double> v = {5, 10};
    auto s = TimingStats::compute(v, 100.0);
    // ceil(2*0.95)-1 = 1 -> 10; all tail percentiles land on max
    assert(approx(s.p95_ns, 10.0));
    assert(approx(s.p99_ns, 10.0));
    assert(approx(s.p999_ns, 10.0));
}

static void test_dropout_count_strictly_over_deadline()
{
    std::vector<double> v = {50, 100, 150, 200};  // deadline 100
    auto s = TimingStats::compute(v, 100.0);
    assert(s.dropout_count == 2);  // 150 and 200; exactly-on-deadline is not a dropout
}

static void test_stddev_bessel()
{
    // {2, 4, 4, 4, 5, 5, 7, 9}: mean 5, sample stddev sqrt(32/7)
    std::vector<double> v = {2, 4, 4, 4, 5, 5, 7, 9};
    auto s = TimingStats::compute(v, 1000.0);
    assert(approx(s.mean_ns, 5.0));
    assert(approx(s.stddev_ns, std::sqrt(32.0 / 7.0), 1e-9));
}

int main()
{
    test_empty();
    test_odd_n_median_exact_middle();
    test_even_n_median_interpolated();
    test_nearest_rank_percentiles();
    test_small_n_percentiles_clamped();
    test_dropout_count_strictly_over_deadline();
    test_stddev_bessel();
    printf("test_stats: all assertions passed\n");
    return 0;
}
