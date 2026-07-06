// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Dharanipathi Rathna Kumar Balasubramaniam
#pragma once

#include <mach/mach_time.h>
#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// High-precision timing using mach_absolute_time (macOS)
// ---------------------------------------------------------------------------
struct TimingUtils
{
    static void init()
    {
        std::call_once(initFlag, [] {
            mach_timebase_info(&timebase);
        });
    }

    static uint64_t now() { return mach_absolute_time(); }

    static double toNanoseconds(uint64_t elapsed)
    {
        return static_cast<double>(elapsed) * timebase.numer / timebase.denom;
    }

    static double bufferDurationNs(int bufferSize, double sampleRate)
    {
        return static_cast<double>(bufferSize) / sampleRate * 1e9;
    }

private:
    static mach_timebase_info_data_t timebase;
    static std::once_flag initFlag;
};

inline mach_timebase_info_data_t TimingUtils::timebase = {0, 0};
inline std::once_flag TimingUtils::initFlag;

// ---------------------------------------------------------------------------
// Statistics computed from a vector of timing samples
// ---------------------------------------------------------------------------
struct TimingStats
{
    double median_ns = 0;
    double mean_ns = 0;
    double p95_ns = 0;
    double p99_ns = 0;
    double p999_ns = 0;
    double min_ns = 0;
    double max_ns = 0;
    double stddev_ns = 0;
    double rtf = 0;           // median / deadline
    int dropout_count = 0;     // times > deadline
    int total_samples = 0;

    // Utilization percentiles (% of deadline)
    double util_p50 = 0;
    double util_p95 = 0;
    double util_p99 = 0;
    double util_p999 = 0;
    double util_max = 0;

    static TimingStats compute(std::vector<double>& times_ns, double deadline_ns)
    {
        TimingStats s{};
        if (times_ns.empty())
            return s;

        std::sort(times_ns.begin(), times_ns.end());
        size_t n = times_ns.size();
        s.total_samples = static_cast<int>(n);

        s.min_ns    = times_ns.front();
        s.max_ns    = times_ns.back();

        // Median: average of two middle values for even N (interpolated).
        // Note: p95/p99/p999 use nearest-rank (non-interpolated). This is a
        // standard mixed approach; document in the paper methodology section.
        if (n % 2 == 1)
            s.median_ns = times_ns[n / 2];
        else
            s.median_ns = (times_ns[n / 2 - 1] + times_ns[n / 2]) / 2.0;

        // Percentiles: use ceil(n*p)-1 (nearest-rank method)
        auto pctIdx = [n](double p) -> size_t {
            size_t idx = static_cast<size_t>(std::ceil(static_cast<double>(n) * p));
            return (idx > 0 ? idx - 1 : 0);
        };
        s.p95_ns    = times_ns[std::min(pctIdx(0.95),  n - 1)];
        s.p99_ns    = times_ns[std::min(pctIdx(0.99),  n - 1)];
        s.p999_ns   = times_ns[std::min(pctIdx(0.999), n - 1)];

        double sum = std::accumulate(times_ns.begin(), times_ns.end(), 0.0);
        s.mean_ns = sum / n;

        double sq_sum = 0.0;
        for (auto t : times_ns)
        {
            double diff = t - s.mean_ns;
            sq_sum += diff * diff;
        }
        // Sample standard deviation (Bessel's correction, N-1)
        s.stddev_ns = std::sqrt(sq_sum / (n > 1 ? n - 1 : 1));

        s.rtf = s.median_ns / deadline_ns;

        s.dropout_count = 0;
        for (auto t : times_ns)
            if (t > deadline_ns)
                s.dropout_count++;

        // Utilization as % of deadline
        if (deadline_ns > 0)
        {
            s.util_p50  = (s.median_ns / deadline_ns) * 100.0;
            s.util_p95  = (s.p95_ns / deadline_ns) * 100.0;
            s.util_p99  = (s.p99_ns / deadline_ns) * 100.0;
            s.util_p999 = (s.p999_ns / deadline_ns) * 100.0;
            s.util_max  = (s.max_ns / deadline_ns) * 100.0;
        }

        return s;
    }
};

// ---------------------------------------------------------------------------
// Lock-free thread ID logger.
// Records the unique macOS thread IDs of every thread that calls record().
// Uses a generation counter so reset() causes threads to re-register.
//
// Thread model: record() is called on audio threads. reset()/getUniqueCount()
// are called from the main thread AFTER playback has stopped.
// ---------------------------------------------------------------------------
class ThreadIDLogger
{
public:
    void allocate(int maxThreads = 32)
    {
        maxSlots = maxThreads;
        threadIds = std::make_unique<std::atomic<uint64_t>[]>(static_cast<size_t>(maxThreads));
        for (int i = 0; i < maxThreads; i++)
            threadIds[static_cast<size_t>(i)].store(0, std::memory_order_relaxed);
        writeIdx.store(0, std::memory_order_relaxed);
        generation.store(0, std::memory_order_release);
    }

    void reset()
    {
        writeIdx.store(0, std::memory_order_relaxed);
        for (int i = 0; i < maxSlots; i++)
            threadIds[static_cast<size_t>(i)].store(0, std::memory_order_relaxed);
        generation.fetch_add(1, std::memory_order_release);
    }

    // Call from audio thread — lock-free. Each thread registers at most once
    // per generation (measurement run).
    void record()
    {
        thread_local int lastGen = -1;
        int curGen = generation.load(std::memory_order_acquire);
        if (lastGen == curGen) return;
        lastGen = curGen;

        int idx = writeIdx.fetch_add(1, std::memory_order_relaxed);
        if (idx < maxSlots)
        {
            uint64_t tid = 0;
            pthread_threadid_np(nullptr, &tid);
            threadIds[static_cast<size_t>(idx)].store(tid, std::memory_order_relaxed);
        }
    }

    // Call from main thread AFTER playback stops.
    int getUniqueCount() const
    {
        return std::min(writeIdx.load(std::memory_order_acquire), maxSlots);
    }

private:
    std::unique_ptr<std::atomic<uint64_t>[]> threadIds;
    std::atomic<int> writeIdx{0};
    std::atomic<int> generation{0};
    int maxSlots = 32;
};

// ---------------------------------------------------------------------------
// Lock-free timing logger for use inside audio callbacks.
// Pre-allocates storage, records mach_absolute_time pairs.
//
// Thread model: recordStart()/recordEnd() are called on the audio thread only.
// reset()/getDurationsNs()/getCount() are called from the main thread AFTER
// playback has stopped (no concurrent access).
// ---------------------------------------------------------------------------
class TimingLogger
{
public:
    TimingLogger() = default;

    void allocate(int maxEntries)
    {
        entries.resize(static_cast<size_t>(maxEntries));
        writeIdx.store(0, std::memory_order_release);
    }

    // Call from main thread ONLY when audio thread is NOT running
    void reset()
    {
        writeIdx.store(0, std::memory_order_release);
    }

    // Call from audio thread -- lock-free, single-writer
    void recordStart()
    {
        // Audio thread only — no concurrent access, relaxed is sufficient
        currentStart.store(TimingUtils::now(), std::memory_order_relaxed);
    }

    void recordEnd()
    {
        int idx = writeIdx.load(std::memory_order_relaxed);
        if (idx < static_cast<int>(entries.size()))
        {
            entries[static_cast<size_t>(idx)] = {
                currentStart.load(std::memory_order_relaxed),
                TimingUtils::now()
            };
            writeIdx.store(idx + 1, std::memory_order_release);
        }
    }

    // Record a pre-computed (start, end) pair from an external source.
    // Used by CallbackEndPlugin to store the full-callback span.
    void recordExternalPair(uint64_t start, uint64_t end)
    {
        int idx = writeIdx.load(std::memory_order_relaxed);
        if (idx < static_cast<int>(entries.size()))
        {
            entries[static_cast<size_t>(idx)] = {start, end};
            writeIdx.store(idx + 1, std::memory_order_release);
        }
    }

    // Call from main thread AFTER playback stops (acquire pairs with release in recordEnd)
    std::vector<double> getDurationsNs() const
    {
        TimingUtils::init();
        int count = writeIdx.load(std::memory_order_acquire);
        std::vector<double> durations;
        durations.reserve(static_cast<size_t>(count));

        for (int i = 0; i < count; i++)
        {
            auto& e = entries[static_cast<size_t>(i)];
            durations.push_back(TimingUtils::toNanoseconds(e.end - e.start));
        }
        return durations;
    }

    int getCount() const { return writeIdx.load(std::memory_order_acquire); }

private:
    struct Entry { uint64_t start = 0, end = 0; };
    std::vector<Entry> entries;
    std::atomic<int> writeIdx{0};
    std::atomic<uint64_t> currentStart{0};
};

// ---------------------------------------------------------------------------
// CSV output helpers
//
// Results schema v2: every row starts with schema_version,status,error_msg.
// status is "ok" for measured rows; "skipped"/"error" rows document why a
// backend x model combination produced no measurement (never a silent hole).
// Timing fields on non-ok rows are zeros and must be ignored by consumers
// (filter on status == "ok"). Documented in docs/results-schema.md and
// schemas/results.schema.json.
// ---------------------------------------------------------------------------
namespace CSVOutput
{
    constexpr int kSchemaVersion = 2;

    // Reasons land in a CSV field: strip the delimiters.
    inline std::string csvSafe(const std::string& msg)
    {
        std::string out = msg;
        for (auto& c : out)
            if (c == ',' || c == '"' || c == '\n')
                c = ';';
        return out;
    }

    inline void printIsolatedHeader(FILE* f = stdout)
    {
        fprintf(f, "schema_version,status,error_msg,"
                   "mode,backend,model,model_size,buffer_size,rep,median_ns,mean_ns,p95_ns,p99_ns,"
                   "p999_ns,min_ns,max_ns,stddev_ns,rtf,dropouts,total_samples\n");
    }

    inline void printIsolatedRow(FILE* f, const char* mode, const char* backend,
                                  const char* model, const char* modelSize,
                                  int bufSize, int rep,
                                  const TimingStats& s)
    {
        fprintf(f, "%d,ok,,%s,%s,%s,%s,%d,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,%.6f,%d,%d\n",
                kSchemaVersion, mode, backend, model, modelSize, bufSize, rep,
                s.median_ns, s.mean_ns, s.p95_ns, s.p99_ns, s.p999_ns,
                s.min_ns, s.max_ns, s.stddev_ns, s.rtf, s.dropout_count, s.total_samples);
    }

    inline void printIsolatedStatusRow(FILE* f, const char* status, const std::string& reason,
                                        const char* backend, const char* model,
                                        const char* modelSize)
    {
        fprintf(f, "%d,%s,%s,-,%s,%s,%s,0,0,0,0,0,0,0,0,0,0.0,0.000000,0,0\n",
                kSchemaVersion, status, csvSafe(reason).c_str(), backend, model, modelSize);
    }

    inline void printContentionHeader(FILE* f = stdout)
    {
        fprintf(f, "schema_version,status,error_msg,"
                   "dimension,backend,model,model_size,buffer_size,contention_level,instance_count,"
                   "rep,median_ns,mean_ns,p95_ns,p99_ns,p999_ns,min_ns,max_ns,stddev_ns,"
                   "rtf,dropouts,total_samples,"
                   "util_p50,util_p95,util_p99,util_p999,util_max,"
                   "hw_xruns,inf_underruns,thread_count\n");
    }

    inline void printContentionRow(FILE* f, const char* dimension, const char* backend,
                                    const char* model, const char* modelSize,
                                    int bufSize, int contentionLevel,
                                    int instanceCount, int rep, const TimingStats& s,
                                    int hwXruns = 0, int infUnderruns = 0, int threadCount = 0)
    {
        fprintf(f, "%d,ok,,%s,%s,%s,%s,%d,%d,%d,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.1f,"
                   "%.6f,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d\n",
                kSchemaVersion, dimension, backend, model, modelSize, bufSize, contentionLevel,
                instanceCount, rep,
                s.median_ns, s.mean_ns, s.p95_ns, s.p99_ns, s.p999_ns,
                s.min_ns, s.max_ns, s.stddev_ns, s.rtf, s.dropout_count, s.total_samples,
                s.util_p50, s.util_p95, s.util_p99, s.util_p999, s.util_max,
                hwXruns, infUnderruns, threadCount);
    }

    inline void printContentionStatusRow(FILE* f, const char* status, const std::string& reason,
                                          const char* dimension, const char* backend,
                                          const char* model, const char* modelSize,
                                          int bufSize, int contentionLevel,
                                          int instanceCount, int rep)
    {
        fprintf(f, "%d,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,"
                   "0,0,0,0,0,0,0,0.0,0.000000,0,0,0.00,0.00,0.00,0.00,0.00,0,0,0\n",
                kSchemaVersion, status, csvSafe(reason).c_str(), dimension, backend, model,
                modelSize, bufSize, contentionLevel, instanceCount, rep);
    }
}
