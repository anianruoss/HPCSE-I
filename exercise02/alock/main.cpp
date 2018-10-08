#include <atomic>
#include <chrono>
#include <omp.h>
#include <random>
#include <thread>

using interval_t = std::chrono::duration<double>;
const int MAX_T = 100;  // Maximum number of threads.

class ALock {
private:
    std::atomic_int tail = {0};
    volatile bool *flag = new volatile bool[MAX_T];
    int *slot = new int[MAX_T];

public:
    ALock() {
        for (int i = 0; i < MAX_T; ++i) {
            flag[i] = false;
        }
        flag[0] = true;
    }

    ~ALock() {
        delete[] flag;
        delete[] slot;
    }

    void lock(int tid) {
        slot[tid] = tail++ % MAX_T;
        while (not flag[slot[tid]]);
    }

    void unlock(int tid) {
        flag[slot[tid]] = false;
        flag[(slot[tid] + 1) % MAX_T] = true;
    }
};

/*
 * Print the thread ID and the current time.
 */
void log(int tid, const char *info, const double time) {
    printf("%1i %7s %7.6lf\n", tid, info, time);
}

/*
 * Sleep for `ms` milliseconds.
 */
void suspend(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void emulate() {
    ALock lock;

    const auto t0 = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> t1;

#pragma omp parallel private(t1)
    {
        // Begin parallel region.
        int tid = omp_get_thread_num();  // Thread ID.

        for (int i = 0; i < 5; ++i) {
            t1 = std::chrono::steady_clock::now();
            log(tid, "BEFORE",
                std::chrono::duration_cast<interval_t>(t1 - t0).count());
            lock.lock(tid);

            t1 = std::chrono::steady_clock::now();
            log(tid, "INSIDE",
                std::chrono::duration_cast<interval_t>(t1 - t0).count());
            suspend(50 + (std::rand() % (250 - 50 + 1)));
            lock.unlock(tid);

            t1 = std::chrono::steady_clock::now();
            log(tid, "AFTER",
                std::chrono::duration_cast<interval_t>(t1 - t0).count());
            suspend(50 + (std::rand() % (250 - 50 + 1)));
        }
    }
}

/*
 * Test that a lock works properly by executing some calculations.
 */
void test_alock() {
    const int N = 1000000, A[2] = {2, 3};
    int result = 0, curr = 0;
    ALock lock;

#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();  // Thread ID.
        lock.lock(tid);

        // Something not as trivial as a single ++x.
        result += A[curr = 1 - curr];

        lock.unlock(tid);
    }

    int expected = (N / 2) * A[0] + (N - N / 2) * A[1];
    if (expected == result) {
        fprintf(stderr, "Test OK!\n");
    } else {
        fprintf(stderr, "Test NOT OK: %d != %d\n", result, expected);
        exit(1);
    }
}


int main() {

    test_alock();
    emulate();

    return 0;
}
