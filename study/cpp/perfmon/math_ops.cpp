#include <math.h>
#include <chrono>
#include <cstdlib>
#include <cstdio>
using namespace std;
using namespace std::chrono;

// timer cribbed from
// https://gist.github.com/gongzhitaao/7062087
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return duration_cast<second_>
            (clock_::now() - beg_).count();
    }

private:
    typedef high_resolution_clock clock_;
    typedef duration<double, ratio<1>> second_;
    time_point<clock_> beg_;
};

int main(int argc, char** argv)
{
    double total;
    Timer tmr;
    srand(52);
    double r1 = 1000.;
    double r2 = 1000.;

#define randf() ((double) rand()) / ((double) (RAND_MAX))
#define OP_TEST(name, expr)               \
    total = 0.0; \
    tmr.reset(); \
    for ( int i = 0; i < 100000000; i++) { \
         expr; \
    }                                     \
    double name = tmr.elapsed(); \
    printf(#name); \
    printf(" %.7f\n", name );

    // time the baseline code:
    //   for loop with no extra math op
//    OP_TEST(baseline, 1.0)
    OP_TEST(baseline, r1 + r2)

    // time various floating point operations.
    //   subtracts off the baseline time to give
    //   a better approximation of the cost
    //   for just the specified operation
    OP_TEST(plus, r1 + r2)
    OP_TEST(minus, r1 - r2)
    OP_TEST(mult, r1 * r2)
    OP_TEST(div, r1 / r2)
    OP_TEST(sqrt, sqrt(r1))
    OP_TEST(sin, sin(r1))
    OP_TEST(cos, cos(r1))
    OP_TEST(tan, tan(r1))
    OP_TEST(atan, atan(r1))
    OP_TEST(exp, exp(r1))
    return 0;
}
