#include <algorithm>
#include <numeric>
#include <vector>
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() (0)
#endif

void topN_byrow_cpp
(
    double scores[],
    long outp[],
    long nrow,
    long ncol,
    long n,
    int nthreads
)
{
    #ifdef _OPENMP
        nthreads = (nrow < nthreads)? nrow : nthreads;
    #else
        nthreads = 1;
    #endif
    std::vector<long> ix(nthreads * ncol);
    int tid;
    for (tid = 0; tid < nthreads; tid++)
        std::iota(ix.begin() + tid*ncol, ix.begin() + (tid+1)*ncol, (long)0);

    long row;
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(ix, scores, outp, nrow, ncol, n) \
            private(tid)
    for (row = 0; row < nrow; row++)
    {
        tid = omp_get_thread_num();
        std::partial_sort(ix.begin() + tid*ncol,
                          ix.begin() + tid*ncol + n,
                          ix.begin() + (tid+1)*ncol,
                          [&scores, &ncol, &row](long a, long b)
                            {return scores[row*ncol + a] > scores[row*ncol + b];});
        std::copy(ix.begin() + tid*ncol,
                  ix.begin() + tid*ncol + n,
                  outp + row*n);
    }
} 
