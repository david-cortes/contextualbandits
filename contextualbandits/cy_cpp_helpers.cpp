#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <math.h>
#include <limits.h>
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() (0)
#endif

void choice_over_rows_cpp
(
    double pred[],
    long outp[],
    long nrows, long ncols,
    int nthreads,
    unsigned int random_seed
)
{
    std::default_random_engine rng(random_seed);
    std::uniform_real_distribution<double> runif(0., 1.);
    double sum_row;
    double rnd;
    long col;
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            private(sum_row, rnd, col) shared(nrows, ncols, pred, outp)
    for (long row = 0; row < nrows; row++)
    {
        sum_row = 0;
        for (col = 0; col < ncols; col++)
            sum_row += pred[col + row*ncols];
        /* For numerical precision reasons, first standardize the row to sum to 1 */
        for (col = 0; col < ncols; col++)
            pred[col + row*ncols] /= sum_row;

        rnd = runif(rng);
        sum_row = 0;
        for (col = 0; col < ncols; col++)
        {
            sum_row += pred[col + row*ncols];
            if (sum_row >= rnd)
            {
                outp[row] = col;
                break;
            }
        }
        if (col == ncols-1)
            outp[row] = col;
    }
}

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
        nthreads = (nrow < (long)nthreads)? nrow : nthreads;
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

#define pow2(n) ( ((size_t) 1) << (n) )
#define log2ceil(x) ( (long)(ceill(log2l((long double) x))) )
#define ix_parent(ix) (((ix) - 1) / 2)
#define ix_child(ix)  (2 * (ix) + 1)
/* https://stackoverflow.com/questions/57599509/c-random-non-repeated-integers-with-weights */
void weighted_partial_shuffle
(
    long *outp, long n, long n_take, double *weights,
    std::default_random_engine &rnd_generator,
    double *buffer_arr,
    long tree_levels
)
{
    /* initialize vector with place-holders for perfectly-balanced tree */
    std::fill(buffer_arr, buffer_arr + pow2(tree_levels + 1), (double)0);

    /* compute sums for the tree leaves at each node */
    long offset = pow2(tree_levels) - 1;
    for (long ix = 0; ix < n; ix++) {
        buffer_arr[ix + offset] = weights[ix];
    }
    for (long ix = pow2(tree_levels+1) - 1; ix > 0; ix--) {
        buffer_arr[ix_parent(ix)] += buffer_arr[ix];
    }

    /* sample according to uniform distribution */
    double rnd_subrange, w_left;
    double curr_subrange;
    long curr_ix;

    for (long el = 0; el < n_take; el++)
    {
        /* go down the tree by drawing a random number and
           checking if it falls in the left or right sub-ranges */
        curr_ix = 0;
        curr_subrange = buffer_arr[0];
        for (long lev = 0; lev < tree_levels; lev++)
        {
            rnd_subrange = std::uniform_real_distribution<double>(0, curr_subrange)(rnd_generator);
            w_left = buffer_arr[ix_child(curr_ix)];
            curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
            curr_subrange = buffer_arr[curr_ix];
        }

        /* finally, add element from this iteration */
        outp[el] = curr_ix - offset;

        /* now remove the weight of the chosen element */
        buffer_arr[curr_ix] = 0;
        for (long lev = 0; lev < tree_levels; lev++)
        {
            curr_ix = ix_parent(curr_ix);
            buffer_arr[curr_ix] =   buffer_arr[ix_child(curr_ix)]
                                  + buffer_arr[ix_child(curr_ix) + 1];
        }
    }

}

void topN_softmax_cpp
(
    double scores[],
    long outp[],
    long nrow,
    long ncol,
    long n,
    int nthreads,
    unsigned long seed
)
{
    #ifdef _OPENMP
        nthreads = (nrow < (long)nthreads)? nrow : nthreads;
    #else
        nthreads = 1;
    #endif

    long row;
    std::default_random_engine rng_glob(seed);
    std::vector<std::default_random_engine> rng_row(nrow);
    for (int row = 0; row < nrow; row++)
        rng_row[row] = std::default_random_engine(
            std::uniform_int_distribution<unsigned long>(0, ULONG_MAX)(rng_glob)
            );

    long tree_levels = log2ceil(ncol);
    std::vector<double> buffer_arr_thread((long)nthreads * pow2(tree_levels + 1));

    int tid;
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(scores, outp, nrow, ncol, n, rng_row, buffer_arr_thread, tree_levels) \
            private(tid)
    for (row = 0; row < nrow; row++)
    {
        tid = omp_get_thread_num();
        weighted_partial_shuffle(
            outp + row*n, ncol, n, scores + row*ncol,
            rng_row[row],
            buffer_arr_thread.data() + (long)tid*pow2(tree_levels + 1),
            tree_levels
        );
    }
}
