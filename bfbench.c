/*
 * Big float tests
 * 
 * Copyright (c) 2017 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#ifdef CONFIG_MPFR
#include <mpfr.h>
#endif

#include "libbf.h"

/* number of bits per base 10 digit */
#define BITS_PER_DIGIT 3.32192809488736234786

static bf_context_t bf_ctx;

static void *my_bf_realloc(void *opaque, void *ptr, size_t size)
{
    return realloc(ptr, size);
}

static int64_t get_clock_msec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000LL + (tv.tv_usec / 1000);
}

/* we print at least 3 significant digits with at most 5 chars, except
   if larger than 9999T. The value is rounded to zero. */
char *get_si_prefix(char *buf, int buf_size, uint64_t val)
{
    static const char suffixes[4] = "kMGT";
    uint64_t base;
    int i;

    if (val <= 999) {
        snprintf(buf, buf_size, "%" PRId64, val);
    } else {
        base = 1000;
        for(i=0;i<4;i++) {
            /* Note: we round to 0 */
            if (val < base * 10) {
                snprintf(buf, buf_size, "%0.2f%c", 
                         floor((val * 100.0) / base) / 100.0,
                         suffixes[i]);
                break;
            } else if (val < base * 100) {
                snprintf(buf, buf_size, "%0.1f%c", 
                         floor((val * 10.0) / base) / 10.0,
                         suffixes[i]);
                break;
            } else if (val < base * 1000 || (i == 3)) {
                snprintf(buf, buf_size,
                         "%" PRId64 "%c", 
                         val / base,
                         suffixes[i]);
                break;
            }
            base = base * 1000;
        }
    }
    return buf;
}

static uint64_t mp_random64(uint64_t *pseed)
{
    *pseed = *pseed * 6364136223846793005 + 1;
    return *pseed;
}

typedef enum {
    BF_OP_MUL,
    BF_OP_DIV,
    BF_OP_SQRT,

    BF_OP_COUNT,
} BFOPEnum;

const char *op_str[BF_OP_COUNT] = {
    "mul",
    "div",
    "sqrt",
};

static BFOPEnum get_op_from_str(const char *str)
{
    BFOPEnum op;
    for(op = 0; op < BF_OP_COUNT; op++) {
        if (!strcmp(str, op_str[op]))
            break;
        }
    if (op == BF_OP_COUNT) {
        fprintf(stderr, "Unknown operation: %s\n", str);
        exit(1);
    }
    return op;
}

#define K_STEPS 10

static void bf_op_speed(double k_start1, double k_end1,
                        const char *filename, int log_scale, BFOPEnum op)
{
    int k, nb_its, it, dpl, fft_len_log2, nb_mods, k_end, k_start;
    bf_t A, B, C;
    limb_t n, i, prec;
    int64_t start_time, ti, n_digits;
    FILE *f;
    double tpl, K;
    char buf1[32], buf2[32];
    uint64_t seed = 2;
    
    f = fopen(filename, "wb");
    printf("%5s %5s %5s", "K", "BITS", "DIGIT");
    if (op == BF_OP_MUL) {
        printf(" %3s %3s %2s", "FFT", "DPL", "M");
    }
    printf(" %10s %10s\n", "ms", "ns/limb");

    k_start = lrint(k_start1 * K_STEPS);
    k_end = lrint(k_end1 * K_STEPS);
    for(k = k_start; k <= k_end; k++) {
        K = (double)k / K_STEPS;
        n_digits = (int64_t)ceil(pow(10.0, K));
        n = (limb_t)ceil(n_digits * BITS_PER_DIGIT / LIMB_BITS);
        prec = n * LIMB_BITS;
        fft_len_log2 = bf_get_fft_size(&dpl, &nb_mods, 2 * n);
        printf("%5.1f %5s %5s",
               K,
               get_si_prefix(buf1, sizeof(buf1), prec),
               get_si_prefix(buf2, sizeof(buf2),
                             (int64_t)ceil(prec / BITS_PER_DIGIT)));
        if (op == BF_OP_MUL) {
            printf(" %3d %3d %2d",
                   fft_len_log2,
                   dpl,
                   nb_mods);
        }
        fflush(stdout);
        bf_init(&bf_ctx, &A);
        bf_init(&bf_ctx, &B);
        bf_init(&bf_ctx, &C);
        bf_resize(&A, n);
        bf_resize(&B, n);
        A.expn = n * LIMB_BITS;
        B.expn = n * LIMB_BITS;
        for(i = 0; i < n; i++) {
            A.tab[i] = mp_random64(&seed);
            B.tab[i] = mp_random64(&seed);
        }
        /* normalize */
        A.tab[n - 1] |= (limb_t)1 << (LIMB_BITS - 1);
        B.tab[n - 1] |= (limb_t)1 << (LIMB_BITS - 1);

        /* one multiplication to initialize the constants */
        if (fft_len_log2 <= 20) {
            bf_mul(&C, &A, &B, n, BF_RNDN);
            bf_set_ui(&C, 0);
        }
        nb_its = 1;
        for(;;) {
            start_time = get_clock_msec();
            switch(op) {
            case BF_OP_MUL:
                for(it = 0; it < nb_its; it++) {
                    bf_mul(&C, &A, &B, prec, BF_RNDN);
                }
                break;
            case BF_OP_DIV:
                for(it = 0; it < nb_its; it++) {
                    bf_div(&C, &A, &B, prec, BF_RNDF);
                }
                break;
            case BF_OP_SQRT:
                for(it = 0; it < nb_its; it++) {
                    bf_sqrt(&C, &A, prec, BF_RNDF);
                }
                break;
            default:
                break;
            }
            ti = get_clock_msec() - start_time;
            if (ti >= 100)
                break;
            nb_its *= 2;
        }
        bf_delete(&A);
        bf_delete(&B);
        bf_delete(&C);
        tpl = (double)ti / nb_its / n * 1e6;
        printf(" %10.3f %10.1f\n",
               (double)ti / nb_its,
               tpl);
        if (log_scale)
            fprintf(f, "%f %f\n", K, tpl);
        else
            fprintf(f, "%" PRIu64 " %f\n", n_digits, tpl);
        fflush(f);
    }
    fclose(f);
}

#ifdef CONFIG_MPFR

static void mpfr_mul_speed(double k_start1, double k_end1,
                           const char *filename)
{
    int k, nb_its, it, k_end, k_start;
    mpfr_t A, B, C;
    limb_t n, prec;
    int64_t start_time, ti, n_digits;
    FILE *f;
    double tpl, K;
    char buf1[32], buf2[32];
    gmp_randstate_t rnd_state;

    gmp_randinit_mt(rnd_state);
    f = fopen(filename, "wb");
    printf("%5s %5s %5s %10s %10s\n", "K", "BITS", "DIGIT",
           "ms", "ns/limb");
    k_start = lrint(k_start1 * K_STEPS);
    k_end = lrint(k_end1 * K_STEPS);
    for(k = k_start; k <= k_end; k++) {
        K = (double)k / K_STEPS;
        n_digits = (int64_t)ceil(pow(10.0, K));
        n = (limb_t)ceil(n_digits * BITS_PER_DIGIT / LIMB_BITS);
        printf("%5.1f %5s %5s",
               K,
               get_si_prefix(buf1, sizeof(buf1), n * LIMB_BITS),
               get_si_prefix(buf2, sizeof(buf2),
                             (int64_t)ceil(n * LIMB_BITS / BITS_PER_DIGIT)));
        fflush(stdout);
        prec = n * LIMB_BITS;
        mpfr_init2(A, prec);
        mpfr_init2(B, prec);
        mpfr_init2(C, prec);
        mpfr_urandomb(A, rnd_state);
        mpfr_urandomb(B, rnd_state);
        nb_its = 1;
        for(;;) {
            start_time = get_clock_msec();
            for(it = 0; it < nb_its; it++) {
                mpfr_mul(C, A, B, MPFR_RNDZ);
            }
            ti = get_clock_msec() - start_time;
            if (ti >= 100)
                break;
            nb_its *= 2;
        }
        mpfr_clear(A);
        mpfr_clear(B);
        mpfr_clear(C);
        tpl = (double)ti / nb_its / n * 1e6;
        printf(" %10.3f %10.1f\n",
               (double)ti / nb_its,
               tpl);
        fprintf(f, "%" PRIu64 " %f\n", n_digits, tpl);
        fflush(f);
    }
    fclose(f);
    gmp_randclear(rnd_state);
}

static void mpfr_bench(double k_start, double k_end,
                       const char *output_filename)
{
    FILE *f;
    const char *name;
    
    printf("LIBBF:\n");
    bf_op_speed(k_start, k_end, "/tmp/bf_mul.txt", 0, BF_OP_MUL);
    printf("MPFR:\n");
    mpfr_mul_speed(k_start, k_end, "/tmp/mpfr_mul.txt");

    f = fopen("/tmp/gnuplot.cmd", "wb");
    if (output_filename) {
        fprintf(f, "set terminal png\n"
                "set output \"%s\"\n", 
                output_filename);
    }
    fprintf(f, "set xlabel \"Number of digits\"\n");
    fprintf(f, "set ylabel \"ns/limb\"\n");
    fprintf(f, "set logscale x 10\n");
    fprintf(f, "plot ");
#ifdef __AVX2__
    name = "LIBBF(AVX2)";
#else
    name = "LIBBF";
#endif
    fprintf(f, "\"/tmp/bf_mul.txt\" with linespoints title \"%s\","
            "\"/tmp/mpfr_mul.txt\" with linespoints title \"MPFR\"\n", name);
    if (!output_filename) {
        fprintf(f, "pause -1\n");
    }
    fclose(f);

    system("gnuplot /tmp/gnuplot.cmd");
}

#endif /* CONFIG_MPFR */

int main(int argc, char **argv)
{
    const char *cmd;
    
    if (argc < 2) {
        printf("usage: bftest cmd [arguments...]\n"
               "cmd is:\n"
               "[mul|div|sqrt] [k_start] [k_end] test function on numbers of 10^k digits\n"
#ifdef CONFIG_MPFR
               "mpfr_bench [k_start] [k_end] [png_file] benchmark with MPFR\n"
#endif
               );
        exit(1);
    }
    bf_context_init(&bf_ctx, my_bf_realloc, NULL);
    cmd = argv[1];
#ifdef CONFIG_MPFR
    if (!strcmp(cmd, "mpfr_bench")) {
        double k_start, k_end;
        const char *filename;
        k_start = 4;
        if (argc > 2)
            k_start = strtod(argv[2], NULL);
        k_end = k_start;
        if (argc > 3)
            k_end = strtod(argv[3], NULL);
        filename = NULL;
        if (argc > 4)
            filename = argv[4];
        mpfr_bench(k_start, k_end, filename);
    } else
#endif
    {
        double k_start, k_end;
        BFOPEnum op;
        op = get_op_from_str(cmd);
        k_start = 4;
        if (argc > 2)
            k_start = strtod(argv[2], NULL);
        k_end = k_start;
        if (argc > 3)
            k_end = strtod(argv[3], NULL);
        bf_op_speed(k_start, k_end, "/tmp/plot.txt", 1, op);
    }
    return 0;
}
