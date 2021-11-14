/*
 * Tiny PI computation
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
#include <sys/time.h>

#include "libbf.h"

#define CHUD_A 13591409
#define CHUD_B 545140134
#define CHUD_C 640320
/* log2(C/12)*3 */
#define CHUD_BITS_PER_TERM 47.11041313821584202247

/* number of bits per base 10 digit */
#define BITS_PER_DIGIT 3.32192809488736234786

static bf_context_t bf_ctx;

static void *my_bf_realloc(void *opaque, void *ptr, size_t size)
{
    return realloc(ptr, size);
}

static void chud_bs(bf_t *P, bf_t *Q, bf_t *G, int64_t a, int64_t b, int need_g,
                    limb_t prec)
{
    int64_t c;

    if (a == (b - 1)) {
        bf_t T0, T1;
        
        bf_init(&bf_ctx, &T0);
        bf_init(&bf_ctx, &T1);
        bf_set_ui(G, 2 * b - 1);
        bf_mul_ui(G, G, 6 * b - 1, prec, BF_RNDN);
        bf_mul_ui(G, G, 6 * b - 5, prec, BF_RNDN);
        bf_set_ui(&T0, CHUD_B);
        bf_mul_ui(&T0, &T0, b, prec, BF_RNDN);
        bf_set_ui(&T1, CHUD_A);
        bf_add(&T0, &T0, &T1, prec, BF_RNDN);
        bf_mul(P, G, &T0, prec, BF_RNDN);
        P->sign = b & 1;

        bf_set_ui(Q, b);
        bf_mul_ui(Q, Q, b, prec, BF_RNDN);
        bf_mul_ui(Q, Q, b, prec, BF_RNDN);
#if LIMB_BITS == 64
        bf_mul_ui(Q, Q, (uint64_t)CHUD_C * CHUD_C * CHUD_C / 24, prec, BF_RNDN);
#else
        bf_mul_ui(Q, Q, CHUD_C, prec, BF_RNDN);
        bf_mul_ui(Q, Q, CHUD_C, prec, BF_RNDN);
        bf_mul_ui(Q, Q, CHUD_C / 24, prec, BF_RNDN);
#endif
        bf_delete(&T0);
        bf_delete(&T1);
    } else {
        bf_t P2, Q2, G2;
        
        bf_init(&bf_ctx, &P2);
        bf_init(&bf_ctx, &Q2);
        bf_init(&bf_ctx, &G2);

        c = (a + b) / 2;
        chud_bs(P, Q, G, a, c, 1, prec);
        chud_bs(&P2, &Q2, &G2, c, b, need_g, prec);
        
        /* Q = Q1 * Q2 */
        /* G = G1 * G2 */
        /* P = P1 * Q2 + P2 * G1 */
        bf_mul(&P2, &P2, G, prec, BF_RNDN);
        if (!need_g)
            bf_set_ui(G, 0);
        bf_mul(P, P, &Q2, prec, BF_RNDN);
        bf_add(P, P, &P2, prec, BF_RNDN);
        bf_delete(&P2);

        bf_mul(Q, Q, &Q2, prec, BF_RNDN);
        bf_delete(&Q2);
        if (need_g)
            bf_mul(G, G, &G2, prec, BF_RNDN);
        bf_delete(&G2);
#if 0
        printf("%" PRId64 "-%" PRId64 " limbs: P=%" PRId64 " Q=%" PRId64 " G=%" PRId64 "\n",
               a, b, P->len, Q->len, G->len);
#endif
    }
}

static int64_t time_start;
int verbose;

static int64_t get_clock_msec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000LL + (tv.tv_usec / 1000);
}

static void step_start(const char *str)
{
    if (verbose) {
        printf("%-20s", str);
        fflush(stdout);
        time_start = get_clock_msec();
    }
}

static void step_end(void)
{
    int64_t ti;
    if (verbose) {
        ti = get_clock_msec() - time_start;
        printf("(%0.3f s)\n", ti / 1000.0);
    }
}

static void pi_chud(bf_t *Q, int64_t prec)
{
    int64_t n, prec1;
    bf_t P, G;

    /* number of serie terms */
    n = (int64_t)ceil(prec / CHUD_BITS_PER_TERM) + 10;
    prec1 = prec + 32;

    bf_init(&bf_ctx, &P);
    bf_init(&bf_ctx, &G);

    step_start("chud_bs");
    chud_bs(&P, Q, &G, 0, n, 0, prec1);
    
    bf_mul_ui(&G, Q, CHUD_A, prec1, BF_RNDN);
    bf_add(&P, &G, &P, prec1, BF_RNDN);
    step_end();
    
    step_start("div");
    bf_div(Q, Q, &P, prec1, BF_RNDF);
    step_end();
 
    step_start("sqrt");
    bf_set_ui(&P, CHUD_C);
    bf_sqrt(&G, &P, prec1, BF_RNDF);
    bf_mul_ui(&G, &G, (uint64_t)CHUD_C / 12, prec1, BF_RNDF);
    step_end();

    step_start("final mul");
    bf_mul(Q, Q, &G, prec, BF_RNDN);
    step_end();
    
    bf_delete(&P);
    bf_delete(&G);
}

int main(int argc, char **argv)
{
    int64_t n_digits, prec, n_bits, ti_tot;
    bf_t PI;
    const char *output_filename;
    FILE *f;
    int arg_idx, dec_output;
    char *digits;
    size_t digits_len;
    
    dec_output = 1;
    verbose = 0;
    arg_idx = 1;
    while (arg_idx < argc) {
        if (!strcmp(argv[arg_idx], "-b")) {
            dec_output = 0;
            arg_idx++;
        } else if (!strcmp(argv[arg_idx], "-v")) {
            verbose = 1;
            arg_idx++;
        } else {
            break;
        }
    }
        
    if (arg_idx >= argc) {
        printf("usage: tinypi [options] n_digits [output_file]\n"
               "\n"
               "Options:\n"
               "-b : output in binary (hexa) instead of base 10\n"
               "-v : dump computation steps\n");
        exit(1);
    }
    
    n_digits = (int64_t)strtod(argv[arg_idx++], NULL);
    output_filename = NULL;
    if (arg_idx < argc)
        output_filename = argv[arg_idx++];
    
    ti_tot = get_clock_msec();
    n_digits = bf_max(n_digits, 50);
    n_bits = (limb_t)ceil(n_digits * BITS_PER_DIGIT);
    /* we add more bits to reduce the probability of bad rounding for
       the last digits */
    prec = n_bits + 32;
    bf_context_init(&bf_ctx, my_bf_realloc, NULL);
    bf_init(&bf_ctx, &PI);

    pi_chud(&PI, prec);

    if (dec_output) {
        step_start("base conversion");
        digits = bf_ftoa(&digits_len, &PI, 10, n_digits + 1,
                             BF_FTOA_FORMAT_FIXED | BF_RNDZ);
        step_end();
    } else {
        digits = bf_ftoa(&digits_len, &PI, 16, n_bits / 4,
                         BF_FTOA_FORMAT_FIXED | BF_RNDZ);
    }
    ti_tot = get_clock_msec() - ti_tot;
    if (verbose) {
        printf("%-20s(%0.3f s)\n", "total", ti_tot / 1000.0);
    }
    
    if (output_filename) {
        f = fopen(output_filename, "wb");
        if (!f) {
            perror(output_filename);
            exit(1);
        }
        fwrite(digits, 1, digits_len, f);
        fclose(f);
    }
    free(digits);
    bf_delete(&PI);
    bf_context_end(&bf_ctx);
    return 0;
}
