/*
 * Tiny arbitrary precision floating point library tests
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
#include <string.h>
#include <assert.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <gmp.h>
#include <mpfr.h>

#include "libbf.h"
#include "cutils.h"
#include "softfp.h"

typedef enum {
    BF_OP_MUL,
    BF_OP_ADD,
    BF_OP_SUB,
    BF_OP_RINT,
    BF_OP_ROUND,
    BF_OP_CMP_EQ,
    BF_OP_CMP_LT,
    BF_OP_CMP_LE,
    BF_OP_RECIP,
    BF_OP_DIV,
    BF_OP_FMOD,
    BF_OP_REM,
    BF_OP_RSQRT,
    BF_OP_SQRT,
    BF_OP_OR,
    BF_OP_XOR,
    BF_OP_AND,
    BF_OP_CAN_ROUND,
    BF_OP_MUL_L2RADIX,
    BF_OP_DIV_L2RADIX,
    BF_OP_ATOF,
    BF_OP_FTOA,
    BF_OP_EXP,
    BF_OP_LOG,
    BF_OP_COS,
    BF_OP_SIN,
    BF_OP_TAN,
    BF_OP_ATAN,
    BF_OP_ATAN2,
    BF_OP_ASIN,
    BF_OP_ACOS,
    BF_OP_POW,
    BF_OP_COUNT,
} MPFTestOPEnum;

const char *op_str[BF_OP_COUNT] = {
    "mul",
    "add",
    "sub",
    "rint",
    "round",
    "cmp_eq",
    "cmp_lt",
    "cmp_le",
    "recip",
    "div",
    "fmod",
    "rem",
    "rsqrt",
    "sqrt",
    "or",
    "xor",
    "and",
    "can_round",
    "mul_l2radix",
    "div_l2radix",
    "atof",
    "ftoa",
    "exp",
    "log",
    "cos",
    "sin",
    "tan",
    "atan",
    "atan2",
    "asin",
    "acos",
    "pow",
};

const char *rnd_str[6] = {
    "N",
    "Z",
    "D",
    "U",
    "NA",
    "F",
};

#define SPECIAL_COUNT 7

void *bf_realloc(void *ptr, size_t size)
{
    return realloc(ptr, size);
}

static void set_special(bf_t *a, int idx)
{
    switch(idx) {
    case 0:
        bf_set_ui(a, 0);
        break;
    case 1:
        bf_set_ui(a, 0);
        a->sign = 1; /* -0 */
        break;
    case 2:
        bf_set_si(a, 0);
        a->expn = BF_EXP_INF;
        break;
    case 3:
        bf_set_si(a, 0);
        a->expn = BF_EXP_INF;
        a->sign = 1;
        break;
    case 4:
        bf_set_si(a, 1);
        break;
    case 5:
        bf_set_si(a, -1);
        break;
    case 6:
        bf_set_nan(a);
        break;
    default:
        abort();
    }
}

typedef struct mp_randstate_t {
    uint64_t val;
} mp_randstate_t;

void mp_randinit(mp_randstate_t *state, uint64_t seed)
{
    state->val = seed;
}

static inline uint64_t mp_random64(mp_randstate_t *s)
{
    s->val = s->val * 6364136223846793005 + 1;
    /* avoid bad modulo properties 
       XXX: use mersenne twistter generator */
    return (s->val << 32) | (s->val >> 32);
}

/* random number between 0 and 1 with large sequences of identical bits */
static void bf_rrandom(bf_t *a, limb_t prec, mp_randstate_t *state)
{
    slimb_t n, max_run_len, cur_len, j, len, bit_index, nb_bits;
    int cur_state, m;
    
    n = (prec + LIMB_BITS - 1) / LIMB_BITS;

    /* same idea as GMP. It would be probably better to use a non
       uniform law */
    m = mp_random64(state) % 4 + 1;
    max_run_len = bf_max(prec / m, 1);
    cur_state = mp_random64(state) & 1;
    cur_len = mp_random64(state) % max_run_len + 1;
    nb_bits = n * LIMB_BITS;
    
    bf_resize(a, n);
    memset(a->tab, 0, sizeof(limb_t) * n);
    bit_index = nb_bits - prec;
    while (bit_index < nb_bits) {
        len = bf_min(cur_len, nb_bits - bit_index);
        if (cur_state) {
            /* XXX: inefficient */
            for(j = 0; j < len; j++) {
                a->tab[bit_index >> LIMB_LOG2_BITS] |= (limb_t)1 << (bit_index & (LIMB_BITS - 1));
                bit_index++;
            }
        }
        bit_index += len;
        cur_len -= len;
        if (cur_len == 0) {
            cur_len = mp_random64(state) % max_run_len + 1;
            cur_state ^= 1;
        }
    }
    a->sign = 0;
    a->expn = 0;
    bf_normalize_and_round(a, prec, BF_RNDZ);
}

static void bf_rrandom_large(bf_t *a, limb_t prec, mp_randstate_t *s)
{
    limb_t prec1;
    prec1 = mp_random64(s) % (2 * prec) + 1;
    bf_rrandom(a, prec1, s);
    a->sign = mp_random64(s) & 1;
}

/* random integer with 0 to prec bits */
static void bf_rrandom_int(bf_t *a, limb_t prec, mp_randstate_t *rnd_state)
{
    limb_t prec1;
    prec1 = mp_random64(rnd_state) % prec + 1;
    bf_rrandom(a, prec1, rnd_state);
    if (a->expn != BF_EXP_ZERO)
        a->expn += prec1;
    a->sign = mp_random64(rnd_state) & 1;
}

/* random integer with long sequences of '0' and '1' */
uint64_t rrandom_u(int len, mp_randstate_t *s)
{
    int bit, pos, n, end;
    uint64_t a;
    
    bit = mp_random64(s) & 1;
    pos = 0;
    a = 0;
    for(;;) {
        n = (mp_random64(s) % len) + 1;
        end = pos + n;
        if (end > len)
            end = len;
        if (bit) {
            n = end - pos;
            a |= ((uint64_t)(1 << n) - 1) << pos;
        }
        if (end >= len)
            break;
        pos = end;
        bit ^= 1;
    }
    return a;
}

#define F64_MANT_SIZE 52
#define F64_EXP_MASK ((1 << 11) - 1)

uint64_t rrandom_sf64(mp_randstate_t *s)
{
    uint32_t a_exp, a_sign;
    uint64_t a_mant;
    a_sign = mp_random64(s) & 1;

    /* generate exponent close to the min/max more often than random */
    switch(mp_random64(s) & 15) {
    case 0:
        a_exp = (mp_random64(s) % (2 * F64_MANT_SIZE)) & F64_EXP_MASK;
        break;
    case 1:
        a_exp = (F64_EXP_MASK - (mp_random64(s) % (2 * F64_MANT_SIZE))) & F64_EXP_MASK;
        break;
    default:
        a_exp = mp_random64(s) & F64_EXP_MASK;
        break;
    }
    a_mant = rrandom_u(F64_MANT_SIZE, s);
    return ((uint64_t)a_sign << 63) | ((uint64_t)a_exp << F64_MANT_SIZE) | a_mant;
}

static int64_t get_clock_msec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000LL + (tv.tv_usec / 1000);
}

static inline uint64_t get_cycles(void)
{
    uint32_t low,high;
    uint64_t val;
    asm volatile("rdtsc" : "=a" (low), "=d" (high));
    val = high;
    val <<= 32;
    val |= low;
    return val;
}

static mpfr_rnd_t mpfr_get_rnd_mode(bf_rnd_t rnd_mode)
{
    const mpfr_rnd_t rnd_mode_tab[] = {
        MPFR_RNDN,
        MPFR_RNDZ,
        MPFR_RNDD,
        MPFR_RNDU,
        MPFR_RNDNA,
    };
    return rnd_mode_tab[rnd_mode];
}

static void mpfr_to_bf(bf_t *r1, mpfr_t r)
{
    char *str;
    mpfr_asprintf(&str, "%Ra", r);
    //    printf("mpfr r=%s\n", str);
    assert(bf_atof(r1, str, NULL, 16, BF_PREC_INF, BF_RNDZ) == 0);
    mpfr_free_str(str);
}

static void bf_to_mpfr(mpfr_t a, const bf_t *a1)
{
    char *str;
    //    bf_print_str("a", a1);
    bf_ftoa(&str, a1, 16, BF_PREC_INF, BF_RNDZ | BF_FTOA_FORMAT_FREE |
            BF_FTOA_ADD_PREFIX);
    //    printf("mpfr a=%s\n", str);
    mpfr_set_str(a, str, 0, MPFR_RNDZ);
    free(str);
}

void mpfr_exec_init(void)
{
    slimb_t e_max, e_min;
    e_max = (limb_t)1 << (BF_EXP_BITS_MAX - 1);
    e_min = -e_max + 3;
    mpfr_set_emin(e_min);
    mpfr_set_emax(e_max);
}

int mpfr_exec_op(MPFTestOPEnum op, bf_t *r1, bf_t *a1, bf_t *b1,
                 int64_t prec, int rnd_mode1, int64_t *pcycles)
{
    mpfr_t a, b, r;
    mpfr_rnd_t rnd_mode;
    int ret, mpfr_ret;
    
    mpfr_init2(a, bf_max(a1->len, 1) * LIMB_BITS);
    mpfr_init2(b, bf_max(b1->len, 1) * LIMB_BITS);
    mpfr_init2(r, prec);

    bf_to_mpfr(a, a1);
    bf_to_mpfr(b, b1);
    
    rnd_mode = mpfr_get_rnd_mode(rnd_mode1);

    ret = 0;
    mpfr_ret = 0;
    *pcycles -= get_cycles();
    switch(op) {
    case BF_OP_MUL:
        mpfr_ret = mpfr_mul(r, a, b, rnd_mode);
        break;
    case BF_OP_ADD:
        mpfr_ret = mpfr_add(r, a, b, rnd_mode);
        break;
    case BF_OP_SUB:
        mpfr_ret = mpfr_sub(r, a, b, rnd_mode);
        break;
    case BF_OP_RINT:
        mpfr_ret = mpfr_rint(r, a, rnd_mode);
        break;
    case BF_OP_ROUND:
        mpfr_ret = mpfr_set(r, a, rnd_mode);
        break;
    case BF_OP_CMP_EQ:
        ret = mpfr_equal_p(a, b);
        break;
    case BF_OP_CMP_LT:
        ret = mpfr_less_p(a, b);
        break;
    case BF_OP_CMP_LE:
        ret = mpfr_lessequal_p(a, b);
        break;
    case BF_OP_RECIP:
        mpfr_ui_div(r, 1, a, rnd_mode);
        break;
    case BF_OP_DIV:
        mpfr_ret = mpfr_div(r, a, b, rnd_mode);
        break;
    case BF_OP_FMOD:
        mpfr_ret = mpfr_fmod(r, a, b, rnd_mode);
        break;
    case BF_OP_REM:
        mpfr_ret = mpfr_remainder(r, a, b, rnd_mode);
        break;
    case BF_OP_RSQRT:
        mpfr_rec_sqrt(r, a, rnd_mode);
        break;
    case BF_OP_SQRT:
        mpfr_ret = mpfr_sqrt(r, a, rnd_mode);
        break;
    case BF_OP_OR:
    case BF_OP_XOR:
    case BF_OP_AND:
        {
            mpz_t ai, bi;

            mpz_init(ai);
            mpz_init(bi);
            mpfr_get_z(ai, a, MPFR_RNDZ);
            mpfr_get_z(bi, b, MPFR_RNDZ);
            switch(op) {
            case BF_OP_OR:
                mpz_ior(ai, ai, bi);
                break;
            case BF_OP_XOR:
                mpz_xor(ai, ai, bi);
                break;
            case BF_OP_AND:
                mpz_and(ai, ai, bi);
                break;
            default:
                break;
            }
            mpfr_set_z(r, ai, MPFR_RNDZ);
            mpz_clear(ai);
            mpz_clear(bi);
        }
        break;
    case BF_OP_EXP:
        mpfr_ret = mpfr_exp(r, a, rnd_mode);
        break;
    case BF_OP_LOG:
        mpfr_ret = mpfr_log(r, a, rnd_mode);
        break;
    case BF_OP_COS:
        mpfr_ret = mpfr_cos(r, a, rnd_mode);
        break;
    case BF_OP_SIN:
        mpfr_ret = mpfr_sin(r, a, rnd_mode);
        break;
    case BF_OP_TAN:
        mpfr_ret = mpfr_tan(r, a, rnd_mode);
        break;
    case BF_OP_ATAN:
        mpfr_ret = mpfr_atan(r, a, rnd_mode);
        break;
    case BF_OP_ATAN2:
        mpfr_ret = mpfr_atan2(r, a, b, rnd_mode);
        break;
    case BF_OP_ASIN:
        mpfr_ret = mpfr_asin(r, a, rnd_mode);
        break;
    case BF_OP_ACOS:
        mpfr_ret = mpfr_acos(r, a, rnd_mode);
        break;
    case BF_OP_POW:
        mpfr_ret = mpfr_pow(r, a, b, rnd_mode);
        break;
    default:
        abort();
    }
    *pcycles += get_cycles();
    if (mpfr_ret != 0)
        ret |= BF_ST_INEXACT;
    mpfr_to_bf(r1, r);
    mpfr_clear(a);
    mpfr_clear(b);
    mpfr_clear(r);
    return ret;
}

int mpfr_exec_setstr(bf_t *r, const char *str, int radix,
                     int64_t prec, int rnd_mode)
{
    mpfr_t r1;
    int mpfr_ret, ret;
    mpfr_init2(r1, prec);
    mpfr_ret = mpfr_strtofr(r1, str, NULL, radix, mpfr_get_rnd_mode(rnd_mode));
    ret = 0;
    if (mpfr_ret != 0)
        ret |= BF_ST_INEXACT;
    mpfr_to_bf(r, r1);
    mpfr_clear(r1);
    return ret;
}

static int softfp_get_rnd_mode(bf_rnd_t rnd_mode)
{
    switch(rnd_mode) {
    case BF_RNDN:
        return RM_RNE;
    case BF_RNDZ:
        return RM_RTZ;
    case BF_RNDU:
        return RM_RUP;
    case BF_RNDD:
        return RM_RDN;
    case BF_RNDNA:
        return RM_RMM;
    default:
        abort();
    }
}

static int softfp_set_status(uint32_t fflags)
{
    int ret = 0;
    if (fflags & FFLAG_INVALID_OP)
        ret |= BF_ST_INVALID_OP;
    if (fflags & FFLAG_DIVIDE_ZERO)
        ret |= BF_ST_DIVIDE_ZERO;
    if (fflags & FFLAG_OVERFLOW)
        ret |= BF_ST_OVERFLOW;
    if (fflags & FFLAG_UNDERFLOW)
        ret |= BF_ST_UNDERFLOW;
    if (fflags & FFLAG_INEXACT)
        ret |= BF_ST_INEXACT;
    return ret;
}

typedef union {
    double d;
    sfloat64 u;
} Float64Union;

int softfp_exec_op(MPFTestOPEnum op, bf_t *r1, bf_t *a1, bf_t *b1,
                   limb_t prec, bf_rnd_t rnd_mode, int64_t *pcycles)
{
    sfloat64 r, a, b;
    int ret = 0;
    uint32_t fflags, rm;
    Float64Union u;
    
    *pcycles -= get_cycles();
    /* Note: the inputs must already be float64 */
    bf_get_float64(a1, &u.d, BF_RNDZ);
    //    printf("ad=%a\n", u.d);
    a = u.u;
    /* Note: the inputs must already be float64 */
    bf_get_float64(b1, &u.d, BF_RNDZ);
    //    printf("bd=%a\n", u.d);
    b = u.u;
    
    rm = softfp_get_rnd_mode(rnd_mode);
    fflags = 0;
    switch(op) {
    case BF_OP_MUL:
        r = mul_sf64(a, b, rm, &fflags);
        ret = softfp_set_status(fflags);
        break;
    case BF_OP_ADD:
        r = add_sf64(a, b, rm, &fflags);
        ret = softfp_set_status(fflags);
        break;
    case BF_OP_SUB:
        r = sub_sf64(a, b, rm, &fflags);
        ret = softfp_set_status(fflags);
        break;
    case BF_OP_CMP_EQ:
        r = 0;
        ret = eq_quiet_sf64(a, b, &fflags);
        break;
    case BF_OP_CMP_LT:
        r = 0;
        ret = lt_sf64(a, b, &fflags);
        break;
    case BF_OP_CMP_LE:
        r = 0;
        ret = le_sf64(a, b, &fflags);
        break;
    case BF_OP_DIV:
        r = div_sf64(a, b, rm, &fflags);
        ret = softfp_set_status(fflags);
        break;
    case BF_OP_SQRT:
        r = sqrt_sf64(a, rm, &fflags);
        ret = softfp_set_status(fflags);
        break;
        //    case BF_OP_RINT:
        //    case BF_OP_RECIP:
        //    case BF_OP_RSQRT:
        //    case BF_OP_OR:
        //    case BF_OP_XOR:
        //    case BF_OP_AND:
    default:
        abort();
    }
    /* Note: the inputs must already be float64 */
    u.u = r;
    //    printf("rd=%a\n", u.d);
    bf_set_float64(r1, u.d);
    *pcycles += get_cycles();
    return ret;
}

int bf_exec_op(MPFTestOPEnum op, bf_t *r, bf_t *a, bf_t *b,
               limb_t prec, bf_flags_t flags, int64_t *pcycles)
{
    int ret = 0;

    *pcycles -= get_cycles();
    switch(op) {
    case BF_OP_MUL:
        ret = bf_mul(r, a, b, prec, flags);
        break;
    case BF_OP_ADD:
        ret = bf_add(r, a, b, prec, flags);
        break;
    case BF_OP_SUB:
        ret = bf_sub(r, a, b, prec, flags);
        break;
    case BF_OP_RINT:
        bf_set(r, a);
        ret = bf_rint(r, prec, flags);
        break;
    case BF_OP_ROUND:
        bf_set(r, a);
        ret = bf_round(r, prec, flags);
        break;
    case BF_OP_CMP_EQ:
        ret = bf_cmp_eq(a, b);
        break;
    case BF_OP_CMP_LT:
        ret = bf_cmp_lt(a, b);
        break;
    case BF_OP_CMP_LE:
        ret = bf_cmp_le(a, b);
        break;
    case BF_OP_RECIP:
        bf_recip(r, a, prec);
        break;
    case BF_OP_DIV:
        ret = bf_div(r, a, b, prec, flags);
        break;
    case BF_OP_FMOD:
        ret = bf_fmod(r, a, b, prec, flags);
        break;
    case BF_OP_REM:
        ret = bf_remainder(r, a, b, prec, flags);
        break;
    case BF_OP_RSQRT:
        bf_rsqrt(r, a, prec);
        break;
    case BF_OP_SQRT:
        ret = bf_sqrt(r, a, prec, flags);
        break;
    case BF_OP_OR:
        bf_logic_or(r, a, b);
        break;
    case BF_OP_XOR:
        bf_logic_xor(r, a, b);
        break;
    case BF_OP_AND:
        bf_logic_and(r, a, b);
        break;
    case BF_OP_EXP:
        ret = bf_exp(r, a, prec, flags);
        break;
    case BF_OP_LOG:
        ret = bf_log(r, a, prec, flags);
        break;
    case BF_OP_COS:
        ret = bf_cos(r, a, prec, flags);
        break;
    case BF_OP_SIN:
        ret = bf_sin(r, a, prec, flags);
        break;
    case BF_OP_TAN:
        ret = bf_tan(r, a, prec, flags);
        break;
    case BF_OP_ATAN:
        ret = bf_atan(r, a, prec, flags);
        break;
    case BF_OP_ATAN2:
        ret = bf_atan2(r, a, b, prec, flags);
        break;
    case BF_OP_ASIN:
        ret = bf_asin(r, a, prec, flags);
        break;
    case BF_OP_ACOS:
        ret = bf_acos(r, a, prec, flags);
        break;
    case BF_OP_POW:
        ret = bf_pow(r, a, b, prec, flags);
        break;
    default:
        abort();
    }
    *pcycles += get_cycles();
    return ret;
}

void print_status(int status)
{
    printf("%c%c%c%c%c",
           (status & BF_ST_INVALID_OP) ? 'I' : '-',
           (status & BF_ST_DIVIDE_ZERO) ? 'Z' : '-',
           (status & BF_ST_OVERFLOW) ? 'O' : '-',
           (status & BF_ST_UNDERFLOW) ? 'U' : '-',
           (status & BF_ST_INEXACT) ? 'X' : '-');
}

static BOOL bf_is_same(const bf_t *a, const bf_t *b)
{
    return a->sign == b->sign && bf_cmpu(a, b) == 0;
}

void test_atof(limb_t prec, int duration_ms,
               int exp_bits, bf_rnd_t rnd_mode, int seed)
{
    DynBuf dbuf;
    int radix, it, c, e, status, ref_status, err, rnd_mode1, test_loop;
    mp_randstate_t rnd_state;
    slimb_t n_digits, prec1, i;
    char *str;
    bf_t r, r_ref;
    int64_t ti, ti_ref, nb_limbs, start_time;
    
    mp_randinit(&rnd_state, seed);

    bf_init(&r);
    bf_init(&r_ref);
    ti = 0;
    ti_ref = 0;
    start_time = get_clock_msec();
    test_loop = 1;
    it = 0;
    for(;;) {
        /* build a random string representing a number */
        if (mp_random64(&rnd_state) & 1)
            radix = (mp_random64(&rnd_state) % 35) + 2;
        else
            radix = 10;
        prec1 = (limb_t)ceil(prec / log2(radix));
        n_digits = mp_random64(&rnd_state) % (prec1 * 3) + 1;
        dbuf_init(&dbuf);
        if (mp_random64(&rnd_state) & 1)
            dbuf_putc(&dbuf, '-');

        for(i = 0; i < n_digits; i++) {
            c = mp_random64(&rnd_state) % radix;
            if (c < 10)
                c += '0';
            else
                c += 'a' - 10;
            dbuf_putc(&dbuf, c);
        }
        if (radix == 10)
            dbuf_putc(&dbuf, 'e');
        else
            dbuf_putc(&dbuf, '@');
        e = prec1 * 20;
        e = (mp_random64(&rnd_state) % (2 * e + 1)) - e;
        dbuf_printf(&dbuf, "%d", e);
        dbuf_putc(&dbuf, '\0');
        str = (char *)dbuf.buf;

        ti -= get_cycles();
        status = bf_atof(&r, str, NULL, radix, prec, rnd_mode) &
            BF_ST_INEXACT;
        ti += get_cycles();
        rnd_mode1 = rnd_mode;
        if (rnd_mode == BF_RNDF)
            rnd_mode1 = BF_RNDD;

        ti_ref -= get_cycles();
        ref_status = mpfr_exec_setstr(&r_ref, str, radix, prec, rnd_mode1);
        ti_ref += get_cycles();
        
        if (rnd_mode == BF_RNDF) {
            err = !bf_is_same(&r, &r_ref);
            if (err && rnd_mode == BF_RNDF) {
                ref_status = mpfr_exec_setstr(&r_ref, str, radix, prec, BF_RNDU);
                err = !bf_is_same(&r, &r_ref);
            }
        } else {
            err = !bf_is_same(&r, &r_ref) || status != ref_status;
        }
        
        if (err) {
            printf("\nERROR (%d):\n", it);
            printf("radix=%d\n", radix);
            printf("str=%s\n", str);
            bf_print_str("r  ", &r);
            bf_print_str("ref", &r_ref);
            printf("st    ="); print_status(status); printf("\n");
            printf("ref_st="); print_status(ref_status); printf("\n");
            exit(1);
        }
        free(str);
        it++;
        if ((it & (test_loop - 1)) == 0) {
            if ((get_clock_msec() - start_time) >= duration_ms)
                break;
            test_loop *= 2;
        }
    }
    bf_delete(&r);
    bf_delete(&r_ref);

    nb_limbs = (prec + 63) / 64;
    printf(" %8u %8.1f %8.1f\n",
           it,
           (double)ti / it / nb_limbs,
           (double)ti_ref / it / nb_limbs);
}

void test_ftoa(limb_t prec, int duration_ms,
               int exp_bits, bf_rnd_t rnd_mode, int seed)
{
    int radix, it, e, test_loop;
    mp_randstate_t rnd_state;
    slimb_t n_digits, prec1, nb_limbs;
    char *r_str, *r_ref_str;
    bf_t a;
    int64_t ti, ti_ref, start_time;
    
    mp_randinit(&rnd_state, seed);
    bf_init(&a);
    ti_ref = 0;
    ti = 0;
    start_time = get_clock_msec();
    test_loop = 1;
    it = 0;
    for(;;) {
        /* build a random string representing a number */
        if ((mp_random64(&rnd_state) & 1) && 0)
            radix = (mp_random64(&rnd_state) % 35) + 2;
        else
            radix = 10;
        n_digits = (limb_t)ceil(prec / log2(radix));
        prec1 = mp_random64(&rnd_state) % (3 * prec) + 2;
        bf_rrandom(&a, prec1, &rnd_state);
        e = prec * 20;
        if (a.expn != BF_EXP_ZERO)
            a.expn += (mp_random64(&rnd_state) % (2 * e + 1)) - e;
        ti -= get_cycles();
        bf_ftoa(&r_str, &a, radix, n_digits, rnd_mode |
                BF_FTOA_FORMAT_FIXED | BF_FTOA_FORCE_EXP);
        ti += get_cycles();
        {
            mpfr_t a1;
            mpfr_exp_t expn;
            DynBuf s_s, *s = &s_s;
            char *str, *p;
            slimb_t i;
            
            mpfr_init2(a1, bf_max(a.len, 1) * LIMB_BITS);
            bf_to_mpfr(a1, &a);
            ti_ref -= get_cycles();
            str = mpfr_get_str(NULL, &expn, radix, n_digits, a1,
                               mpfr_get_rnd_mode(rnd_mode));
            ti_ref += get_cycles();
            /* add the decimal point and exponent */
            dbuf_init(s);
            p = str;
            if (*p == '-')
                dbuf_putc(s, *p++);
            dbuf_putc(s, *p++);
            if (n_digits > 1) {
                dbuf_putc(s, '.');
                for(i = 1; i < n_digits; i++) {
                    dbuf_putc(s, *p++);
                }
            }
            expn--;
            if ((radix & (radix - 1)) == 0 && radix <= 16) {
                int radix_bits = 1;
                while ((1 << radix_bits) != radix)
                    radix_bits++;
                dbuf_printf(s, "p%" PRId64 , (int64_t)(expn * radix_bits));
            } else {
                dbuf_printf(s, "%c%" PRId64 , radix <= 10 ? 'e' : '@', (int64_t)expn);
            }
            dbuf_putc(s, '\0');
            
            r_ref_str = (char *)s->buf;
            mpfr_clear(a1);
            mpfr_free_str(str);
        }
        
        if (strcmp(r_ref_str, r_str) != 0) {
            printf("\nERROR (%d):\n", it);
            printf("radix=%d\n", radix);
            bf_print_str("a  ", &a);
            printf("r  =%s\n", r_str);
            printf("ref=%s\n", r_ref_str);
            exit(1);
        }
        free(r_str);
        free(r_ref_str);
        it++;
        if ((it & (test_loop - 1)) == 0) {
            if ((get_clock_msec() - start_time) >= duration_ms)
                break;
            test_loop *= 2;
        }
    }
    bf_delete(&a);

    nb_limbs = (prec + 63) / 64;
    printf(" %8u %8.1f %8.1f\n",
           it,
           (double)ti / it / nb_limbs,
           (double)ti_ref / it / nb_limbs);
}

void test_can_round(limb_t prec, int duration_ms, bf_rnd_t rnd_mode, int seed)
{
    mp_randstate_t rnd_state;
    bf_t a, b, a_rounded, c;
    limb_t prec1, k;
    int res, it, i, res1, test_loop;
    int64_t start_time;
    
    mp_randinit(&rnd_state, seed);
    bf_init(&a);
    bf_init(&a_rounded);
    bf_init(&b);
    bf_init(&c);
    start_time = get_clock_msec();
    test_loop = 1;
    it = 0;
    for(;;) {
        prec1 = mp_random64(&rnd_state) % (3 * prec) + 2;
        bf_rrandom(&a, prec1, &rnd_state);
        a.sign = mp_random64(&rnd_state) & 1;

        k = prec + (mp_random64(&rnd_state) % 10);
        bf_set(&a_rounded, &a);
        bf_round(&a_rounded, prec, rnd_mode);
        res = bf_can_round(&a, prec, rnd_mode, k);
        if (res) {
            for(i = 0; i < 100; i++) {
                bf_rrandom(&c, prec1, &rnd_state);
                c.sign = mp_random64(&rnd_state) & 1;
                if (c.expn != BF_EXP_ZERO)
                    c.expn += a.expn - k;
                
                bf_add(&b, &a, &c, BF_PREC_INF, BF_RNDZ);
                bf_round(&b, prec, rnd_mode);
                res1 = !bf_is_same(&b, &a_rounded);
                if (res1) {
                    printf("\nERROR (%d):\n", it);
                    printf("k=%" PRId64 "\n", (int64_t)k);
                    bf_print_str("a    ", &a);
                    bf_print_str("a_rnd", &a_rounded);
                    bf_print_str("e    ", &c);
                    bf_print_str("b    ", &b);
                    exit(1);
                }
            }
        }
        it++;
        if ((it & (test_loop - 1)) == 0) {
            if ((get_clock_msec() - start_time) >= duration_ms)
                break;
            test_loop *= 2;
        }
    }
    bf_delete(&a);
    bf_delete(&a_rounded);
    bf_delete(&b);
    bf_delete(&c);
    printf(" %8u\n", it);
}

void test_mul_log2(int duration_ms, BOOL is_inv, BOOL is_ceil, int seed)
{
    mp_randstate_t rnd_state;
    int it, radix, err, test_loop;
    slimb_t a, v_max, r, r_ref, prec, d;
    mpfr_t a1, log2_radix[BF_RADIX_MAX - 1];
    int64_t start_time;
    
    mp_randinit(&rnd_state, seed);
    prec = 256;
    mpfr_init2(a1, prec);

    for(radix = 2; radix <= BF_RADIX_MAX; radix++) {
        mpfr_init2(log2_radix[radix - 2], prec);
        mpfr_set_ui(a1, radix, MPFR_RNDN);
        mpfr_log2(log2_radix[radix - 2], a1, MPFR_RNDN);
    }

    if (is_inv)
        v_max = BF_PREC_MAX;
    else
        v_max = BF_PREC_MAX / 6;
    start_time = get_clock_msec();
    test_loop = 1;
    it = 0;
    for(;;) {
        for(radix = 2; radix <= BF_RADIX_MAX; radix++) {
            a = (mp_random64(&rnd_state) % (2 * v_max + 1)) - v_max;
            r = bf_mul_log2_radix(a, radix, is_inv, is_ceil);
            
            mpfr_set_si(a1, a, MPFR_RNDN);
            if (is_inv)
                mpfr_div(a1, a1, log2_radix[radix - 2], MPFR_RNDN);
            else
                mpfr_mul(a1, a1, log2_radix[radix - 2], MPFR_RNDN);
            if (is_ceil)
                mpfr_ceil(a1, a1);
            else
                mpfr_floor(a1, a1);
            r_ref = mpfr_get_si(a1, MPFR_RNDN);
            if (is_inv) {
                err = (r != r_ref);
            } else {
                d = r - r_ref;
                err = (d > 1 || d < -1);
            }
            if (err) {
                printf("\nERROR (%d):\n", it);
                printf("a=%" PRId64 " radix=%d inv=%d ceil=%d res=%" PRId64 " ref=%" PRId64 "\n",
                       (int64_t)a, radix, is_inv, is_ceil,
                       (int64_t)r, (int64_t)r_ref);
                exit(1);
            }
        }
        it++;
        if ((it & (test_loop - 1)) == 0) {
            if ((get_clock_msec() - start_time) >= duration_ms)
                break;
            test_loop *= 2;
        }
    }

    for(radix = 2; radix <= BF_RADIX_MAX; radix++)
        mpfr_clear(log2_radix[radix - 2]);
    mpfr_clear(a1);
    printf(" %8u\n", it);
}

void test_op_rm(MPFTestOPEnum op, limb_t rprec, int duration_ms,
                int exp_bits, bf_rnd_t rnd_mode, int seed)
{
    bf_t a, b, r, r_ref;
    int op_count, status, ref_status, test_loop, it;
    int  nb_limbs;
    int64_t ti, ti_ref, ti_dummy;
    mp_randstate_t rnd_state;
    BOOL res, use_float64_ref;
    bf_rnd_t rnd_mode1;
    bf_flags_t bf_flags;
    int64_t start_time;
    limb_t prec;
    
    printf("%-20s %5d %3d %3s %5d", op_str[op], (int)rprec, exp_bits,
           rnd_str[rnd_mode], seed);
    fflush(stdout);
    
    switch(op) {
    case BF_OP_ATOF:
        test_atof(rprec, duration_ms, exp_bits, rnd_mode, seed);
        return;
    case BF_OP_FTOA:
        test_ftoa(rprec, duration_ms, exp_bits, rnd_mode, seed);
        return;
    case BF_OP_CAN_ROUND:
        test_can_round(rprec, duration_ms, rnd_mode, seed);
        return;
    case BF_OP_MUL_L2RADIX:
    case BF_OP_DIV_L2RADIX:
        test_mul_log2(duration_ms, (op == BF_OP_DIV_L2RADIX), rnd_mode == BF_RNDU, seed);
        return;
    default:
        break;
    }
    
    use_float64_ref = (rprec == 53 && exp_bits == 11);
    bf_flags = rnd_mode | bf_set_exp_bits(exp_bits);
    if (use_float64_ref)
        bf_flags |= BF_FLAG_SUBNORMAL;
    
    mp_randinit(&rnd_state, seed);
    bf_init(&a);
    bf_init(&b);
    bf_init(&r);
    bf_init(&r_ref);
    bf_set_ui(&b, 0);
    bf_set_ui(&r, 0);
    bf_set_ui(&r_ref, 0);
    ti = 0;
    ti_ref = 0;
    ti_dummy = 0;
    start_time = get_clock_msec();
    test_loop = 1;
    it = 0;
    for(;;) {
        if (rprec == 0) {
            prec = (mp_random64(&rnd_state) % 1000) + 24;
        } else {
            prec = rprec;
        }
        switch(op) {
        case BF_OP_RINT:
        case BF_OP_RECIP:
        case BF_OP_RSQRT:
        case BF_OP_SQRT:
        case BF_OP_EXP:
        case BF_OP_LOG:
        case BF_OP_COS:
        case BF_OP_SIN:
        case BF_OP_TAN:
        case BF_OP_ATAN:
        case BF_OP_ASIN:
        case BF_OP_ACOS:
            op_count = 1;
            break;
        default:
            op_count = 2;
            break;
        }

        if (op_count == 1) {
            if (it < SPECIAL_COUNT) {
                set_special(&a, it);
            } else {
                limb_t prec1;
                
                if (use_float64_ref) {
                    Float64Union u;
                    u.u = rrandom_sf64(&rnd_state);
                    bf_set_float64(&a, u.d);
                } else {
                    prec1 = mp_random64(&rnd_state) % (3 * prec) + 1;
                    bf_rrandom(&a, prec1, &rnd_state);
                    if (op == BF_OP_COS || op == BF_OP_SIN || op == BF_OP_TAN) {
                        int k;
                        bf_t c_s, *c = &c_s;
                        if (a.expn != BF_EXP_ZERO)
                            a.expn++;
                        k = (mp_random64(&rnd_state) % 2000) - 1000;
                        bf_init(c);
                        bf_const_pi(c, prec1 + 1, BF_RNDN);
                        c->expn--; /* pi/2 */
                        bf_mul_si(c, c, k, prec1 + 1, BF_RNDN);
                        bf_add(&a, &a, c, prec1, BF_RNDN);
                        bf_delete(c);
                    } else if (op == BF_OP_ACOS || op == BF_OP_ASIN) {
                    } else {
                        if (a.expn != BF_EXP_ZERO)
                            a.expn += prec1 / 2;
                    }
                }
                if (op == BF_OP_RSQRT || op == BF_OP_SQRT || op == BF_OP_LOG) {
                    a.sign = 0;
                } else {
                    a.sign = mp_random64(&rnd_state) & 1;
                }
            }
        } else if (op == BF_OP_OR ||
                   op == BF_OP_XOR ||
                   op == BF_OP_AND) {
            bf_rrandom_int(&a, prec, &rnd_state);
            bf_rrandom_int(&b, prec, &rnd_state);
        } else {
            if (it < SPECIAL_COUNT * SPECIAL_COUNT) {
                set_special(&a, it % SPECIAL_COUNT);
                set_special(&b, it / SPECIAL_COUNT);
            } else {
                if (op == BF_OP_POW) {
                    bf_rrandom_large(&a, prec, &rnd_state);
                    if ((it % 10) == 0) {
                        bf_set_si(&b, (int32_t)mp_random64(&rnd_state));
                    } else {
                        bf_rrandom_large(&b, prec, &rnd_state);
                    }
                } else if (use_float64_ref) {
                    Float64Union u;
                    u.u = rrandom_sf64(&rnd_state);
                    bf_set_float64(&a, u.d);
                    u.u = rrandom_sf64(&rnd_state);
                    bf_set_float64(&b, u.d);
                } else {
                    bf_rrandom_large(&a, prec, &rnd_state);
                    bf_rrandom_large(&b, prec, &rnd_state);
                }
            }
        }
        
        status = bf_exec_op(op, &r, &a, &b, prec, bf_flags, &ti);
        //        bf_print_str("r", &r);
        
        rnd_mode1 = rnd_mode;
        if (rnd_mode == BF_RNDF)
            rnd_mode1 = BF_RNDD;
        if (use_float64_ref) {
            ref_status = softfp_exec_op(op, &r_ref, &a, &b, prec, rnd_mode1, &ti_ref);
        } else {
            ref_status = mpfr_exec_op(op, &r_ref, &a, &b, prec, rnd_mode1, &ti_ref);
        }
        //        bf_print_str("r_ref", &r_ref);

        if (op == BF_OP_CMP_EQ ||
            op == BF_OP_CMP_LE ||
            op == BF_OP_CMP_LT) {
            res = (status != ref_status);
        } else {
            res = !bf_is_same(&r, &r_ref);
            if (rnd_mode == BF_RNDF) {
                if (res) {
                    if (use_float64_ref) {
                        softfp_exec_op(op, &r_ref, &a, &b, prec, BF_RNDU, &ti_dummy);
                    } else {
                        mpfr_exec_op(op, &r_ref, &a, &b, prec, BF_RNDU, &ti_dummy);
                    }
                    res = !bf_is_same(&r, &r_ref);
                }
            } else {
                if ((status & BF_ST_INEXACT) !=
                    (ref_status & BF_ST_INEXACT))
                    res = 1;
            }
        }
        
        if (res) {
            printf("\nERROR (%d):\n", it);
            
            bf_print_str("a  ", &a);
            if (op_count > 1) {
                bf_print_str("b  ", &b);
            }
            bf_print_str("r  ", &r);
            bf_print_str("ref", &r_ref);
            printf("st    ="); print_status(status); printf("\n");
            printf("ref_st="); print_status(ref_status); printf("\n");
            exit(1);
        }
        it++;
        if ((it & (test_loop - 1)) == 0) {
            if ((get_clock_msec() - start_time) >= duration_ms)
                break;
            test_loop *= 2;
        }
    }

    nb_limbs = (prec + 63) / 64;
    printf(" %8u %8.1f %8.1f\n",
           it,
           (double)ti / it / nb_limbs,
           (double)ti_ref / it / nb_limbs);

    bf_delete(&a);
    bf_delete(&b);
    bf_delete(&r);
    bf_delete(&r_ref);
}

void test_op(MPFTestOPEnum op, limb_t prec, int duration_ms, int exp_bits,
             int seed)
{
    BOOL use_float64_ref;
    uint8_t rm_allowed[BF_RNDF + 1];
    bf_rnd_t rnd_mode;

    use_float64_ref = (prec == 53 && exp_bits == 11);
    memset(rm_allowed, 0, sizeof(rm_allowed));
    if (use_float64_ref) {
        rm_allowed[BF_RNDN] = 1;
        rm_allowed[BF_RNDZ] = 1;
        rm_allowed[BF_RNDU] = 1;
        rm_allowed[BF_RNDD] = 1;
        rm_allowed[BF_RNDNA] = 1;
    } else {
        switch(op) {
        case BF_OP_ADD:
        case BF_OP_MUL:
        case BF_OP_DIV:
        case BF_OP_FMOD:
        case BF_OP_REM:
        case BF_OP_RINT:
        case BF_OP_ROUND:
        case BF_OP_SQRT:
        case BF_OP_ATOF:
        case BF_OP_EXP:
        case BF_OP_LOG:
        case BF_OP_COS:
        case BF_OP_SIN:
        case BF_OP_TAN:
        case BF_OP_ATAN:
        case BF_OP_ATAN2:
        case BF_OP_ASIN:
        case BF_OP_ACOS:
        case BF_OP_POW:
            rm_allowed[BF_RNDN] = 1;
            rm_allowed[BF_RNDZ] = 1;
            rm_allowed[BF_RNDU] = 1;
            rm_allowed[BF_RNDD] = 1;
            rm_allowed[BF_RNDF] = 1;
            break;
        case BF_OP_CAN_ROUND:
            rm_allowed[BF_RNDN] = 1;
            rm_allowed[BF_RNDZ] = 1;
            rm_allowed[BF_RNDU] = 1;
            rm_allowed[BF_RNDD] = 1;
            rm_allowed[BF_RNDNA] = 1;
            break;
        case BF_OP_FTOA:
            rm_allowed[BF_RNDN] = 1;
            rm_allowed[BF_RNDZ] = 1;
            rm_allowed[BF_RNDU] = 1;
            rm_allowed[BF_RNDD] = 1;
            break;
        case BF_OP_SUB:
            /* minimal test for SUB which is like ADD */
            rm_allowed[BF_RNDN] = 1;
            break;
        case BF_OP_RECIP:
        case BF_OP_RSQRT:
            rm_allowed[BF_RNDF] = 1;
            break;
        case BF_OP_MUL_L2RADIX:
        case BF_OP_DIV_L2RADIX:
            rm_allowed[BF_RNDU] = 1;
            rm_allowed[BF_RNDD] = 1;
            break;
        default:
            rm_allowed[BF_RNDZ] = 1;
            break;
        }
    }
    for(rnd_mode = 0; rnd_mode < countof(rm_allowed); rnd_mode++) {
        if (rm_allowed[rnd_mode]) {
            test_op_rm(op, prec, duration_ms, exp_bits, rnd_mode, seed);
        }
    }
}

static MPFTestOPEnum get_op_from_str(const char *str)
{
    MPFTestOPEnum op;
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

void help(void)
{
    printf("usage: bftest [options] [first_op [last_op]]\n"
           "\n"
           "Options:\n"
           "-h         this help\n"
           "-s seed    set the initial seed\n");
    exit(1);
}

int main(int argc, char **argv)
{
    int seed, duration_ms, c;
    MPFTestOPEnum op, op_start, op_last;
                        
    seed = 1234;
    duration_ms = 100;
    for(;;) {
        c = getopt(argc, argv, "hs:");
        if (c == -1)
            break;
        switch(c) {
        case 'h':
            help();
        case 's':
            seed = strtoul(optarg, NULL, 0);
            duration_ms = 1000;
            break;
        default:
            exit(1);
        }
    }

    op_start = 0;
    op_last = BF_OP_COUNT - 1;
    if (optind < argc)
        op_start = get_op_from_str(argv[optind++]);
    if (optind < argc)
        op_last = get_op_from_str(argv[optind++]);

    mpfr_exec_init();
    
    printf("%-20s %5s %3s %3s %5s %8s %8s %8s\n", "OP", "PREC", "EXP", "RND", "SEED", "CNT", "c/64bit", "ref");
    for(;;) {
        for(op = op_start; op <= op_last; op++) {
            if (op == BF_OP_MUL_L2RADIX || op == BF_OP_DIV_L2RADIX) {
                test_op(op, LIMB_BITS, duration_ms, 0, seed);
            } else if (op == BF_OP_CAN_ROUND) {
                test_op(op, 8, duration_ms, BF_EXP_BITS_MAX, seed);
                test_op(op, 53, duration_ms, BF_EXP_BITS_MAX, seed);
                test_op(op, 256, duration_ms, BF_EXP_BITS_MAX, seed);
            } else {
                if (op == BF_OP_MUL ||
                    op == BF_OP_ADD ||
                    op == BF_OP_DIV ||
                    op == BF_OP_SQRT ||
                    op == BF_OP_CMP_EQ ||
                    op == BF_OP_CMP_LT ||
                    op == BF_OP_CMP_LE) {
                    test_op(op, 53, duration_ms, 11, seed);
                }
                test_op(op, 53, duration_ms, BF_EXP_BITS_MAX, seed);
                test_op(op, 112, duration_ms, BF_EXP_BITS_MAX, seed);
                /* mpfr bug ? */
                if (op !=  BF_OP_SQRT)
                    test_op(op, 256, duration_ms, BF_EXP_BITS_MAX, seed);
                test_op(op, 3000, duration_ms, BF_EXP_BITS_MAX, seed);
                if (op == BF_OP_RECIP) {
                    test_op(op, 0, duration_ms, BF_EXP_BITS_MAX, seed);
                }
            }
        }
        seed++;
        duration_ms = 1000;
    }
    return 0;
}
