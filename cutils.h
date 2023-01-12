/*
 * C utilities
 *
 * Copyright (c) 2017 Fabrice Bellard
 * Copyright (c) 2018 Charlie Gordon
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
#ifndef CUTILS_H
#define CUTILS_H

#include <stdlib.h>
#include <inttypes.h>

#include "quickjs.h"

#ifdef _MSC_VER
#include <windows.h>
#include <intrin.h>
#include <malloc.h>
#else
#include <alloca.h>
#endif


/* set if CPU is big endian */
#undef WORDS_BIGENDIAN

#if !defined(__GNUC__) && !defined(__clang__)
#undef __attribute__
#define __attribute__(x)

#undef __builtin_expect
#define __builtin_expect(cond, m)	(cond)
#endif

#if defined(_MSC_VER)
#define likely(x)    (x)
#define unlikely(x)  (x)
#define force_inline __forceinline
#define no_inline __declspec(noinline)
#define __maybe_unused
#define __js_printf_like(a, b)
#define __attribute__(x)
#define __attribute(x)
typedef intptr_t ssize_t;
#else
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define force_inline inline __attribute__((always_inline))
#define no_inline __attribute__((noinline))
#define __maybe_unused __attribute__((unused))
#define __js_printf_like(f, a)   __attribute__((format(printf, f, a)))
#endif

#define xglue(x, y) x ## y
#define glue(x, y) xglue(x, y)
#define stringify(s)    tostring(s)
#define tostring(s)     #s

#ifndef offsetof
#define offsetof(type, field) ((size_t) &((type *)0)->field)
#endif
#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

typedef int BOOL;

#ifndef FALSE
enum {
    FALSE = 0,
    TRUE = 1,
};
#endif

#ifndef no_return
#if defined(__GNUC__) || defined(__clang__)
#define no_return __attribute__ ((noreturn))
#elif defined(_MSC_VER)
#define no_return __declspec(noreturn)
#else
#define no_return
#endif
#endif

void pstrcpy(char *buf, int buf_size, const char *str);
char *pstrcat(char *buf, int buf_size, const char *s);
int strstart(const char *str, const char *val, const char **ptr);
int has_suffix(const char *str, const char *suffix);

static inline int max_int(int a, int b)
{
    if (a > b)
        return a;
    else
        return b;
}

static inline int min_int(int a, int b)
{
    if (a < b)
        return a;
    else
        return b;
}

static inline uint32_t max_uint32(uint32_t a, uint32_t b)
{
    if (a > b)
        return a;
    else
        return b;
}

static inline uint32_t min_uint32(uint32_t a, uint32_t b)
{
    if (a < b)
        return a;
    else
        return b;
}

static inline int64_t max_int64(int64_t a, int64_t b)
{
    if (a > b)
        return a;
    else
        return b;
}

static inline int64_t min_int64(int64_t a, int64_t b)
{
    if (a < b)
        return a;
    else
        return b;
}


// this chunk ripped from https://github.com/llvm-mirror/libcxx/blob/9dcbb46826fd4d29b1485f25e8986d36019a6dca/include/support/win32/support.h#L106-L182
#if defined(_MSC_VER)

// Bit builtin's make these assumptions when calling _BitScanForward/Reverse
// etc. These assumptions are expected to be true for Win32/Win64 which this
// file supports.
static_assert(sizeof(unsigned long long) == 8, "");
static_assert(sizeof(unsigned long) == 4, "");
static_assert(sizeof(unsigned int) == 4, "");

static inline int __builtin_popcount(unsigned int x)
{
	// Binary: 0101...
	static const unsigned int m1 = 0x55555555;
	// Binary: 00110011..
	static const unsigned int m2 = 0x33333333;
	// Binary:  4 zeros,  4 ones ...
	static const unsigned int m4 = 0x0f0f0f0f;
	// The sum of 256 to the power of 0,1,2,3...
	static const unsigned int h01 = 0x01010101;
	// Put count of each 2 bits into those 2 bits.
	x -= (x >> 1) & m1;
	// Put count of each 4 bits into those 4 bits.
	x = (x & m2) + ((x >> 2) & m2);
	// Put count of each 8 bits into those 8 bits.
	x = (x + (x >> 4)) & m4;
	// Returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24).
	return (x * h01) >> 24;
}

static inline int __builtin_popcountl(unsigned long x)
{
	return __builtin_popcount((int)(x));
}

static inline int __builtin_popcountll(unsigned long long x)
{
	// Binary: 0101...
	static const unsigned long long m1 = 0x5555555555555555;
	// Binary: 00110011..
	static const unsigned long long m2 = 0x3333333333333333;
	// Binary:  4 zeros,  4 ones ...
	static const unsigned long long m4 = 0x0f0f0f0f0f0f0f0f;
	// The sum of 256 to the power of 0,1,2,3...
	static const unsigned long long h01 = 0x0101010101010101;
	// Put count of each 2 bits into those 2 bits.
	x -= (x >> 1) & m1;
	// Put count of each 4 bits into those 4 bits.
	x = (x & m2) + ((x >> 2) & m2);
	// Put count of each 8 bits into those 8 bits.
	x = (x + (x >> 4)) & m4;
	// Returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
	return (int)((x * h01) >> 56);
}

// Returns the number of trailing 0-bits in x, starting at the least significant
// bit position. If x is 0, the result is undefined.
static inline int __builtin_ctzll(unsigned long long mask)
{
	unsigned long where;
	// Search from LSB to MSB for first set bit.
	// Returns zero if no set bit is found.
#if INTPTR_MAX >= INT64_MAX // 64-bit
	if (_BitScanForward64(&where, mask))
		return (int)(where);
#else
  // Win32 doesn't have _BitScanForward64 so emulate it with two 32 bit calls.
  // Scan the Low Word.
	if (_BitScanForward(&where, (unsigned long)(mask)))
		return (int)(where);
	// Scan the High Word.
	if (_BitScanForward(&where, (unsigned long)(mask >> 32)))
		return (int)(where + 32); // Create a bit offset from the LSB.
#endif
	return 64;
}

static inline int __builtin_ctzl(unsigned long mask)
{
	unsigned long where;
	// Search from LSB to MSB for first set bit.
	// Returns zero if no set bit is found.
	if (_BitScanForward(&where, mask))
		return (int)(where);
	return 32;
}

static inline int __builtin_ctz(unsigned int mask)
{
	// Win32 and Win64 expectations.
	static_assert(sizeof(mask) == 4, "");
	static_assert(sizeof(unsigned long) == 4, "");
	return __builtin_ctzl((unsigned long)(mask));
}

// Returns the number of leading 0-bits in x, starting at the most significant
// bit position. If x is 0, the result is undefined.
static inline int __builtin_clzll(unsigned long long mask)
{
	unsigned long where;
	// BitScanReverse scans from MSB to LSB for first set bit.
	// Returns 0 if no set bit is found.
#if INTPTR_MAX >= INT64_MAX // 64-bit
	if (_BitScanReverse64(&where, mask))
		return (int)(63 - where);
#else
  // Scan the high 32 bits.
	if (_BitScanReverse(&where, (unsigned long)(mask >> 32)))
		return (int)(63 -
			(where + 32)); // Create a bit offset from the MSB.
// Scan the low 32 bits.
	if (_BitScanReverse(&where, (unsigned long)(mask)))
		return (int)(63 - where);
#endif
	return 64; // Undefined Behavior.
}

static inline int __builtin_clzl(unsigned long mask)
{
	unsigned long where;
	// Search from LSB to MSB for first set bit.
	// Returns zero if no set bit is found.
	if (_BitScanReverse(&where, mask))
		return (int)(31 - where);
	return 32; // Undefined Behavior.
}

static inline int __builtin_clz(unsigned int x)
{
	return __builtin_clzl(x);
}

#endif // _LIBCPP_MSVC

/* WARNING: undefined if a = 0 */
static inline int clz32(unsigned int a)
{
    return __builtin_clz(a);
}

/* WARNING: undefined if a = 0 */
static inline int clz64(uint64_t a)
{
  return __builtin_clzll(a);
}

/* WARNING: undefined if a = 0 */
static inline int ctz32(unsigned int a)
{
    return __builtin_ctz(a);
}

/* WARNING: undefined if a = 0 */
static inline int ctz64(uint64_t a)
{
  return __builtin_ctzll(a);
}

#ifdef _MSC_VER
#pragma pack(push, 1)
struct packed_u64 {
    uint64_t v;
};

struct packed_u32 {
    uint32_t v;
};

struct packed_u16 {
    uint16_t v;
};
#pragma pack(pop)
#else
struct __attribute__((packed)) packed_u64 {
    uint64_t v;
};

struct __attribute__((packed)) packed_u32 {
    uint32_t v;
};

struct __attribute__((packed)) packed_u16 {
    uint16_t v;
};
#endif

static inline uint64_t get_u64(const uint8_t *tab)
{
    return ((const struct packed_u64 *)tab)->v;
}

static inline int64_t get_i64(const uint8_t *tab)
{
    return (int64_t)((const struct packed_u64 *)tab)->v;
}

static inline void put_u64(uint8_t *tab, uint64_t val)
{
    ((struct packed_u64 *)tab)->v = val;
}

static inline uint32_t get_u32(const uint8_t *tab)
{
    return ((const struct packed_u32 *)tab)->v;
}

static inline int32_t get_i32(const uint8_t *tab)
{
    return (int32_t)((const struct packed_u32 *)tab)->v;
}

static inline void put_u32(uint8_t *tab, uint32_t val)
{
    ((struct packed_u32 *)tab)->v = val;
}

static inline uint32_t get_u16(const uint8_t *tab)
{
    return ((const struct packed_u16 *)tab)->v;
}

static inline int32_t get_i16(const uint8_t *tab)
{
    return (int16_t)((const struct packed_u16 *)tab)->v;
}

static inline void put_u16(uint8_t *tab, uint16_t val)
{
    ((struct packed_u16 *)tab)->v = val;
}

static inline uint32_t get_u8(const uint8_t *tab)
{
    return *tab;
}

static inline int32_t get_i8(const uint8_t *tab)
{
    return (int8_t)*tab;
}

static inline void put_u8(uint8_t *tab, uint8_t val)
{
    *tab = val;
}

static inline uint16_t bswap16(uint16_t x)
{
    return (x >> 8) | (x << 8);
}

static inline uint32_t bswap32(uint32_t v)
{
    return ((v & 0xff000000) >> 24) | ((v & 0x00ff0000) >>  8) |
        ((v & 0x0000ff00) <<  8) | ((v & 0x000000ff) << 24);
}

static inline uint64_t bswap64(uint64_t v)
{
    return ((v & ((uint64_t)0xff << (7 * 8))) >> (7 * 8)) |
        ((v & ((uint64_t)0xff << (6 * 8))) >> (5 * 8)) |
        ((v & ((uint64_t)0xff << (5 * 8))) >> (3 * 8)) |
        ((v & ((uint64_t)0xff << (4 * 8))) >> (1 * 8)) |
        ((v & ((uint64_t)0xff << (3 * 8))) << (1 * 8)) |
        ((v & ((uint64_t)0xff << (2 * 8))) << (3 * 8)) |
        ((v & ((uint64_t)0xff << (1 * 8))) << (5 * 8)) |
        ((v & ((uint64_t)0xff << (0 * 8))) << (7 * 8));
}

/* XXX: should take an extra argument to pass slack information to the caller */
typedef void *DynBufReallocFunc(void *opaque, void *ptr, size_t size);

typedef struct DynBuf {
    uint8_t *buf;
    size_t size;
    size_t allocated_size;
    BOOL error; /* true if a memory allocation error occurred */
    DynBufReallocFunc *realloc_func;
    void *opaque; /* for realloc_func */
} DynBuf;

void dbuf_init(DynBuf *s);
void dbuf_init2(DynBuf *s, void *opaque, DynBufReallocFunc *realloc_func);
int dbuf_realloc(DynBuf *s, size_t new_size);
int dbuf_write(DynBuf *s, size_t offset, const uint8_t *data, size_t len);
int dbuf_put(DynBuf *s, const uint8_t *data, size_t len);
int dbuf_put_self(DynBuf *s, size_t offset, size_t len);
int dbuf_putc(DynBuf *s, uint8_t c);
int dbuf_putstr(DynBuf *s, const char *str);

static inline int dbuf_put_u16(DynBuf *s, uint16_t val)
{
    return dbuf_put(s, (uint8_t *)&val, 2);
}
static inline int dbuf_put_u32(DynBuf *s, uint32_t val)
{
    return dbuf_put(s, (uint8_t *)&val, 4);
}
static inline int dbuf_put_u64(DynBuf *s, uint64_t val)
{
    return dbuf_put(s, (uint8_t *)&val, 8);
}
int __js_printf_like(2, 3) dbuf_printf(DynBuf* s, const char* fmt, ...);
void dbuf_free(DynBuf *s);
static inline BOOL dbuf_error(DynBuf *s) {
    return s->error;
}
static inline void dbuf_set_error(DynBuf *s)
{
    s->error = TRUE;
}

#define UTF8_CHAR_LEN_MAX 6

int unicode_to_utf8(uint8_t *buf, unsigned int c);
int unicode_from_utf8(const uint8_t *p, int max_len, const uint8_t **pp);

static inline int from_hex(int c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    else if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    else if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    else
        return -1;
}

void rqsort(void *base, size_t nmemb, size_t size,
            int (*cmp)(const void *, const void *, void *),
            void *arg);

#endif  /* CUTILS_H */
