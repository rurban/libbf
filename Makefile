# Tiny arbitrary precision floating point library
# 
# Copyright (c) 2017-2018 Fabrice Bellard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Enable Windows compilation
#CONFIG_WIN32=y
# build AVX2 version
CONFIG_AVX2=y
# Enable profiling with gprof
#CONFIG_PROFILE=y
# compile the bftest utility to do regression tests and benchmarks. Must have
# the MPFR and MPDecimal libraries
#CONFIG_BFTEST=y
# 32 bit compilation
#CONFIG_M32=y

#CONFIG_ASAN=y

ifdef CONFIG_WIN32
CROSS_PREFIX=x86_64-w64-mingw32-
EXE:=.exe
else
EXE:=
endif

CC=$(CROSS_PREFIX)gcc
CFLAGS=-Wall -g $(PROFILE) -MMD
CFLAGS+=-O2
CFLAGS+=-flto
#CFLAGS+=-Os
LDFLAGS=
ifdef CONFIG_PROFILE
CFLAGS+=-p
LDFLAGS+=-p
else
#LDFLAGS+=-s # strip output
endif
ifdef CONFIG_ASAN
CFLAGS+=-fsanitize=address
LDFLAGS+=-fsanitize=address
endif
LIBS=-lm

PROGS+=bfbench$(EXE) tinypi$(EXE)
ifdef CONFIG_BFTEST
PROGS+=bftest$(EXE)
ifdef CONFIG_M32
PROGS+=bftest32$(EXE)
endif
endif
ifdef CONFIG_AVX2
PROGS+=bfbench-avx2$(EXE) tinypi-avx2$(EXE)
endif

all: $(PROGS)

tinypi$(EXE): tinypi.o libbf.o cutils.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

tinypi-avx2$(EXE): tinypi.avx2.o libbf.avx2.o cutils.avx2.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

BFTEST_LIBS:=$(LIBS)

ifdef CONFIG_BFTEST
BFTEST_LIBS:=-lmpfr -lgmp $(BFTEST_LIBS)
bfbench.o bfbench.avx2.o: CFLAGS+=-DCONFIG_MPFR

bftest$(EXE): bftest.o libbf.o cutils.o softfp.o
	$(CC) $(LDFLAGS) -o $@ $^ -lmpdec $(BFTEST_LIBS)

ifdef CONFIG_M32
bftest32$(EXE): bftest.m32.o libbf.m32.o cutils.m32.o softfp.m32.o
	$(CC) $(LDFLAGS) -m32 -o $@ $^ -lmpdec $(BFTEST_LIBS)
endif
endif

bfbench$(EXE): bfbench.o libbf.o cutils.o
	$(CC) $(LDFLAGS) -o $@ $^ $(BFTEST_LIBS)

bfbench-avx2$(EXE): bfbench.avx2.o libbf.avx2.o  cutils.avx2.o
	$(CC) $(LDFLAGS) -o $@ $^ $(BFTEST_LIBS)

test: all
	time ./tinypi 1e5 pi_1e5.txt
	sha1sum -c pi_1e5.sha1sum
ifdef CONFIG_AVX2
	time ./tinypi-avx2 1e5 pi_1e5.txt
	sha1sum -c pi_1e5.sha1sum
endif
#
	time ./tinypi 1e6 pi_1e6.txt
	sha1sum -c pi_1e6.sha1sum
ifdef CONFIG_AVX2
	time ./tinypi-avx2 1e6 pi_1e6.txt
	sha1sum -c pi_1e6.sha1sum
#
	time ./tinypi-avx2 1e7 pi_1e7.txt
	sha1sum -c pi_1e7.sha1sum
#
#	time ./tinypi-avx2 1e8 pi_1e8.txt
#	sha1sum -c pi_1e8.sha1sum
endif

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.m32.o: %.c
	$(CC) -m32 $(CFLAGS) -c -o $@ $<

%.avx2.o: %.c
	$(CC) $(CFLAGS) -mavx -mavx2 -mfma -mbmi2 -c -o $@ $<

clean:
	rm -f $(PROGS) *.o *.d *~

-include $(wildcard *.d)
