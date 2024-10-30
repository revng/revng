/// \file ZstdStream.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Assert.h"
#include "revng/Support/ZstdStream.h"

#include "zstd.h"

constexpr size_t BufferSize = 16 * 1024;
static_assert(BufferSize <= UINT_MAX);

static void zstdContextFree(ZSTD_CCtx *Ctx) {
  size_t RC = ZSTD_freeCCtx(Ctx);
  revng_assert(ZSTD_isError(RC) == 0);
}

static void zstdContextFree(ZSTD_DCtx *Ctx) {
  size_t RC = ZSTD_freeDCtx(Ctx);
  revng_assert(ZSTD_isError(RC) == 0);
}

struct ZSTDCompress {
  using Context = ZSTD_CCtx;
  static constexpr auto readInput = ZSTD_compressStream;
  static constexpr auto flushOutput = ZSTD_flushStream;
  static constexpr auto endOutput = ZSTD_endStream;
};

struct ZSTDDecompress {
  using Context = ZSTD_DCtx;
  static constexpr auto readInput = ZSTD_decompressStream;
  static size_t flushOutput(ZSTD_DCtx *Ctx, ZSTD_outBuffer *Out) {
    ZSTD_inBuffer In = { .src = nullptr, .size = 0, .pos = 0 };
    return ZSTD_decompressStream(Ctx, Out, &In);
  }
  static size_t endOutput(ZSTD_DCtx *, ZSTD_outBuffer *) { return 0; }
};

template<typename T>
static void zstdReadInput(typename T::Context *Ctx,
                          llvm::raw_ostream &OutputOS,
                          llvm::ArrayRef<uint8_t> InputBuffer,
                          llvm::SmallVector<char> &OutBuffer) {
  ZSTD_inBuffer Input = { .src = InputBuffer.data(),
                          .size = InputBuffer.size(),
                          .pos = 0 };
  ZSTD_outBuffer Output = { .dst = OutBuffer.data(),
                            .size = BufferSize,
                            .pos = 0 };

  while (Input.pos < Input.size) {
    size_t RC = T::readInput(Ctx, &Output, &Input);
    revng_assert(ZSTD_isError(RC) == 0);

    if (Output.pos > 0) {
      OutputOS.write(OutBuffer.data(), Output.pos);
      Output.pos = 0;
    }
  }
}

template<typename T>
static void zstdFlushOutput(typename T::Context *Ctx,
                            llvm::raw_ostream &OutputOS,
                            llvm::SmallVector<char> &OutBuffer) {
  ZSTD_outBuffer Output = { .dst = OutBuffer.data(),
                            .size = BufferSize,
                            .pos = 0 };
  while (true) {
    size_t RC = T::flushOutput(Ctx, &Output);
    revng_assert(ZSTD_isError(RC) == 0);

    if (Output.pos > 0) {
      OutputOS.write(OutBuffer.data(), Output.pos);
      Output.pos = 0;
    } else {
      break;
    }
  }

  size_t RC = T::endOutput(Ctx, &Output);
  revng_assert(ZSTD_isError(RC) == 0);
  if (Output.pos > 0) {
    OutputOS.write(OutBuffer.data(), Output.pos);
    Output.pos = 0;
  }
}

void zstdCompress(llvm::raw_ostream &OS,
                  llvm::ArrayRef<uint8_t> InputBuffer,
                  int CompressionLevel) {
  revng_assert(CompressionLevel >= 1 and CompressionLevel <= 19);
  ZSTD_CCtx *Ctx = ZSTD_createCCtx();
  size_t RC = ZSTD_initCStream(Ctx, CompressionLevel);
  revng_assert(ZSTD_isError(RC) == 0);

  llvm::SmallVector<char> OutBuffer;
  OutBuffer.resize_for_overwrite(BufferSize);
  zstdReadInput<ZSTDCompress>(Ctx, OS, InputBuffer, OutBuffer);
  zstdFlushOutput<ZSTDCompress>(Ctx, OS, OutBuffer);
  zstdContextFree(Ctx);
}

void zstdDecompress(llvm::raw_ostream &OS,
                    llvm::ArrayRef<uint8_t> InputBuffer) {
  ZSTD_DCtx *Ctx = ZSTD_createDCtx();
  size_t RC = ZSTD_initDStream(Ctx);
  revng_assert(ZSTD_isError(RC) == 0);

  llvm::SmallVector<char> OutBuffer;
  OutBuffer.resize_for_overwrite(BufferSize);
  zstdReadInput<ZSTDDecompress>(Ctx, OS, InputBuffer, OutBuffer);
  zstdFlushOutput<ZSTDDecompress>(Ctx, OS, OutBuffer);
  zstdContextFree(Ctx);
}

ZstdCompressedOstream::ZstdCompressedOstream(llvm::raw_ostream &DestOS,
                                             int CompressionLevel) :
  llvm::raw_ostream(),
  OS(DestOS),
  OutBuffer(),
  Ctx(ZSTD_createCCtx(), zstdContextFree) {
  revng_assert(CompressionLevel >= 1 and CompressionLevel <= 19);
  OutBuffer.resize_for_overwrite(BufferSize);
  size_t RC = ZSTD_initCStream(&*Ctx, CompressionLevel);
  revng_assert(ZSTD_isError(RC) == 0);
}

ZstdCompressedOstream::~ZstdCompressedOstream() {
  flush();
}

void ZstdCompressedOstream::flush() {
  llvm::raw_ostream::flush();
  zstdFlushOutput<ZSTDCompress>(&*Ctx, OS, OutBuffer);
  OS.flush();
}

void ZstdCompressedOstream::write_impl(const char *Ptr, size_t Size) {
  zstdReadInput<ZSTDCompress>(&*Ctx,
                              OS,
                              { reinterpret_cast<const uint8_t *>(Ptr), Size },
                              OutBuffer);
}
