/// \file GzipStream.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Assert.h"
#include "revng/Support/GzipStream.h"

#include "zlib.h"

constexpr size_t OutputBufferSize = 16 * 1024;
static_assert(OutputBufferSize <= UINT_MAX);

constexpr int WindowBits = 15 // 2**15 bytes (32k) of window
                           + 16; // Magic offset for gzip

template<int (*Next)(z_stream *, int)>
static void zlibReadInput(z_stream &Stream,
                          llvm::raw_ostream &OutputOS,
                          llvm::ArrayRef<uint8_t> InputBuffer,
                          llvm::SmallVector<uint8_t> &OutBuffer) {
  const char *OutBufferPtr = reinterpret_cast<const char *>(OutBuffer.data());

  size_t RemainingInput = InputBuffer.size();

  // next_in has type `uint8_t *`, because each call to inflate/deflate will
  // automatically increment the pointer, while not changing the contents. Due
  // to this, we must use const_cast to conform to the struct's type
  Stream.next_in = const_cast<uint8_t *>(InputBuffer.data());

  Stream.next_out = OutBuffer.data();
  Stream.avail_out = OutBuffer.size();

  // Consume input
  while (RemainingInput > 0) {
    size_t InputSize = RemainingInput > UINT_MAX ? UINT_MAX : RemainingInput;
    Stream.avail_in = InputSize;

    int RC = Next(&Stream, Z_NO_FLUSH);
    revng_assert(RC == Z_OK or RC == Z_STREAM_END);

    // Compute bytes consumed
    RemainingInput -= InputSize - Stream.avail_in;

    if (Stream.avail_out == 0) {
      // Empty the buffer
      OutputOS.write(OutBufferPtr, OutBuffer.size());
      Stream.next_out = OutBuffer.data();
      Stream.avail_out = OutBuffer.size();
    }
  }
}

template<int (*Next)(z_stream *, int)>
static void zlibFlushOutput(z_stream &Stream,
                            llvm::raw_ostream &OutputOS,
                            llvm::SmallVector<uint8_t> &OutBuffer) {
  const char *OutBufferPtr = reinterpret_cast<const char *>(OutBuffer.data());
  Stream.next_in = Z_NULL;
  Stream.avail_in = 0;

  // Flush out
  while (true) {
    int RC = Next(&Stream, Z_FINISH);
    revng_assert(RC == Z_OK or RC == Z_STREAM_END);

    if (RC == Z_OK) {
      revng_assert(Stream.avail_out == 0);
      OutputOS.write(OutBufferPtr, OutBuffer.size());
      Stream.next_out = OutBuffer.data();
      Stream.avail_out = OutBuffer.size();
    } else {
      size_t Consumed = OutBuffer.size() - Stream.avail_out;
      OutputOS.write(OutBufferPtr, Consumed);
      // We got Z_STREAM_END, we can get out of the flush loop
      break;
    }
  }

  OutputOS.flush();
}

void gzipCompress(llvm::raw_ostream &OutputOS,
                  llvm::ArrayRef<uint8_t> InputBuffer,
                  int CompressionLevel) {
  revng_assert(CompressionLevel >= 1 and CompressionLevel <= 9);
  z_stream Stream = { .zalloc = Z_NULL, .zfree = Z_NULL, .opaque = Z_NULL };

  int Strategy = Z_DEFAULT_STRATEGY;
  int RC = deflateInit2(&Stream,
                        CompressionLevel,
                        Z_DEFLATED,
                        WindowBits,
                        8,
                        Strategy);
  revng_assert(RC == Z_OK);

  llvm::SmallVector<uint8_t> OutBuffer;
  OutBuffer.resize_for_overwrite(OutputBufferSize);
  zlibReadInput<deflate>(Stream, OutputOS, InputBuffer, OutBuffer);
  zlibFlushOutput<deflate>(Stream, OutputOS, OutBuffer);

  revng_assert(deflateEnd(&Stream) == Z_OK);
}

void gzipDecompress(llvm::raw_ostream &OutputOS,
                    llvm::ArrayRef<uint8_t> InputBuffer) {
  z_stream Stream = { .zalloc = Z_NULL, .zfree = Z_NULL, .opaque = Z_NULL };
  revng_assert(inflateInit2(&Stream, WindowBits) == Z_OK);

  llvm::SmallVector<uint8_t> OutBuffer;
  OutBuffer.resize_for_overwrite(OutputBufferSize);
  zlibReadInput<inflate>(Stream, OutputOS, InputBuffer, OutBuffer);
  zlibFlushOutput<inflate>(Stream, OutputOS, OutBuffer);

  revng_assert(inflateEnd(&Stream) == Z_OK);
}

GzipCompressedOstream::GzipCompressedOstream(llvm::raw_ostream &OS,
                                             int CompressionLevel) :
  llvm::raw_ostream(), OS(OS), OutBuffer() {
  revng_assert(CompressionLevel >= 1 and CompressionLevel <= 9);
  Stream = { .zalloc = Z_NULL, .zfree = Z_NULL, .opaque = Z_NULL };

  int RC = deflateInit2(&Stream,
                        CompressionLevel,
                        Z_DEFLATED,
                        WindowBits,
                        8,
                        Z_DEFAULT_STRATEGY);
  revng_assert(RC == Z_OK);
  OutBuffer.resize_for_overwrite(OutputBufferSize);
}

GzipCompressedOstream::~GzipCompressedOstream() {
  flush();
  revng_assert(deflateEnd(&Stream) == Z_OK);
}

void GzipCompressedOstream::flush() {
  llvm::raw_ostream::flush();
  zlibFlushOutput<deflate>(Stream, OS, OutBuffer);
  OS.flush();
}

void GzipCompressedOstream::write_impl(const char *Ptr, size_t Size) {
  llvm::ArrayRef<uint8_t> Input(reinterpret_cast<const uint8_t *>(Ptr), Size);
  zlibReadInput<deflate>(Stream, OS, Input, OutBuffer);
}
