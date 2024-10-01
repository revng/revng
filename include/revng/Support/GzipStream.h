#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/STLExtras.h"

#include "zlib.h"

void gzipCompress(llvm::raw_ostream &OS,
                  llvm::ArrayRef<uint8_t> Buffer,
                  int CompressionLevel = 3);

template<DataBuffer T>
inline void
gzipCompress(llvm::raw_ostream &OS, T Buffer, int CompressionLevel = 3) {
  gzipCompress(OS,
               { reinterpret_cast<const uint8_t *>(Buffer.data()),
                 Buffer.size() },
               CompressionLevel);
}

template<DataBuffer T>
inline llvm::SmallVector<char>
gzipCompress(T Buffer, int CompressionLevel = 3) {
  llvm::SmallVector<char> Result;
  llvm::raw_svector_ostream OS(Result);
  gzipCompress(OS, Buffer);
  return Result;
}

void gzipDecompress(llvm::raw_ostream &OS, llvm::ArrayRef<uint8_t> Buffer);

template<DataBuffer T>
inline void gzipDecompress(llvm::raw_ostream &OS, T Buffer) {
  gzipDecompress(OS,
                 { reinterpret_cast<const uint8_t *>(Buffer.data()),
                   Buffer.size() });
}

template<DataBuffer T>
inline llvm::SmallVector<char> gzipDecompress(T Buffer) {
  llvm::SmallVector<char> Result;
  llvm::raw_svector_ostream OS(Result);
  gzipDecompress(OS, Buffer);
  return Result;
}

class GzipCompressedOstream : public llvm::raw_ostream {
private:
  llvm::raw_ostream &OS;
  llvm::SmallVector<uint8_t> OutBuffer;
  z_stream Stream;

public:
  GzipCompressedOstream(llvm::raw_ostream &OS, int CompressionLevel = 3);
  ~GzipCompressedOstream() override;
  void flush();

private:
  void write_impl(const char *Ptr, size_t Size) override;
  uint64_t current_pos() const override { return OS.tell(); }
};
