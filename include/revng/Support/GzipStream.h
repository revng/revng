#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/DataBuffer.h"

#include "zlib.h"

void gzipCompress(llvm::raw_ostream &OS,
                  DataBuffer Buffer,
                  int CompressionLevel = 3);

inline llvm::SmallVector<char> gzipCompress(DataBuffer Buffer) {
  llvm::SmallVector<char> Result;
  llvm::raw_svector_ostream OS(Result);
  gzipCompress(OS, Buffer);
  return Result;
}

void gzipDecompress(llvm::raw_ostream &OS, DataBuffer Buffer);

inline llvm::SmallVector<char> gzipDecompress(DataBuffer Buffer) {
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
