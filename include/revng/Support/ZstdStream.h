#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/DataBuffer.h"

#include "zstd.h"

void zstdCompress(llvm::raw_ostream &OS,
                  DataBuffer Buffer,
                  int CompressionLevel = 3);

inline llvm::SmallVector<char> zstdCompress(DataBuffer Buffer) {
  llvm::SmallVector<char> Result;
  llvm::raw_svector_ostream OS(Result);
  zstdCompress(OS, Buffer);
  return Result;
}

void zstdDecompress(llvm::raw_ostream &OS, DataBuffer Buffer);

inline llvm::SmallVector<char> zstdDecompress(DataBuffer Buffer) {
  llvm::SmallVector<char> Result;
  llvm::raw_svector_ostream OS(Result);
  zstdDecompress(OS, Buffer);
  return Result;
}

class ZstdCompressedOstream : public llvm::raw_ostream {
private:
  llvm::raw_ostream &OS;
  llvm::SmallVector<char> OutBuffer;
  ZSTD_CCtx *Ctx = nullptr;

public:
  ZstdCompressedOstream(llvm::raw_ostream &DestOS, int CompressionLevel = 3);
  ~ZstdCompressedOstream() override;
  void flush();

private:
  void write_impl(const char *Ptr, size_t Size) override;
  uint64_t current_pos() const override { return OS.tell(); }
};
