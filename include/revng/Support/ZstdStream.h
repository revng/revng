#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/STLExtras.h"

#include "zstd.h"

void zstdCompress(llvm::raw_ostream &OS,
                  llvm::ArrayRef<uint8_t> Buffer,
                  int CompressionLevel = 3);

template<DataBuffer T>
inline void
zstdCompress(llvm::raw_ostream &OS, T Buffer, int CompressionLevel = 3) {
  zstdCompress(OS,
               { reinterpret_cast<const uint8_t *>(Buffer.data()),
                 Buffer.size() },
               CompressionLevel);
}

template<DataBuffer T>
inline llvm::SmallVector<char>
zstdCompress(T Buffer, int CompressionLevel = 3) {
  llvm::SmallVector<char> Result;
  llvm::raw_svector_ostream OS(Result);
  zstdCompress(OS, Buffer, CompressionLevel);
  return Result;
}

void zstdDecompress(llvm::raw_ostream &OS, llvm::ArrayRef<uint8_t> Buffer);

template<DataBuffer T>
inline void zstdDecompress(llvm::raw_ostream &OS, T Buffer) {
  zstdDecompress(OS,
                 { reinterpret_cast<const uint8_t *>(Buffer.data()),
                   Buffer.size() });
}

template<DataBuffer T>
inline llvm::SmallVector<char> zstdDecompress(T Buffer) {
  llvm::SmallVector<char> Result;
  llvm::raw_svector_ostream OS(Result);
  zstdDecompress(OS, Buffer);
  return Result;
}

class ZstdCompressedOstream : public llvm::raw_ostream {
private:
  llvm::raw_ostream &OS;
  llvm::SmallVector<char> OutBuffer;
  std::unique_ptr<ZSTD_CCtx, void (*)(ZSTD_CCtx *)> Ctx;

public:
  ZstdCompressedOstream(llvm::raw_ostream &DestOS, int CompressionLevel = 3);
  ~ZstdCompressedOstream() override;
  void flush();

private:
  void write_impl(const char *Ptr, size_t Size) override;
  uint64_t current_pos() const override { return OS.tell(); }
};
