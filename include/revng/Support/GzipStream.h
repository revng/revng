#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

void gzipCompress(llvm::raw_ostream &OS,
                  llvm::ArrayRef<uint8_t> Buffer,
                  int CompressionLevel = 3);

inline void gzipCompress(llvm::raw_ostream &OS, llvm::ArrayRef<char> Buffer) {
  return gzipCompress(OS,
                      { reinterpret_cast<const uint8_t *>(Buffer.data()),
                        Buffer.size() });
}

void gzipDecompress(llvm::raw_ostream &OS, llvm::ArrayRef<uint8_t> Buffer);

inline void gzipDecompress(llvm::raw_ostream &OS, llvm::ArrayRef<char> Buffer) {
  return gzipDecompress(OS,
                        { reinterpret_cast<const uint8_t *>(Buffer.data()),
                          Buffer.size() });
}
