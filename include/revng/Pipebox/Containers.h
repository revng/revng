#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

class PTMLCContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "PTMLCContainer";
  static constexpr llvm::StringRef MimeType = "text/x.c+ptml";
  static constexpr llvm::StringRef Compression = "zstd;level=2";
};

class PTMLCFunctionContainer : public FunctionToBytesContainer {
public:
  static constexpr llvm::StringRef Name = "PTMLCFunctionContainer";
  static constexpr llvm::StringRef MimeType = "text/x.c+ptml";
  static constexpr llvm::StringRef Compression = "zstd;level=1";
};

class CTypeContainer : public TypeDefinitionToBytesContainer {
public:
  static constexpr llvm::StringRef Name = "CTypeContainer";
  static constexpr llvm::StringRef MimeType = "text/x.c";
  static constexpr llvm::StringRef Compression = "zstd;level=-1";
};

class RecompilableArchiveContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "RecompilableArchiveContainer";
  static constexpr llvm::StringRef MimeType = "application/x-object";
  static constexpr llvm::StringRef Compression = "none";
};

} // namespace revng::pypeline
