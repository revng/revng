#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

class PTMLCBytesContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "PTMLCBytesContainer";
  static constexpr llvm::StringRef MimeType = "text/x.c+ptml";
};

class PTMLCFunctionBytesContainer : public FunctionToBytesContainer {
public:
  static constexpr llvm::StringRef Name = "PTMLCFunctionBytesContainer";
  static constexpr llvm::StringRef MimeType = "text/x.c+ptml";
};

class PTMLCTypeBytesContainer : public TypeDefinitionToBytesContainer {
public:
  static constexpr llvm::StringRef Name = "PTMLCTypeBytesContainer";
  static constexpr llvm::StringRef MimeType = "text/x.c+ptml";
};

class RecompilableArchiveContainer : public BytesContainer {
public:
  static constexpr llvm::StringRef Name = "RecompilableArchiveContainer";
  static constexpr llvm::StringRef MimeType = "application/x-object";
};

} // namespace revng::pypeline
