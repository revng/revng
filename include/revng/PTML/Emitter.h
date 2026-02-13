#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace ptml {

/// \brief Generic streaming interface for emitting arbitrary text.
template<typename EmitterT>
concept Emitter = requires(EmitterT &Emitter, llvm::StringRef String) {
  // void emit(llvm::StringRef String);
  Emitter.emit(String);
};

class StreamEmitter {
protected:
  llvm::raw_ostream &OS;

public:
  explicit StreamEmitter(llvm::raw_ostream &OS) : OS(OS) {}

  void emit(llvm::StringRef String) { OS << String; }
};
static_assert(Emitter<StreamEmitter>);

} // namespace ptml
