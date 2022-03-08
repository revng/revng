#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/SavableObject.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

namespace pipeline {

template<Yamlizable T>
class YamlizableGlobal : public pipeline::SavableObject<YamlizableGlobal<T>> {
private:
  TupleTree<T> Obj;

public:
  static const char ID;

  void clear() final { Obj = TupleTree<T>(); }

  llvm::Error serialize(llvm::raw_ostream &OS) const final {
    Obj.serialize(OS);
    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) final {
    if (auto MaybeObject = TupleTree<T>::deserialize(Buffer.getBuffer());
        !MaybeObject)
      return llvm::errorCodeToError(MaybeObject.getError());
    else
      Obj = std::move(*MaybeObject);
    return llvm::Error::success();
  }

  explicit YamlizableGlobal(T &&Obj) : Obj(std::forward<T>(Obj)) {}
  YamlizableGlobal() = default;

  const TupleTree<T> &get() const { return Obj; }

  TupleTree<T> &get() { return Obj; }
};

} // namespace pipeline
