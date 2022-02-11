#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

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
  llvm::Error storeToDisk(llvm::StringRef Path) const final {
    return serializeToFile(*Obj, Path);
  }

  llvm::Error loadFromDisk(llvm::StringRef Path) final {
    if (not llvm::sys::fs::exists(Path))
      return llvm::Error::success();

    auto MaybeObj = TupleTree<T>::fromFile(Path);
    if (not MaybeObj)
      return llvm::make_error<llvm::StringError>("Could not parse yamlizable ",
                                                 MaybeObj.getError());

    Obj = std::move(*MaybeObj);
    return llvm::Error::success();
  }

  explicit YamlizableGlobal(T &&Obj) : Obj(std::forward<T>(Obj)) {}
  YamlizableGlobal() = default;

  const TupleTree<T> &get() const { return Obj; }

  TupleTree<T> &get() { return Obj; }
};

} // namespace pipeline
