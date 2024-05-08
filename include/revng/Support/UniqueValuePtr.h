#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <memory>
#include <type_traits>

template<typename T>
concept ErasableFromParent = requires(T *Pointer) {
  Pointer->eraseFromParent();
};

template<ErasableFromParent T>
void eraseLLVMValue(T *Pointer) {
  revng_assert(Pointer != nullptr);

  if (Pointer->getParent() != nullptr) {
    // TODO: extend eraseFromParent to dump users of llvm::Function and adopt it
    //       there
    Pointer->eraseFromParent();
  } else {
    delete Pointer;
  }
}

template<ErasableFromParent T>
using LLVMValueEraser = std::integral_constant<decltype(&eraseLLVMValue<T>),
                                               &eraseLLVMValue<T>>;

template<ErasableFromParent T>
using UniqueValuePtr = std::unique_ptr<T, LLVMValueEraser<T>>;
