#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"

namespace pipeline {

namespace detail {
template<typename LLVMContainerType>
ContainerFactory makeLLVMContainerFactoryImpl(pipeline::Context *Ctx,
                                              llvm::LLVMContext *Context) {
  return [Ctx, Context](llvm::StringRef Name) {
    auto Module = std::make_unique<llvm::Module>(Name, *Context);
    return std::make_unique<LLVMContainerType>(*Ctx, std::move(Module), Name);
  };
}
} // namespace detail

template<typename LLVMContainerType>
ContainerFactory
makeLLVMContainerFactory(pipeline::Context &Ctx, llvm::LLVMContext &Context) {
  return detail::makeLLVMContainerFactoryImpl<LLVMContainerType>(&Ctx,
                                                                 &Context);
}

inline auto
  makeDefaultLLVMContainerFactory = makeLLVMContainerFactory<LLVMContainer>;

} // namespace pipeline
