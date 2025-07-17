#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/Native/Analysis.h"
#include "revng/PipeboxCommon/Helpers/Native/Container.h"
#include "revng/PipeboxCommon/Helpers/Native/Pipe.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::helpers::native {

class RegistryImpl {
private:
  template<typename T, typename... Args>
  using FactoryMap = llvm::StringMap<std::unique_ptr<T> (*)(Args...)>;

public:
  FactoryMap<Container> Containers;
  FactoryMap<Analysis> Analyses;
  FactoryMap<Pipe, llvm::StringRef> Pipes;

public:
  RegistryImpl() = default;
  ~RegistryImpl() = default;
  RegistryImpl(const RegistryImpl &) = delete;
  RegistryImpl &operator=(const RegistryImpl &) = delete;
  RegistryImpl(RegistryImpl &&) = delete;
  RegistryImpl &operator=(const RegistryImpl &&) = delete;
};

inline RegistryImpl Registry;

} // namespace revng::pypeline::helpers::native
