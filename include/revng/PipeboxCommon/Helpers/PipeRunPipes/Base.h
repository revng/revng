#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/CompilationTime.h"
#include "revng/ADT/Concepts.h"
#include "revng/ADT/STLExtras.h"
#include "revng/ADT/TypeList.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Helpers.h"

namespace detail {

template<StrictSpecializationOf<TypeList> T>
constexpr size_t writableContainersCount() {
  size_t Result = 0;
  forEach<T>([&Result]<typename A, size_t I>() {
    if constexpr (not std::is_const_v<A>)
      Result += 1;
  });
  return Result;
}

template<StrictSpecializationOf<TypeList> T>
  requires(writableContainersCount<T>() == 1)
constexpr size_t writableContainerIndex() {
  int Result = -1;
  forEach<T>([&Result]<typename A, size_t I>() {
    if constexpr (not std::is_const_v<A>) {
      Result = I;
    }
  });
  return Result;
}

} // namespace detail

template<typename T>
concept SingleOutputPipeBaseCompatible = requires {
  requires HasName<T>;
  requires SpecializationOf<PipeRunContainerTypes<T>, TypeList>;
};

template<SingleOutputPipeBaseCompatible T>
class SingleOutputPipeBase {
public:
  static constexpr llvm::StringRef Name = T::Name;
  using ContainerTypes = PipeRunContainerTypes<T>;

  const std::string StaticConfiguration;

  SingleOutputPipeBase(llvm::StringRef Configuration) :
    StaticConfiguration(Configuration.str()) {}

protected:
  static constexpr size_t ContainerCount = std::tuple_size_v<ContainerTypes>;
  static constexpr size_t
    OutputContainerIndex = detail::writableContainerIndex<ContainerTypes>();
  using OutputContainerType = std::tuple_element_t<OutputContainerIndex,
                                                   ContainerTypes>;
};
