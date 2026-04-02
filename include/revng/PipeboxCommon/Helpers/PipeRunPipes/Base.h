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

template<typename ContainerType>
constexpr bool isReadOnly(const revng::pypeline::Access &Access) {
  using AccessEnum = revng::pypeline::Access;
  return Access == AccessEnum::Read
         or (Access == AccessEnum::Auto and std::is_const_v<ContainerType>);
}

template<StrictSpecializationOf<TypeList> T, typename PipeRunT>
constexpr size_t writableContainersCount() {
  size_t Result = 0;
  forEach<T>([&Result]<typename A, size_t I>() {
    using Argument = std::tuple_element_t<I, typename PipeRunT::Arguments>;
    if constexpr (not isReadOnly<A>(Argument::Access))
      Result += 1;
  });
  return Result;
}

template<StrictSpecializationOf<TypeList> T, typename PipeRunT>
  requires(writableContainersCount<T, PipeRunT>() == 1)
constexpr size_t writableContainerIndex() {
  int Result = -1;
  forEach<T>([&Result]<typename A, size_t I>() {
    using Argument = std::tuple_element_t<I, typename PipeRunT::Arguments>;
    if constexpr (not isReadOnly<A>(Argument::Access)) {
      Result = I;
    }
  });
  return Result;
}

template<typename T>
concept HasPipeRunCheckPrecondition = requires(const Model &Model) {
  { T::checkPrecondition(Model) } -> std::same_as<llvm::Error>;
};

template<typename T>
concept HasPipeRunInvalidate = requires(const T &A,
                                        const revng::pypeline::InvalidationData
                                          &ID,
                                        const ModelDiff &Diff) {
  { T::requiresCustomInvalidation(Diff) } -> std::same_as<bool>;
  {
    T::processCustomInvalidation(ID, Diff)
  } -> std::same_as<std::vector<std::set<ObjectID>>>;
};

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

  llvm::Error checkPrecondition(const Model &Model) const
    requires detail::HasPipeRunCheckPrecondition<T>
  {
    return T::checkPrecondition(Model);
  }

  bool requiresCustomInvalidation(const ModelDiff &Diff) const
    requires detail::HasPipeRunInvalidate<T>
  {
    return T::requiresCustomInvalidation(Diff);
  }

  std::vector<std::set<ObjectID>>
  processCustomInvalidation(const revng::pypeline::InvalidationData &ID,
                            const ModelDiff &Diff) const
    requires detail::HasPipeRunInvalidate<T>
  {
    return T::processCustomInvalidation(ID, Diff);
  }

protected:
  static constexpr size_t ContainerCount = std::tuple_size_v<ContainerTypes>;
  static constexpr size_t
    OutputContainerIndex = detail::writableContainerIndex<ContainerTypes, T>();
  using OutputContainerType = std::tuple_element_t<OutputContainerIndex,
                                                   ContainerTypes>;
};
