#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/ADT/StringRef.h"

#include "revng/ADT/TypeList.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/ObjectID.h"

template<typename T>
concept HasName = requires {
  { T::Name } -> std::same_as<const llvm::StringRef &>;
  requires not T::Name.empty();
};

//
// IsContainer
//

template<typename T>
concept IsContainer = requires(T &A, const T &AConst) {
  requires HasName<T>;
  { T() } -> std::same_as<T>;
  { T::Kind } -> std::same_as<const Kind &>;
  { AConst.objects() } -> std::same_as<std::set<ObjectID>>;
  { AConst.verify() } -> std::same_as<bool>;
  {
    A.deserialize(std::declval<const std::map<const ObjectID *,
                                              llvm::ArrayRef<const char>>>())
  } -> std::same_as<void>;
  {
    AConst.serialize(std::declval<const std::vector<const ObjectID *>>())
  } -> std::same_as<std::map<ObjectID, revng::pypeline::Buffer>>;
};

namespace detail {

template<typename T>
constexpr bool IsContainerArgument = IsContainer<T>;

template<typename T>
constexpr bool IsContainerArgument<const T> = IsContainer<T>;

template<typename T>
constexpr bool IsContainerReference = false;

template<typename T>
constexpr bool IsContainerReference<T &> = IsContainerArgument<T>;

} // namespace detail

//
// IsAnalysis
//

namespace detail {

template<typename T>
struct AnalysisRunTraits {};

template<typename C, typename... Args>
  requires(IsContainerReference<Args> and ...)
struct AnalysisRunTraits<llvm::Error (C::*)(Model &,
                                            const revng::pypeline::Request &,
                                            llvm::StringRef,
                                            Args...)> {
  using ContainerTypes = TypeList<std::remove_reference_t<Args>...>;
  static constexpr size_t Size = sizeof...(Args);
};

} // namespace detail

template<typename T>
using AnalysisRunTraits = detail::AnalysisRunTraits<decltype(&T::run)>;

template<typename T>
concept IsAnalysis = requires(T &A) {
  requires HasName<T>;
  { T() } -> std::same_as<T>;
  requires AnalysisRunTraits<T>::Size >= 0;
};

//
// IsPipe
//

namespace detail {

template<typename T>
concept HasContainerTypes = SpecializationOf<typename T::ContainerTypes,
                                             TypeList>;

template<typename T>
struct PipeRunTraitsHelper {};

template<typename C, typename... Args>
  requires(not HasContainerTypes<C>) and (IsContainerReference<Args> and ...)
struct PipeRunTraitsHelper<
  revng::pypeline::ObjectDependencies (C::*)(const Model &,
                                             const revng::pypeline::Request &,
                                             const revng::pypeline::Request &,
                                             llvm::StringRef,
                                             Args...)> {
  using ContainerTypes = TypeList<std::remove_reference_t<Args>...>;
};

template<typename T>
struct PipeRunTraits {};

template<HasContainerTypes T>
struct PipeRunTraits<T> {
  using ContainerTypes = T::ContainerTypes;
  static constexpr size_t ContainerCount = std::tuple_size_v<ContainerTypes>;
};

template<typename T>
  requires(not HasContainerTypes<T>)
struct PipeRunTraits<T> {
  using ContainerTypes = PipeRunTraitsHelper<decltype(&T::run)>::ContainerTypes;
  static constexpr size_t ContainerCount = std::tuple_size_v<ContainerTypes>;
};

template<typename T>
concept IsPipeArgumentDocumentation = requires {
  { T::Name } -> std::same_as<const llvm::StringRef &>;
  { T::HelpText } -> std::same_as<const llvm::StringRef &>;
};

template<SpecializationOf<TypeList> List>
inline constexpr bool checkDocumentation() {
  constexpr bool Size = std::tuple_size_v<List>;
  return compile_time::repeatAnd<Size>([]<size_t I>() {
    return IsPipeArgumentDocumentation<std::tuple_element_t<I, List>>;
  });
}

template<typename T>
inline constexpr size_t
  DocSize = std::tuple_size_v<typename T::ArgumentsDocumentation>;

} // namespace detail

template<typename T>
concept HasArgumentsDocumenation = requires {
  requires StrictSpecializationOf<typename T::ArgumentsDocumentation, TypeList>;
  requires detail::checkDocumentation<typename T::ArgumentsDocumentation>();
};

template<typename T>
using PipeRunTraits = detail::PipeRunTraits<T>;

template<typename T>
concept IsPipe = requires(T &A, llvm::StringRef StaticConfig) {
  requires HasName<T>;
  { T(StaticConfig) } -> std::same_as<T>;
  { A.StaticConfiguration } -> std::same_as<const std::string &>;
  requires HasArgumentsDocumenation<T>;
  requires PipeRunTraits<T>::ContainerCount == detail::DocSize<T>;
};
