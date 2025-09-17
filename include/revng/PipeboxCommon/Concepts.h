#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/ADT/StringRef.h"

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
constexpr bool IsContainerReference = false;

template<typename T>
constexpr bool IsContainerReference<const T &> = IsContainer<T>;

template<typename T>
constexpr bool IsContainerReference<T &> = IsContainer<T>;

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
  using ContainerTypes = std::tuple<std::remove_reference_t<Args>...>;
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
struct PipeRunTraits {};

template<typename C, typename... Args>
  requires(IsContainerReference<Args> and ...)
struct PipeRunTraits<
  revng::pypeline::ObjectDependencies (C::*)(const Model &,
                                             const revng::pypeline::Request &,
                                             const revng::pypeline::Request &,
                                             llvm::StringRef,
                                             Args...)> {
  using ContainerTypes = std::tuple<std::remove_reference_t<Args>...>;
  static constexpr size_t Size = sizeof...(Args);
};

template<typename T>
using DocTraits = compile_time::ArrayTraits<T::ArgumentsDocumentation>;

} // namespace detail

template<typename T>
using PipeRunTraits = detail::PipeRunTraits<decltype(&T::run)>;

template<typename T>
concept IsPipe = requires(T &A, llvm::StringRef StaticConfig) {
  requires HasName<T>;
  { T(StaticConfig) } -> std::same_as<T>;
  { A.StaticConfiguration } -> std::same_as<const std::string &>;
  requires std::is_array_v<decltype(T::ArgumentsDocumentation)>;
  requires std::is_same_v<typename detail::DocTraits<T>::value_type,
                          const revng::pypeline::PipeArgumentDocumentation>;
  requires PipeRunTraits<T>::Size == detail::DocTraits<T>::Size;
};
