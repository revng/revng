#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/ObjectID.h"

template<typename T>
concept HasName = requires {
  { T::Name } -> std::same_as<const llvm::StringRef &>;
  requires not T::Name.empty();
};

//
// IsContainer
//

template<typename T>
concept IsContainer = requires(T A,
                               const T AConst,
                               const std::map<const ObjectID *,
                                              llvm::ArrayRef<const char>>
                                 InputData,
                               const std::vector<const ObjectID *> OutputSet) {
  requires HasName<T>;
  { T() } -> std::same_as<T>;
  { T::Kind } -> std::same_as<const ObjectID::Kind &>;
  { AConst.objects() } -> std::same_as<std::set<ObjectID>>;
  { AConst.verify() } -> std::same_as<bool>;
  { A.deserialize(InputData) } -> std::same_as<void>;
  {
    AConst.serialize(OutputSet)
  } -> std::same_as<std::map<ObjectID, revng::pypeline::Buffer>>;
};

//
// IsAnalysis
//

template<typename T>
inline constexpr bool IsAnalysisRunMethod = false;

template<typename Return, typename C, typename... Args>
  requires(std::same_as<Return, llvm::Error>
           && (IsContainer<std::remove_reference_t<Args>> && ...))
inline constexpr bool
  IsAnalysisRunMethod<Return (C::*)(Model *,
                                    revng::pypeline::Request,
                                    llvm::StringRef,
                                    Args...)> = true;

template<typename T>
concept IsAnalysis = requires(T A) {
  requires HasName<T>;
  { T() } -> std::same_as<T>;
  requires IsAnalysisRunMethod<decltype(&T::run)>;
};

//
// IsPipe
//

template<typename T>
inline constexpr bool IsPipeRunMethod = false;

template<typename Return, typename C, typename... Args>
  requires(std::same_as<Return, revng::pypeline::ObjectDependencies>
           && (std::is_reference_v<Args> && ...)
           && (IsContainer<std::remove_reference_t<Args>> && ...))
inline constexpr bool IsPipeRunMethod<Return (C::*)(const Model *,
                                                    revng::pypeline::Request,
                                                    revng::pypeline::Request,
                                                    llvm::StringRef,
                                                    Args...)> = true;

template<typename T>
concept IsPipe = requires(T A, llvm::StringRef StaticConfig) {
  requires HasName<T>;
  { T(StaticConfig) } -> std::same_as<T>;
  requires IsPipeRunMethod<decltype(&T::run)>;
};
