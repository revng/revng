#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/ObjectID.h"

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
  { T::Name } -> std::same_as<const llvm::StringRef &>;
  { T() };
  { AConst.kind() } -> std::same_as<ObjectID::Kind>;
  { AConst.objects() } -> std::same_as<std::set<ObjectID>>;
  { AConst.verify() } -> std::same_as<bool>;
  { A.deserialize(InputData) } -> std::same_as<void>;
  {
    AConst.serialize(OutputSet)
  } -> std::same_as<std::map<ObjectID, pypeline::Buffer>>;
};

//
// IsAnalysis
//

template<typename T>
inline constexpr bool IsAnalysisRunMethod = false;

template<typename Return, typename C, typename... Args>
  requires(std::same_as<Return, llvm::Error>
           && (IsContainer<std::remove_reference_t<Args>> && ...))
inline constexpr bool IsAnalysisRunMethod<
  Return (C::*)(Model *, pypeline::Request, llvm::StringRef, Args...)> = true;

template<typename T>
concept IsAnalysis = requires(T A) {
  { T::Name } -> std::same_as<const llvm::StringRef &>;
} && IsAnalysisRunMethod<decltype(&T::run)>;

//
// IsPipe
//

template<typename T>
inline constexpr bool IsPipeRunMethod = false;

template<typename Return, typename C, typename... Args>
  requires(std::same_as<Return, pypeline::ObjectDependencies>
           && (IsContainer<std::remove_reference_t<Args>> && ...))
inline constexpr bool IsPipeRunMethod<Return (C::*)(const Model *,
                                                    pypeline::Request,
                                                    pypeline::Request,
                                                    llvm::StringRef,
                                                    Args...)> = true;

template<typename T>
concept IsPipe = requires(T A, llvm::StringRef StaticConfig) {
  { T::Name } -> std::same_as<const llvm::StringRef &>;
  { T(StaticConfig) };
} && IsPipeRunMethod<decltype(&T::run)>;
