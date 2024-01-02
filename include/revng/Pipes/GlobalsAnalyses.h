#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <tuple>
#include <vector>

#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Option.h"

namespace revng::pipes {

namespace options {

using pipeline::Option;

constexpr inline std::tuple DiffOptions = { Option("global-name", ""),
                                            Option("diff-content", "") };

constexpr inline std::tuple SetOptions = { Option("global-name", ""),
                                           Option("global-content", "") };

} // namespace options

/// This analysis that applies a tuple-tree diff to the specified global. If
/// applying the diff fails the global is guaranteed to be left untouched.
struct ApplyDiffAnalysis {
  static constexpr auto Name = "ApplyDiff";
  constexpr static std::tuple Options = options::DiffOptions;

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  llvm::Error run(pipeline::ExecutionContext &Ctx,
                  std::string DiffGlobalName,
                  std::string DiffContent);
};

/// This analysis that verifies that a diff will apply to a global. If
/// llvm::Error::success() is returned then the diff is safe to be applied to
/// the current global, otherwise an Error is returned.
struct VerifyDiffAnalysis {
  static constexpr auto Name = "VerifyDiff";
  constexpr static std::tuple Options = options::DiffOptions;

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  llvm::Error run(pipeline::ExecutionContext &Ctx,
                  std::string DiffGlobalNane,
                  std::string DiffContent);
};

/// This analysis replaces the contents of the specified global with the one
/// provided. The global will be verified before replacement and an error will
/// be returned if it's not conformant. In case of error this analysis does not
/// change the contents of the global.
struct SetGlobalAnalysis {
  static constexpr auto Name = "SetGlobal";
  constexpr static std::tuple Options = options::SetOptions;

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  llvm::Error run(pipeline::ExecutionContext &Ctx,
                  std::string SetGlobalNane,
                  std::string GlobalContent);
};

/// This analysis verifies that the contents provided conform to the `verify`
/// method of the specified global. If successful a llvm::Error::success() is
/// returned, otherwise the error from the `verify` is returned.
struct VerifyGlobalAnalysis {
  static constexpr auto Name = "VerifyGlobal";
  constexpr static std::tuple Options = options::SetOptions;

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  llvm::Error run(pipeline::ExecutionContext &Ctx,
                  std::string SetGlobalNane,
                  std::string GlobalContent);
};

} // namespace revng::pipes
