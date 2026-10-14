//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Model/ABI.h"
#include "revng/Support/Assert.h"
#include "revng/Support/YAMLTraits.h"

#include "ExpectedFailures.h"

template<>
struct llvm::yaml::MappingTraits<Expectation> {
  static void mapping(IO &IO, Expectation &E) {
    IO.mapRequired("ABIs", E.ABIs);
    IO.mapRequired("tests", E.Tests);
    IO.mapRequired("errors", E.Errors);
    IO.mapOptional("can-succeed", E.CanSucceed, false);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(Expectation)

std::vector<Expectation> loadExpectedFailures(llvm::ArrayRef<std::string> Paths,
                                              llvm::StringRef ABIName) {
  std::vector<Expectation> Result;

  for (const std::string &Path : Paths) {
    // Load the list,
    auto Parsed = fromFileOrSTDIN<std::vector<Expectation>>(Path);
    if (not Parsed) {
      std::string Message = "Unable to read the expected failures from `" + Path
                            + "`: " + consumeToString(Parsed.takeError());
      revng_abort(Message.c_str());
    }

    // verify it is well-formed,
    for (const Expectation &E : *Parsed) {
      revng_assert(not E.ABIs.empty());
      revng_assert(not E.Tests.empty());

      for (const std::string &Name : E.ABIs) {
        if (model::ABI::fromName(Name) == model::ABI::Invalid) {
          std::string Message = "`" + Name + "` is not a known ABI name.";
          revng_abort(Message.c_str());
        }
      }
    }

    // then filter it,
    auto F = *Parsed | std::views::filter([&ABIName](const Expectation &E) {
      return llvm::is_contained(E.ABIs, ABIName.str());
    });

    // and append to the result.
    Result.insert(Result.end(),
                  std::make_move_iterator(F.begin()),
                  std::make_move_iterator(F.end()));
  }

  return Result;
}

std::vector<Failure> filterFailures(llvm::ArrayRef<Failure> Failures,
                                    llvm::ArrayRef<Expectation> Expectations) {
  std::vector<Failure> Result;

  // This pseudo-array is here to keep track of every expectation test
  // that has a corresponding failure.
  std::vector<std::unordered_set<std::string_view>> MatchedExpectations;
  MatchedExpectations.resize(Expectations.size());

  for (const Failure &Failure : Failures) {
    bool WasFailureExpected = false;
    for (const auto &[Index, Expectation] : llvm::enumerate(Expectations)) {
      if (llvm::is_contained(Expectation.Tests, Failure.Test)) {
        bool ErrorMatches = Expectation.Errors.empty()
                            or llvm::is_contained(Expectation.Errors,
                                                  Failure.Reason);
        if (ErrorMatches) {
          MatchedExpectations[Index].emplace(Failure.Test);
          WasFailureExpected = true;
        }
      }
    }

    if (not WasFailureExpected)
      Result.emplace_back(std::move(Failure));
  }

  for (const auto &[Index, Expectation] : llvm::enumerate(Expectations)) {
    if (Expectation.CanSucceed) {
      // Explicitly allowed to describe tests that may succeed, so the
      // stale-expectation check does not apply to it.
      continue;
    }

    if (Expectation.Tests.size() == MatchedExpectations[Index].size()) {
      // All the expected failures were matched.
      continue;
    }

    for (const std::string &Test : Expectation.Tests) {
      if (not MatchedExpectations[Index].contains(Test))
        Result.emplace_back(Failure{ .Test = Test,
                                     .Reason = "An entry in the expectation "
                                               + std::to_string(Index)
                                               + " didn't match a failure." });
    }
  }

  return Result;
}
