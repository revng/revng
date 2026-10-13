#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

/// A single failure of a single test, as reported by the verification.
struct Failure {
  std::string Test;
  std::string Reason;
};

/// One entry in the expected failure list.
///
/// An entry applies to the cross product of every ABI in \ref ABIs and every
/// test in \ref Tests. Each (ABI, test) pair it covers requires *one* of
/// \ref Errors to occur.
///
/// If \ref Errors list is empty, any failure is accepted, no matter what
/// the message is.
///
/// If \ref CanSucceed is set, every (ABI, test) pair of the entry is not
/// *required* to fail. This is intended to cover big groups tests for
/// areas that are still work-in-progress.
struct Expectation {
  std::vector<std::string> ABIs;
  std::vector<std::string> Tests;
  std::vector<std::string> Errors;
  bool CanSucceed = false;
};

/// Filters \p Failures against the list of expectations (see \ref Expectation).
///
/// \p Failures that match *any* \ref Expectation are removed from the list.
///
/// \p Expectations that were never triggered are mentioned as failures in
/// the output.
std::vector<Failure> filterFailures(llvm::ArrayRef<Failure> Failures,
                                    llvm::ArrayRef<Expectation> Expectations);

/// Reads the expected-failure documents at \p Paths, concatenated in order.
///
/// Each document is a top-level sequence of entries:
///
/// ```yaml
/// - ABIs: [a-list-of-abis]
///   tests: [a-list-of-tests]
///   errors: [a-list-of-error-messages]
/// ```
///
/// The documents are read as one list, so an entry in the second document is
/// numbered after every entry in the first. This lets consecutive steps share
/// the list.
///
/// For example, it's likely that a conversion introduces a new failure. But,
/// something that was already failing in the reference binary will never start
/// working because of the conversion.
///
/// An unreadable or malformed document is a hard error.
///
/// Important: this loader filters the list! As we have a single invocation
/// per ABI, there's no reason to keep unrelated entries - so those that do
/// *not* match \p ABIName are dropped.
std::vector<Expectation> loadExpectedFailures(llvm::ArrayRef<std::string> Paths,
                                              llvm::StringRef ABIName);
