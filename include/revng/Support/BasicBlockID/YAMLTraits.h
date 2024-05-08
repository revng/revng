#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Support/YAMLTraits.h"

#include "revng/Support/BasicBlockID.h"

template<>
struct llvm::yaml::ScalarTraits<BasicBlockID> {

  static void
  output(const BasicBlockID &Value, void *, llvm::raw_ostream &Output) {
    Output << Value.toString();
  }

  static StringRef input(llvm::StringRef Scalar, void *, BasicBlockID &Value) {
    Value = BasicBlockID::fromString(Scalar);
    return StringRef();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Double; }
};
