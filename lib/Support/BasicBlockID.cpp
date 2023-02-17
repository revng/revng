/// \file BasicBlockID.cpp
/// \brief Implementation of BasicBlockID.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include "revng/Support/BasicBlockID.h"
#include "revng/Support/IRHelpers.h"

static const char *BlockIDUniqueStringID = "block-id";

BasicBlockID BasicBlockID::fromString(llvm::StringRef Text) {
  using namespace llvm;
  SmallVector<StringRef, 2> Splitted;
  Text.split(Splitted, "-");

  if (not(Splitted.size() == 1 or Splitted.size() == 2))
    return BasicBlockID::invalid();

  auto Start = MetaAddress::fromString(Splitted[0]);
  if (not Start.isValid())
    return BasicBlockID::invalid();

  uint64_t InliningIndex = 0;

  if (Splitted.size() == 2) {
    APInt APIndex;
    bool Failure = Splitted[1].getAsInteger(10, APIndex);
    if (Failure or APIndex.getBitWidth() > 64)
      return BasicBlockID::invalid();
    InliningIndex = APIndex.getLimitedValue();
  }

  return BasicBlockID(Start, InliningIndex);
}

std::string BasicBlockID::toString() const {
  std::string Result = Start.toString();

  if (isInlined()) {
    Result += "-" + llvm::Twine(InliningIndex).str();
  }

  return Result;
}

BasicBlockID BasicBlockID::fromValue(llvm::Value *V) {
  return BasicBlockID::fromString(extractFromConstantStringPtr(V));
}

llvm::Constant *BasicBlockID::toValue(llvm::Module *M) const {
  return getUniqueString(M, BlockIDUniqueStringID, toString());
}
