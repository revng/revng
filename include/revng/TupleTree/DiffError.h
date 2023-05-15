#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/TupleTree/DocumentError.h"
#include "revng/TupleTree/TupleTreePath.h"

namespace revng {

class DiffLocation : public LocationBase {
public:
  enum class KindType {
    All,
    New,
    Old,
    Path
  };

private:
  size_t DiffEntryIndex;
  KindType Kind;

public:
  DiffLocation(size_t DiffEntryIndex, KindType Kind) :
    DiffEntryIndex(DiffEntryIndex), Kind(Kind) {}

  static std::string kindToString(KindType Kind) {
    switch (Kind) {
    case KindType::All:
      return "*";
    case KindType::New:
      return "New";
    case KindType::Old:
      return "Old";
    case KindType::Path:
      return "Path";
    }
    revng_unreachable("Unreachable");
    return "";
  }

  std::string toString() const override {
    return ("/Changes/" + llvm::Twine(DiffEntryIndex) + "/"
            + kindToString(Kind))
      .str();
  }

  ~DiffLocation() override = default;
  static std::string getTypeName() { return "DiffLocation"; }
};

class DiffError : public DocumentError<DiffError, DiffLocation> {
public:
  using DocumentError<DiffError, DiffLocation>::DocumentError;
  inline static char ID = '0';

  std::string getTypeName() const override { return "DiffError"; }
};

} // namespace revng
