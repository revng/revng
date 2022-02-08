#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

namespace pipeline {

/// PathComponent represents either a name or *, it's meant to represent either
/// a single object or all possible object
class PathComponent {
private:
  std::optional<std::string> Name;
  PathComponent() : Name(std::nullopt) {}

public:
  PathComponent(std::optional<std::string> Name) : Name(std::move(Name)) {}

  static PathComponent all() { return PathComponent(); }

public:
  bool isAll() const { return not Name.has_value(); }
  bool isSingle() const { return Name.has_value(); }

  const std::string &getName() const {
    revng_assert(isSingle());
    return *Name;
  }

public:
  bool operator<(const PathComponent &Other) const { return Name < Other.Name; }

  bool operator<=>(const PathComponent &Other) const {
    if (not Name.has_value() and not Other.Name.has_value())
      return 0;
    if (not Name.has_value())
      return -1;
    if (not Other.Name.has_value())
      return 1;
    return strcmp(Name->c_str(), Other.Name->c_str());
  }

  bool operator==(const PathComponent &Other) const {
    return (*this <=> Other) == 0;
  }

  bool operator!=(const PathComponent &Other) const {
    return (*this == Other) == false;
  }

public:
  std::string toString() const debug_function {
    if (Name.has_value())
      return *Name;
    else
      return "*";
  }

public:
  template<typename OStream>
  void dump(OStream &OS) const debug_function {
    OS << toString();
  }

  void dump() const debug_function { dump(dbg); }
};

using PathComponents = llvm::SmallVector<PathComponent, 4>;

} // namespace pipeline
