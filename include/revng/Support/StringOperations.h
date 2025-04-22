#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA1.h"

namespace revng {

inline std::string nameHash(llvm::StringRef String) {
  llvm::ArrayRef Data(reinterpret_cast<const uint8_t *>(String.data()),
                      String.size());
  return llvm::toHex(llvm::SHA1::hash(Data), true);
}

inline std::string mangleName(llvm::StringRef String) {
  auto IsPrintable = [](llvm::StringRef String) {
    return llvm::all_of(String, llvm::isPrint);
  };

  auto ContainsSpaces = [](llvm::StringRef String) {
    return llvm::any_of(String, llvm::isSpace);
  };

  constexpr auto SHA1HexLength = 40;
  if (String.size() > SHA1HexLength or not IsPrintable(String)
      or ContainsSpaces(String) or String.empty()) {
    return nameHash(String);
  } else {
    return String.str();
  }
}

} // namespace revng
