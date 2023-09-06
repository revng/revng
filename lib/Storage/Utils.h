#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"

#include "revng/Support/PathList.h"

/// Utility function that checks the "cleanliness" of a path, returning false if
/// the path is "dirty".
/// Things that will result in a dirty path:
/// * Use of a leading '/'
/// * Use of './'
/// * Use of '../'
/// * Use of '//'
inline bool checkPath(llvm::StringRef InputPath, llvm::sys::path::Style Style) {
  llvm::StringRef Separator = llvm::sys::path::get_separator(Style);

  // TODO: how to make this windows-compatible?
  if (InputPath.starts_with(Separator))
    return false;

  bool TrailingSlash = InputPath.ends_with(Separator);

  llvm::SmallString<128> Temp(joinPath(Style, Separator, InputPath.str()));
  llvm::sys::path::remove_dots(Temp, true, Style);
  if (TrailingSlash)
    Temp.append(Separator);

  return InputPath == Temp.substr(1);
}
