#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "mlir/IR/PatternMatch.h"

class EFAThreadCache {
  struct ImplType;
  std::unique_ptr<ImplType> Impl;

public:
  EFAThreadCache();
  EFAThreadCache(const EFAThreadCache &) = delete;
  EFAThreadCache &operator=(const EFAThreadCache &) = delete;
  ~EFAThreadCache();
};

void populateWithEmitFieldAccessesPatterns(mlir::RewritePatternSet &Set);
