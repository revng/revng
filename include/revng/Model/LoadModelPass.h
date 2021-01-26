#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/Model/Binary.h"

inline const char *ModelMetadataName = "revng.model";

class LoadModelPass : public llvm::ImmutablePass {
public:
  static char ID;

private:
  model::Binary TheBinary;
  bool Modified = false;

public:
  LoadModelPass() : llvm::ImmutablePass(ID) {}

public:
  bool doInitialization(llvm::Module &M) override final;
  bool doFinalization(llvm::Module &M) override final;

public:
  bool hasChanged() const { return Modified; }

  const model::Binary &getReadOnlyModel() { return TheBinary; }
  model::Binary &getWriteableModel() {
    Modified = true;
    return TheBinary;
  }
};
