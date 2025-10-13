#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "llvm/Pass.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/RawBinaryView.h"

class LoadBinaryWrapperPass : public llvm::ModulePass {
public:
  static char ID;

private:
  std::unique_ptr<llvm::MemoryBuffer> MaybeBuffer;
  llvm::ArrayRef<uint8_t> Data;
  std::optional<RawBinaryView> BinaryView;

public:
  LoadBinaryWrapperPass(llvm::ArrayRef<uint8_t> Data) :
    llvm::ModulePass(ID), Data(Data) {
    revng_check(Data.data() != nullptr);
  }

  LoadBinaryWrapperPass(llvm::StringRef Data) :
    LoadBinaryWrapperPass(toArrayRef(Data)) {}

public:
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
  }

public:
  RawBinaryView &get() {
    revng_assert(BinaryView);
    return *BinaryView;
  }

public:
  bool runOnModule(llvm::Module &M) override final;
};
