#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"

/// Keep the CPU-state byte address available after the lifter is destroyed.
inline void setCPUStateOffset(llvm::GlobalVariable &CSV, uint64_t Offset) {
  auto &Context = CSV.getContext();
  auto *Value = llvm::ConstantInt::get(llvm::Type::getInt64Ty(Context), Offset);
  CSV.setMetadata("revng.csv.offset",
                  llvm::MDNode::get(Context,
                                    llvm::ConstantAsMetadata::get(Value)));
}
