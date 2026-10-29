#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class AllocaInst;
class Function;
class GlobalVariable;
class Use;
} // namespace llvm

/// Replaces the uses of global variables within one function with allocas
///
/// The constructor walks the function once and records where each selected
/// global is used, so that a replacement costs no more than the number of uses
/// it replaces.
///
/// Only a global used directly, or through a chain of cast expressions, can be
/// replaced: an equivalent cast of the alloca takes the place of the whole
/// chain. Referring to a selected global in any other way, by pointer
/// arithmetic or through metadata, aborts.
class GlobalToLocalPromoter {
private:
  /// All the uses of a single global variable within the function
  struct IndexedGlobal {
    llvm::GlobalVariable *Global = nullptr;
    llvm::SmallVector<llvm::Use *, 8> Uses;
    bool Replaced = false;
  };

  /// Fills in `Index` and `Position` by walking the function once
  class Initializer;

public:
  /// Which globals to index. An empty filter selects all of them.
  using Filter = std::function<bool(const llvm::GlobalVariable &)>;

private:
  const llvm::Function *TheFunction = nullptr;

  /// The uses of each selected global, sorted by the name of the global
  llvm::SmallVector<IndexedGlobal> Index;

  /// Where each global sits in `Index`
  llvm::DenseMap<const llvm::GlobalVariable *, unsigned> Position;

public:
  GlobalToLocalPromoter(const Filter &ShouldPromote, llvm::Function &F);

public:
  /// \return the globals the function uses, sorted by name
  auto globals() const { return llvm::map_range(Index, getGlobal); }

public:
  /// Replace with \p Alloca all the uses of \p Global within the function
  ///
  /// \return true if at least one use has been replaced
  bool replaceWithAlloca(llvm::GlobalVariable *Global,
                         llvm::AllocaInst *Alloca);

private:
  static llvm::GlobalVariable *getGlobal(const IndexedGlobal &Indexed) {
    return Indexed.Global;
  }

  IndexedGlobal *find(const llvm::GlobalVariable *Global);
};
