#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"

#include "revng/ADT/SortedVector.h"
#include "revng/Clift/Clift.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/MetaAddress.h"

namespace clift {

/// Gather the set of instruction addresses identifying a statement.
///
/// The addresses are those attached to the operations in the statement's own
/// expression regions, i.e. all of its regions except the ones holding nested
/// statements (loop and branch bodies): those addresses identify the nested
/// statements, not this one. So a `return`, a local variable declaration or an
/// expression statement is identified by the addresses of its expression, while
/// an `if` or a loop is identified by the addresses of its condition alone.
///
/// This is the address set used to place comments (see CommentPlacementHelper),
/// so a comment whose location is set to the result of this function matches
/// the statement exactly.
inline SortedVector<MetaAddress>
getStatementExpressionAddresses(mlir::Operation *Op) {
  SortedVector<MetaAddress> Addresses;

  auto GatherFromRegion = [&Addresses](mlir::Region &Region) {
    Region.walk([&Addresses](mlir::Operation *Nested) {
      auto Loc = mlir::dyn_cast_or_null<mlir::NameLoc>(Nested->getLoc());
      if (not Loc)
        return;
      if (auto L = pipeline::locationFromString(revng::ranks::Instruction,
                                                Loc.getName().str())) {
        if (L->back().isValid())
          Addresses.insert(L->back());
      }
    });
  };

  llvm::SmallPtrSet<mlir::Region *, 4> StatementRegions;
  if (auto SRI = mlir::dyn_cast<clift::StatementRegionOpInterface>(Op))
    for (mlir::Region &Region : SRI.getStatementRegions())
      StatementRegions.insert(&Region);

  for (mlir::Region &Region : Op->getRegions())
    if (not StatementRegions.contains(&Region))
      GatherFromRegion(Region);

  return Addresses;
}

/// Gather the set of instruction addresses identifying a value, i.e. a local
/// variable or a label: the addresses attached to the operations that use it.
///
/// This matches the address set rev.ng reports for that value, so a
/// model::LocalVariable or model::GotoLabel located by it is picked up when
/// names and types are assigned. Returns an empty set if any user lacks a valid
/// address.
inline SortedVector<MetaAddress> getUserAddressSet(mlir::Value Value) {
  auto GetAddress = [](mlir::Operation *User) {
    if (auto Loc = mlir::dyn_cast_or_null<mlir::NameLoc>(User->getLoc()))
      if (auto L = pipeline::locationFromString(revng::ranks::Instruction,
                                                Loc.getName().str()))
        return L->back();
    return MetaAddress::invalid();
  };

  SortedVector<MetaAddress> Addresses;
  for (mlir::Operation *User : Value.getUsers()) {
    MetaAddress Address = GetAddress(User);
    if (not Address.isValid()) {
      Addresses.clear();
      break;
    }
    Addresses.insert(Address);
  }
  return Addresses;
}

} // namespace clift
