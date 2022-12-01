#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/ABI/Trait.h"
#include "revng/ADT/STLExtras.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/ABI.h"
#include "revng/Model/Register.h"

namespace abi {

template<ranges::sized_range Container>
llvm::SmallVector<model::Register::Values, 8>
orderArguments(const Container &Registers, model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&Registers]<model::ABI::Values A>() {
    SortedVector<model::Register::Values> Lookup;
    {
      auto Inserter = Lookup.batch_insert();
      for (auto &&Register : Registers)
        Inserter.insert(Register);
    }

    llvm::SmallVector<model::Register::Values, 8> Result;
    for (auto Register : abi::Trait<A>::GeneralPurposeArgumentRegisters)
      if (Lookup.count(Register) != 0)
        Result.emplace_back(Register);
    for (auto Register : abi::Trait<A>::VectorArgumentRegisters)
      if (Lookup.count(Register) != 0)
        Result.emplace_back(Register);

    revng_assert(Result.size() == std::size(Registers));
    return Result;
  });
}

template<ranges::sized_range Container>
llvm::SmallVector<model::Register::Values, 8>
orderReturnValues(const Container &Registers, model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&Registers]<model::ABI::Values A>() {
    SortedVector<model::Register::Values> Lookup;
    {
      auto Inserter = Lookup.batch_insert();
      for (auto &&Register : Registers)
        Inserter.insert(Register);
    }

    llvm::SmallVector<model::Register::Values, 8> Result;
    for (auto Register : abi::Trait<A>::GeneralPurposeReturnValueRegisters)
      if (Lookup.count(Register) != 0)
        Result.emplace_back(Register);
    for (auto Register : abi::Trait<A>::VectorReturnValueRegisters)
      if (Lookup.count(Register) != 0)
        Result.emplace_back(Register);

    revng_assert(Result.size() == std::size(Registers));
    return Result;
  });
}

} // namespace abi
