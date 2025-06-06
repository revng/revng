#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <limits>
#include <string>

#include "revng/ADT/SortedVector.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Yield/TagType.h"

namespace yield {
using ByteContainer = llvm::SmallVector<uint8_t, 16>;
}

#include "revng/Yield/Generated/Early/Instruction.h"

namespace model {
class VerifyHelper;
}

namespace yield {

class Instruction : public generated::Instruction {
public:
  using generated::Instruction::Instruction;

public:
  struct RawTag {
    yield::TagType::Values Type;
    uint64_t From;
    uint64_t To;
  };
  void importTags(std::vector<RawTag> &&Tags, std::string &&Contents);
  void handleSpecialTags(const yield::BasicBlock &BasicBlock,
                         const yield::Function &Function,
                         const model::Binary &Binary,
                         const model::AssemblyNameBuilder &NameBuilder);

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;

public:
  inline MetaAddress getRelativeAddressBase() const {
    switch (Address().type()) {
    case MetaAddressType::Code_x86:
    case MetaAddressType::Code_x86_64:
      return Address() + RawBytes().size(); // Start of the next instruction.
    case MetaAddressType::Code_arm:
      return Address() + 8; // Two instructions later
    case MetaAddressType::Code_arm_thumb:
      return Address() + 4; // Two instructions later
    case MetaAddressType::Code_aarch64:
    case MetaAddressType::Code_systemz:
    case MetaAddressType::Code_mips:
    case MetaAddressType::Code_mipsel:
      return Address();
    default:
      revng_abort("Unsupported instruction.");
    }
  }
};

} // namespace yield

#include "revng/Yield/Generated/Late/Instruction.h"
