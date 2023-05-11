/// \file Verify.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/BasicBlock.h"
#include "revng/Yield/CallEdge.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Instruction.h"
#include "revng/Yield/Tag.h"

bool yield::Tag::verify(model::VerifyHelper &VH) const {
  if (Type() == TagType::Invalid)
    return VH.fail("The type of this tag is not valid.");
  if (From() == std::string::npos)
    return VH.fail("This tag doesn't have a starting point.");
  if (To() == std::string::npos)
    return VH.fail("This tag doesn't have an ending point.");
  if (From() >= To())
    return VH.fail("This tag doesn't have a positive length.");

  return true;
}

bool yield::Instruction::verify(model::VerifyHelper &VH) const {
  if (Address().isInvalid())
    return VH.fail("An instruction has to have a valid address.");
  if (Disassembled().empty())
    return VH.fail("The disassembled view of an instruction cannot be empty.");
  if (RawBytes().empty())
    return VH.fail("An instruction has to be at least one byte big.");

  for (const auto &Tag : Tags()) {
    if (!Tag.verify(VH))
      return VH.fail("Tag verification failed");

    if (Tag.From() >= Disassembled().size()
        || Tag.To() >= Disassembled().size())
      return VH.fail("Tag boundaries must not exceed the size of the text.");
  }

  return true;
}

bool yield::BasicBlock::verify(model::VerifyHelper &VH) const {
  if (not ID().isValid())
    return VH.fail("A basic block has to have a valid start address.");
  if (End().isInvalid())
    return VH.fail("A basic block has to have a valid end address.");
  if (Instructions().empty())
    return VH.fail("A basic block has to store at least a single instruction.");

  MetaAddress PreviousAddress = MetaAddress::invalid();
  for (const auto &Instruction : Instructions()) {
    if (!Instruction.verify(VH))
      return VH.fail("Instruction verification failed.");

    if (PreviousAddress.isValid() && Instruction.Address() >= PreviousAddress) {
      return VH.fail("Instructions must be strongly ordered and their size "
                     "must be bigger than zero.");
    }
    PreviousAddress = Instruction.Address();
  }

  if (PreviousAddress.isInvalid() || PreviousAddress >= End()) {
    return VH.fail("The size of the last instruction must be bigger than "
                   "zero.");
  }

  if (HasDelaySlot() && Instructions().size() < 2) {
    return VH.fail("A basic block with a delay slot must contain at least two "
                   "instructions.");
  }

  return true;
}

bool yield::Function::verify(model::VerifyHelper &VH) const {
  if (Entry().isInvalid())
    return VH.fail("A function has to have a valid entry point.");

  if (ControlFlowGraph().empty())
    return VH.fail("A function has to store at least a single basic block.");

  for (const auto &BasicBlock : ControlFlowGraph())
    if (!BasicBlock.verify(VH))
      return VH.fail("Basic block verification failed.");

  return true;
}
