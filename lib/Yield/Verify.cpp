/// \file Verify.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/VerifyHelper.h"
#include "revng/Yield/BasicBlock.h"
#include "revng/Yield/CallEdge.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Instruction.h"

bool yield::TaggedString::verify(model::VerifyHelper &VH) const {
  if (Type() == TagType::Invalid)
    return VH.fail("The type of this tag is not valid.");
  if (Content().empty())
    return VH.fail("This tag doesn't have any data.");
  if (Content().find('\n') != std::string::npos)
    return VH.fail("String is not allowed to break the line, "
                   "use multiple strings instead.");
  for (const yield::TagAttribute &Attribute : Attributes()) {
    if (Attribute.Name().empty())
      return VH.fail("Attributes without names are not allowed.");
    if (Attribute.Name().empty())
      return VH.fail("Attributes without values are not allowed.");
  }

  return true;
}

bool yield::TaggedLine::verify(model::VerifyHelper &VH) const {
  for (auto [Index, String] : llvm::enumerate(Tags())) {
    if (Index != String.Index())
      return VH.fail("Tagged string indexing is broken.");

    if (!String.verify(VH))
      return VH.fail();
  }

  return true;
}

bool yield::Instruction::verify(model::VerifyHelper &VH) const {
  if (Address().isInvalid())
    return VH.fail("An instruction must have a valid address.");
  if (RawBytes().empty())
    return VH.fail("An instruction has to be at least one byte big.");

  if (Disassembled().empty())
    return VH.fail("An instruction must have at least one tag.");
  for (auto [Index, Tagged] : llvm::enumerate(Disassembled())) {
    if (Index != Tagged.Index())
      return VH.fail("Tagged string indexing is broken.");

    if (!Tagged.verify(VH))
      return VH.fail();
  }

  for (auto [Index, Directive] : llvm::enumerate(PrecedingDirectives())) {
    if (Index != Directive.Index())
      return VH.fail("Preceding directive indexing is broken.");

    if (!Directive.verify(VH))
      return VH.fail();
  }
  for (auto [Index, Directive] : llvm::enumerate(FollowingDirectives())) {
    if (Index != Directive.Index())
      return VH.fail("Following directive indexing is broken.");

    if (!Directive.verify(VH))
      return VH.fail();
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

  if (IsLabelAlwaysRequired())
    if (!Label().verify())
      return false;

  MetaAddress PreviousAddress = MetaAddress::invalid();
  for (const auto &Instruction : Instructions()) {
    if (!Instruction.verify(VH))
      return VH.fail("Instruction verification failed.");

    if (PreviousAddress.isValid() && Instruction.Address() < PreviousAddress) {
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

  for (const auto &BasicBlock : Blocks())
    if (!BasicBlock.verify(VH))
      return VH.fail("Basic block verification failed.");

  return true;
}

bool yield::TaggedString::verify() const {
  return verify(false);
}
bool yield::TaggedString::verify(bool Assert) const {
  model::VerifyHelper VH(Assert);
  return verify(VH);
}

bool yield::TaggedLine::verify() const {
  return verify(false);
}
bool yield::TaggedLine::verify(bool Assert) const {
  model::VerifyHelper VH(Assert);
  return verify(VH);
}

bool yield::Instruction::verify() const {
  return verify(false);
}
bool yield::Instruction::verify(bool Assert) const {
  model::VerifyHelper VH(Assert);
  return verify(VH);
}

bool yield::BasicBlock::verify() const {
  return verify(false);
}
bool yield::BasicBlock::verify(bool Assert) const {
  model::VerifyHelper VH(Assert);
  return verify(VH);
}

bool yield::Function::verify() const {
  return verify(false);
}
bool yield::Function::verify(bool Assert) const {
  model::VerifyHelper VH(Assert);
  return verify(VH);
}
