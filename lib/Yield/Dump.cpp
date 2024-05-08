/// \file Dump.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Yield/BasicBlock.h"
#include "revng/Yield/CallEdge.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Instruction.h"
#include "revng/Yield/TaggedString.h"

void yield::TaggedString::dump() const {
  serialize(dbg, *this);
}

void yield::TaggedLine::dump() const {
  serialize(dbg, *this);
}

void yield::Instruction::dump() const {
  serialize(dbg, *this);
}

void yield::BasicBlock::dump() const {
  serialize(dbg, *this);
}

void yield::Function::dump() const {
  serialize(dbg, *this);
}
