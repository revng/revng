//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "FuncOrCallInst.h"

using namespace dla;
using namespace llvm;

bool FuncOrCallInst::isNull() const {
  if (holds_alternative<const Function *>(Val))
    return get<const Function *>(Val) == nullptr;
  if (holds_alternative<const CallInst *>(Val))
    return get<const CallInst *>(Val) == nullptr;
  revng_abort("Can only be Function or CallInst");
}

const Value *FuncOrCallInst::getVal() const {
  if (holds_alternative<const Function *>(Val))
    return static_cast<const Value *>(get<const Function *>(Val));
  if (holds_alternative<const CallInst *>(Val))
    return static_cast<const Value *>(get<const CallInst *>(Val));
  revng_abort("Can only be Function or CallInst");
}

const Type *FuncOrCallInst::getRetType() const {
  if (holds_alternative<const Function *>(Val))
    return dla::getRetType(get<const Function *>(Val));
  if (holds_alternative<const CallInst *>(Val))
    return dla::getRetType(get<const CallInst *>(Val));
  revng_abort("Can only be Function or CallInst");
}

unsigned long FuncOrCallInst::arg_size() const {
  if (holds_alternative<const Function *>(Val))
    return dla::arg_size(get<const Function *>(Val));
  if (holds_alternative<const CallInst *>(Val))
    return dla::arg_size(get<const CallInst *>(Val));
  revng_abort("Can only be Function or CallInst");
}

const Value *FuncOrCallInst::getArg(unsigned Idx) const {
  if (holds_alternative<const Function *>(Val))
    return dla::getArgs(get<const Function *>(Val)).begin() + Idx;
  if (holds_alternative<const CallInst *>(Val))
    return *(dla::getArgs(get<const CallInst *>(Val)).begin() + Idx);
  revng_abort("Can only be Function or CallInst");
}
