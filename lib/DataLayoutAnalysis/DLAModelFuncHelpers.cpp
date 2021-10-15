//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "DLAModelFuncHelpers.h"

using namespace dla;
using namespace llvm;

static constexpr const char *const IsolatedFunctioMD = "revng.function.entry";
static constexpr const char *const IndirectCallMD = "revng.callerblock.start";

static MetaAddress getMetaAddress(const MDNode *MD) {
  if (const auto *MDT = dyn_cast_or_null<MDTuple>(MD))
    if (const auto *V = dyn_cast<ValueAsMetadata>(MDT->getOperand(0)))
      return MetaAddress::fromConstant(V->getValue());
  return MetaAddress::invalid();
}

MetaAddress dla::getMetaAddress(const Function *F) {
  return ::getMetaAddress(F->getMetadata(IsolatedFunctioMD));
}

MetaAddress dla::getMetaAddress(const CallInst *C) {
  return ::getMetaAddress(C->getMetadata(IndirectCallMD));
}

model::Type *
dla::getIndirectCallPrototype(const CallInst *C, const model::Binary &Model) {

  const auto &FuncMetaAddr = getMetaAddress(C->getParent()->getParent());
  const auto &BBMetaAddr = getMetaAddress(C);

  if (not FuncMetaAddr.isValid() or not BBMetaAddr.isValid())
    return nullptr;

  const auto *ModelBB = &Model.Functions.at(FuncMetaAddr).CFG.at(BBMetaAddr);

  // The basic block containing the indirect call is expected to have exactly
  // one call edge.
  model::CallEdge *ModelCallEdge = nullptr;
  for (auto &Succ : ModelBB->Successors) {
    if (auto *E = dyn_cast<model::CallEdge>(Succ.get())) {
      revng_assert(ModelCallEdge == nullptr);
      ModelCallEdge = E;
    }
  }
  revng_assert(ModelCallEdge);

  return ModelCallEdge->Prototype.get();
}

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
