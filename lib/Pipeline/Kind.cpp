/// \file Kind.cpp
/// A kind is a object associated to target do distinguish semantically
/// different targets that may share the same components path.

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//
#include <vector>

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"

using namespace pipeline;

llvm::Error Kind::verify(const ContainerBase &Container,
                         const Target &T) const {
  return llvm::Error::success();
}

void SingleElementKind::appendAllTargets(const Context &Ctx,
                                         TargetsList &Out) const {
  Out.push_back(Target(*this));
}

TargetsList Kind::allTargets(const Context &Ctx) const {
  TargetsList Out;
  appendAllTargets(Ctx, Out);
  return Out;
}
