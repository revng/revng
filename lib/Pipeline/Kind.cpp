/// \file Kind.cpp
/// \brief A kind is a object associated to target do distinguish semantically
/// different targets that may share the same components path

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <vector>

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"

using namespace Pipeline;

void Kind::expandTarget(const Context &Ctx,
                        const Target &Input,
                        TargetsList &Output) const {
  Output.emplace_back(Input);
}
