/// \file Verify.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <string>

#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Pass/Verify.h"

using namespace llvm;
using namespace model;

static RegisterModelPass R("verify",
                           "Verifies that there are no serious issues with the "
                           "model",
                           model::verify);

void model::verify(TupleTree<model::Binary> &Model) {
  Model->verify(true);
}
