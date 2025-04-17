/// \file FixModel.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Pass/AllPasses.h"
#include "revng/Model/Pass/RegisterModelPass.h"

namespace model {

// TODO: should this be exposed in a header to be used from c++?
static void fixModel(TupleTree<model::Binary> &Model) {
  purgeInvalidTypes(Model);
  deduplicateEquivalentTypes(Model);
  flattenPrimitiveTypedefs(Model);
  deduplicateCollidingNames(Model);
  purgeUnnamedAndUnreachableTypes(Model);
}

} // namespace model

static RegisterModelPass R("fix-model",
                           "This is a combination of multiple other passes to "
                           "fix some common problems introduced to the model "
                           "by the importers. If you're writing your own "
                           "importer, you might want to run this (or at least "
                           "some of the separate passes it uses) to help "
                           "ensure your importer produces valid models.",
                           model::fixModel);
