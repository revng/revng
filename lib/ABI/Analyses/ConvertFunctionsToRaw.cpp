//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/Analyses/ConvertFunctionsToRaw.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/TupleTree/TupleTree.h"

namespace revng::pypeline::analyses {

llvm::Error ConvertFunctionsToRaw::run(Model &TheModel,
                                       const Request &Incoming,
                                       llvm::StringRef Configuration) {
  auto &Model = TheModel.get();

  model::VerifyHelper VH;

  using abi::FunctionType::filterTypes;
  using CABIFD = model::CABIFunctionDefinition;
  auto ToConvert = filterTypes<CABIFD>(Model->TypeDefinitions());
  for (model::CABIFunctionDefinition *Old : ToConvert) {
    model::UpcastableType New = abi::FunctionType::convertToRaw(*Old, Model);
    revng_assert(!New.isEmpty());
    revng_assert(New->verify(VH));
  }

  // Don't forget to clean up any possible remainders of removed types.
  purgeUnnamedAndUnreachableTypes(Model);

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
