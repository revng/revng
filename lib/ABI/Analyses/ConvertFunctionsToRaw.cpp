/// \file ConvertFunctionsToRaw.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Pipeline/Analysis.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/TupleTree/TupleTree.h"

class ConvertFunctionsToRaw {
public:
  static constexpr auto Name = "convert-functions-to-raw";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  void run(pipeline::ExecutionContext &Context) {
    auto &Model = revng::getWritableModelFromContext(Context);

    model::VerifyHelper VH;

    using abi::FunctionType::filterTypes;
    using CABIFD = model::CABIFunctionDefinition;
    auto ToConvert = filterTypes<CABIFD>(Model->TypeDefinitions());
    for (model::CABIFunctionDefinition *Old : ToConvert) {
      model::TypeDefinitionPath New = abi::FunctionType::convertToRaw(*Old,
                                                                      Model);

      // Make sure the returned type is valid,
      revng_assert(New.isValid());

      // and verifies.
      revng_assert(New.get()->verify(VH));
    }

    // Don't forget to clean up any possible remainders of removed types.
    purgeUnnamedAndUnreachableTypes(Model);
  }
};

static pipeline::RegisterAnalysis<ConvertFunctionsToRaw> ToRawAnalysis;
