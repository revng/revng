/// \file ConvertToRawFunctionType.cpp
/// \brief

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

class ConvertToRawFunctionType {
public:
  static constexpr auto Name = "ConvertToRawFunctionType";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  void run(pipeline::Context &Context) {
    auto &Model = revng::getWritableModelFromContext(Context);

    model::VerifyHelper VH;

    using abi::FunctionType::filterTypes;
    auto ToConvert = filterTypes<model::CABIFunctionType>(Model->Types());
    for (model::CABIFunctionType *Old : ToConvert) {
      model::TypePath New = abi::FunctionType::convertToRaw(*Old, Model);

      // Make sure the returned type is valid,
      revng_assert(New.isValid());

      // and verifies.
      revng_assert(New.get()->verify(VH));
    }

    // Don't forget to clean up any possible remainders of removed types.
    purgeUnnamedAndUnreachableTypes(Model);
  }
};

static pipeline::RegisterAnalysis<ConvertToRawFunctionType> ToRawAnalysis;
