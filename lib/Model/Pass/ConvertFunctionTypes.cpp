/// \file ConvertFunctionTypes.cpp
/// \brief Implementation of bulk function type conversion.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Pass/ConvertFunctionTypes.h"
#include "revng/Model/Pass/RegisterModelPass.h"

extern llvm::cl::OptionCategory ModelPassCategory;

template<std::size_t... Indices>
static auto packValues(std::integer_sequence<size_t, Indices...>) {
  using namespace llvm::cl;
  using namespace model::ABI;
  return values(OptionEnumValue{ getName(Values(Indices)),
                                 int(Indices),
                                 getDescription(Values(Indices)) }...);
}

constexpr const char *Description = "Overrides default ABI deduced from the "
                                    "binary.";
constexpr auto ABIList = std::make_index_sequence<model::ABI::Count - 1>{};
static auto Values = packValues(ABIList);

llvm::cl::opt<model::ABI::Values> TargetABI("abi",
                                            Values,
                                            llvm::cl::desc(Description),
                                            llvm::cl::cat(ModelPassCategory));

static void convertAllFunctionsToCABIImpl(TupleTree<model::Binary> &Model) {
  if (TargetABI)
    model::convertAllFunctionsToCABI(Model, TargetABI);
  else
    model::convertAllFunctionsToCABI(Model);
}

static RegisterModelPass ToRaw("convert-all-cabi-functions-to-raw",
                               "Converts as many `CABIFunctionType`s into "
                               "`RawFunctionType` as it possible",
                               model::convertAllFunctionsToRaw);
static RegisterModelPass ToCABI("convert-all-raw-functions-to-cabi",
                                "Converts as many `RawFunctionType`s into "
                                "`CABIFunctionType` as it possible",
                                convertAllFunctionsToCABIImpl);

static Logger Log("convert-function-types");

template<derived_from<model::Type> DerivedType>
static std::vector<DerivedType *>
chooseTypes(SortedVector<UpcastablePointer<model::Type>> &Types) {
  std::vector<DerivedType *> Result;
  for (model::UpcastableType &Type : Types)
    if (auto *Upscaled = llvm::dyn_cast<DerivedType>(Type.get()))
      Result.emplace_back(Upscaled);
  return Result;
}

void model::convertAllFunctionsToRaw(TupleTree<model::Binary> &Model) {
  if (!Model.verify() || !Model->verify()) {
    revng_log(Log,
              "While trying to convert all the function to `RawFunctionType`, "
              "input model verification has failed.\n");
    return;
  }

  auto ToConvert = chooseTypes<model::CABIFunctionType>(Model->Types());
  for (model::CABIFunctionType *Old : ToConvert) {
    auto New = abi::FunctionType::convertToRaw(*Old, Model);
    revng_assert(New.isValid());
  }

  if (!Model.verify() || !Model->verify())
    revng_log(Log,
              "While trying to convert all the function to `RawFunctionType`, "
              "result model verification has failed.\n");
}

void model::convertAllFunctionsToCABI(TupleTree<model::Binary> &Model,
                                      model::ABI::Values ABI) {
  if (!Model.verify() || !Model->verify()) {
    revng_log(Log,
              "While trying to convert all the function to `CABIFunctionType`, "
              "input model verification has failed.\n");
    return;
  }

  auto ToConvert = chooseTypes<model::RawFunctionType>(Model->Types());
  for (model::RawFunctionType *Old : ToConvert)
    if (auto New = abi::FunctionType::tryConvertToCABI(*Old, Model, ABI))
      revng_assert(New->isValid());

  if (!Model.verify() || !Model->verify())
    revng_log(Log,
              "While trying to convert all the function to `CABIFunctionType`, "
              "result model verification has failed.\n");
}
