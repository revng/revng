/// \file ConvertFunctionTypes.cpp
/// \brief Implementation of bulk function type conversion.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/Model/ConvertFunctionType.h"
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

static Logger Log("model/Pass/ConvertFunctionTypes");

static void replaceReferences(const model::Type::Key &OldKey,
                              const model::TypePath &NewTypePath,
                              TupleTree<model::Binary> &Model) {
  auto Visitor = [&](model::TypePath &Visited) {
    if (!Visited.isValid())
      return; // Ignore empty references

    model::Type *Current = Visited.get();
    revng_assert(Current != nullptr);

    if (Current->key() == OldKey)
      Visited = NewTypePath;
  };
  Model.visitReferences(Visitor);
  Model->Types.erase(OldKey);
}

template<DerivesFrom<model::Type> DerivedType>
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

  auto ToConvert = chooseTypes<model::CABIFunctionType>(Model->Types);
  for (model::CABIFunctionType *Old : ToConvert) {
    if (auto New = model::convertToRawFunctionType(*Old, *Model)) {
      // Add converted type to the model.
      auto Ptr = model::UpcastableType::make<model::RawFunctionType>(*New);
      auto NewTypePath = Model->recordNewType(std::move(Ptr));

      // Replace all references to the old type with references to the new one.
      replaceReferences(Old->key(), NewTypePath, Model);
    }
  }

  if (!Model.verify() || !Model->verify())
    revng_log(Log,
              "While trying to convert all the function to `RawFunctionType`, "
              "result model verification has failed.\n");
}

void model::convertAllFunctionsToCABI(TupleTree<model::Binary> &Model,
                                      model::ABI::Values TargetABI) {
  if (!Model.verify() || !Model->verify()) {
    revng_log(Log,
              "While trying to convert all the function to `CABIFunctionType`, "
              "input model verification has failed.\n");
    return;
  }

  auto ToConvert = chooseTypes<model::RawFunctionType>(Model->Types);
  for (model::RawFunctionType *Old : ToConvert) {
    if (auto New = model::convertToCABIFunctionType(*Old, *Model, TargetABI)) {
      // Verify the return type.
      auto *ReturnValueType = New->ReturnType.UnqualifiedType.get();
      auto TypeIterator = Model->Types.find(ReturnValueType->key());
      revng_assert(TypeIterator != Model->Types.end());

      // Add converted type to the model.
      auto Ptr = model::UpcastableType::make<model::CABIFunctionType>(*New);
      auto NewTypePath = Model->recordNewType(std::move(Ptr));

      // Replace all references to the old type with references to the new one.
      replaceReferences(Old->key(), NewTypePath, Model);
    }
  }

  if (!Model.verify() || !Model->verify())
    revng_log(Log,
              "While trying to convert all the function to `CABIFunctionType`, "
              "result model verification has failed.\n");
}
