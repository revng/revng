/// \file ConvertFunctionsToCABI.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Pipeline/Analysis.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Support/IRHelpers.h"
#include "revng/TupleTree/TupleTree.h"

// TODO: Stop using VerifyHelper for this verification.
//       Introduce a new class instead.

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH,
                           const model::TypeDefinition &Type);

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH,
                           const model::PrimitiveKind::Values &Kind) {
  rc_return VH.maybeFail(Kind != model::PrimitiveKind::Float,
                         "Floating Point primitive found.");
}

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH, const model::Type &Type) {
  if (const auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    // Arrays have no impact on the type, so just unwrap them.
    rc_return rc_recur checkVectorRegisterSupport(VH, *Array->ElementType());

  } else if (const auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    // Defined types need special processing, so unwrap them.
    rc_return rc_recur checkVectorRegisterSupport(VH, D->unwrap());

  } else if (const auto *P = llvm::dyn_cast<model::PointerType>(&Type)) {
    // If it's a pointer, it's acceptable no matter what it points to.
    rc_return true;

  } else if (const auto *P = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    // For primitives it depends purely on the kind.
    rc_return rc_recur checkVectorRegisterSupport(VH, P->PrimitiveKind());

  } else {
    revng_abort("Unsupported type.");
  }
}

template<typename RealType>
inline RecursiveCoroutine<bool>
underlyingHelper(model::VerifyHelper &VH, const model::TypeDefinition &Value) {
  const RealType &Cast = llvm::cast<RealType>(Value);
  rc_return rc_recur checkVectorRegisterSupport(VH, *Cast.UnderlyingType());
}

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH,
                           const model::TypeDefinition &Type) {
  if (VH.isVerified(Type))
    rc_return true;

  // Ensure we never recur indefinitely
  if (VH.isVerificationInProgress(Type))
    rc_return VH.fail();

  VH.verificationInProgress(Type);

  bool Result = false;

  switch (Type.Kind()) {
  case model::TypeDefinitionKind::EnumDefinition:
    Result = rc_recur underlyingHelper<model::EnumDefinition>(VH, Type);
    break;

  case model::TypeDefinitionKind::TypedefDefinition:
    Result = rc_recur underlyingHelper<model::TypedefDefinition>(VH, Type);
    break;

  case model::TypeDefinitionKind::StructDefinition:
    Result = true;
    for (const auto &F : llvm::cast<model::StructDefinition>(Type).Fields())
      Result = Result && rc_recur checkVectorRegisterSupport(VH, *F.Type());
    break;

  case model::TypeDefinitionKind::UnionDefinition:
    Result = true;
    for (const auto &F : llvm::cast<model::UnionDefinition>(Type).Fields())
      Result = Result && rc_recur checkVectorRegisterSupport(VH, *F.Type());
    break;

  case model::TypeDefinitionKind::CABIFunctionDefinition: {
    Result = true;
    using CABIFT = model::CABIFunctionDefinition;
    for (const auto &A : llvm::cast<CABIFT>(Type).Arguments())
      Result = Result && rc_recur checkVectorRegisterSupport(VH, *A.Type());
    const auto &ReturnType = llvm::cast<CABIFT>(Type).ReturnType();
    Result = Result && rc_recur checkVectorRegisterSupport(VH, *ReturnType);
  } break;

  case model::TypeDefinitionKind::RawFunctionDefinition: {
    Result = true;
    using RawFT = model::RawFunctionDefinition;
    for (const auto &A : llvm::cast<RawFT>(Type).Arguments()) {
      auto Kind = model::Register::primitiveKind(A.Location());
      Result = Result && rc_recur checkVectorRegisterSupport(VH, Kind);
      Result = Result && rc_recur checkVectorRegisterSupport(VH, *A.Type());
    }
    for (const auto &V : llvm::cast<RawFT>(Type).ReturnValues()) {
      auto Kind = model::Register::primitiveKind(V.Location());
      Result = Result && rc_recur checkVectorRegisterSupport(VH, Kind);
      Result = Result && rc_recur checkVectorRegisterSupport(VH, *V.Type());
    }

    const auto &Stack = llvm::cast<RawFT>(Type).StackArgumentsType();
    if (not Stack.isEmpty())
      Result = Result && rc_recur checkVectorRegisterSupport(VH, *Stack);
  } break;

  default:
    revng_abort("Unknown type.");
  }

  if (Result) {
    VH.setVerified(Type);
    VH.verificationCompleted(Type);
  }

  rc_return VH.maybeFail(Result);
}

using namespace std::string_literals;

static Logger Log("function-type-conversion-to-cabi-analysis");

class ConvertFunctionsToCABI {
public:
  static constexpr auto Name = "convert-functions-to-cabi";
  inline static const std::tuple Options = {
    // Allows overriding the default ABI with a specific value when invoking
    // the analysis.
    pipeline::Option("abi", "Invalid"),

    // Allows specifying the mode of operation,
    // - safe: only convert the function if ABI belongs to the "tested" list.
    // - unsafe: always convert the function.
    pipeline::Option("mode", "safe"),

    // Allows specifying the confidence we have in the ABI, which then leads to
    // different levels of strictness when doing the argument deductions
    // (different behaviour in cases where the function does not seem to comply
    // to the abi):
    // - low: use safe deduction that will avoid changing function in cases of
    //        non-compliance.
    // - high: override/discard any information about the function that does not
    //         comply with an ABI (i.e. an argument in a register that is not
    //         dedicated for passing arguments, etc.).
    pipeline::Option("confidence", "low")
  };
  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  void run(pipeline::ExecutionContext &Context,
           std::string TargetABI,
           std::string Mode,
           std::string ABIConfidence) {
    auto &Model = revng::getWritableModelFromContext(Context);
    revng_assert(!TargetABI.empty());

    model::ABI::Values ABI = model::ABI::fromName(TargetABI);
    if (ABI == model::ABI::Values::Invalid) {
      revng_log(Log,
                "No ABI explicitly specified for the conversion, using the "
                "`Model->DefaultABI()`.");
      ABI = Model->DefaultABI();
    }

    // Minimize the negative impact on binaries with ABI that is not fully
    // supported by disabling the conversion by default.
    //
    // Use `--convert-functions-to-cabi-mode=unsafe` to force conversion even
    // when ABI is not considered fully tested.
    if (Mode != "safe") {
      // TODO: extend this list.
      static constexpr std::array ABIsTheConversionIsEnabledFor = {
        model::ABI::SystemV_x86_64,
        model::ABI::Microsoft_x86_64,
        model::ABI::Microsoft_x86_64_vectorcall,
        model::ABI::SystemV_x86,
        model::ABI::SystemV_x86_regparm_3,
        model::ABI::SystemV_x86_regparm_2,
        model::ABI::SystemV_x86_regparm_1,
        model::ABI::Microsoft_x86_cdecl,
        model::ABI::Microsoft_x86_cdecl_gcc,
        model::ABI::Microsoft_x86_fastcall,
        model::ABI::Microsoft_x86_fastcall_gcc,
        model::ABI::Microsoft_x86_stdcall,
        model::ABI::Microsoft_x86_stdcall_gcc,
        model::ABI::Microsoft_x86_thiscall,
        model::ABI::Microsoft_x86_vectorcall,
        model::ABI::AAPCS

        // There are known issues
        // model::ABI::AAPCS64,

        // There are known issues
        // model::ABI::SystemV_MIPS_o32,
        // model::ABI::SystemV_MIPSEL_o32

        // Unable to reliably test: no easy access to a compiler
        // model::ABI::Pascal_x86,

        // Unable to reliably test: QEMU aborts
        // model::ABI::SystemZ_s390x,
      };
      if (!llvm::is_contained(ABIsTheConversionIsEnabledFor, ABI)) {
        revng_log(Log,
                  "Analysis was aborted because the `safe` (default) mode of "
                  "the conversion was selected and the conversion for the "
                  "current ABI ('"
                    << model::ABI::getName(ABI).str()
                    << "') is not considered stable.");
        return;
      }
    }

    // Determines the strictness of register state deductions
    bool SoftDeductions = (ABIConfidence == "low");

    // This reuses the verification map within the `model::VerifyHelper` in
    // an incompatible manner. DO NOT pass this object into a normal
    // verification routine or things are going to break down.
    model::VerifyHelper VectorVH;

    // Choose functions for the conversion, but leave DefaultPrototype alone
    // (in order to avoid invalidation),
    using abi::FunctionType::filterTypes;
    using RawFD = model::RawFunctionDefinition;
    // TODO: Switch to std::optional after C++26 support arrives.
    llvm::SmallVector<model::TypeDefinition *, 1> TypesToIgnore = {};
    if (auto &DefaultPrototype = Model->DefaultPrototype())
      if (auto *Definition = DefaultPrototype->tryGetAsDefinition())
        TypesToIgnore.push_back(Definition);
    auto ToConvert = filterTypes<RawFD>(Model->TypeDefinitions(),
                                        TypesToIgnore);

    // And convert them.
    for (model::RawFunctionDefinition *Old : ToConvert) {
      auto &DT = llvm::cast<model::DefinedType>(*Model->makeType(Old->key()));
      if (!checkVectorRegisterSupport(VectorVH, *Old)) {
        // TODO: remove this check after `abi::FunctionType` supports vectors.
        revng_log(Log,
                  "Skip a function conversion because it requires vector type "
                  "support: "
                    << serializeToString(DT.Definition()));
        continue;
      }

      revng_log(Log,
                "Converting a function: "
                  << serializeToString(DT.Definition()));
      if (Log.isEnabled()) {
        std::string Message = "";
        for (model::Function &Function : Model->Functions())
          if (Function.prototype() && Function.prototype()->key() == Old->key())
            Message += "'" + Function.name().str().str() + "', ";

        if (!Message.empty()) {
          Message.resize(Message.size() - 2);
          revng_log(Log, "It's a prototype of " << Message);
        }
      }

      namespace FT = abi::FunctionType;
      if (auto New = FT::tryConvertToCABI(*Old, Model, ABI, SoftDeductions)) {
        // If the conversion succeeds, make sure the returned type is valid,
        revng_assert(!New->isEmpty());

        // and verifies
        if (VerifyLog.isEnabled())
          New->get()->verify(true);

        revng_log(Log,
                  "Function Conversion Successful: "
                    << serializeToString(*New));
      } else {
        // Do nothing if the conversion failed (the model is not modified).
        // `RawFunctionDefinition` is still used for those functions.
        // This might be an indication of an ABI misdetection.
        revng_log(Log, "Function Conversion Failed.");
      }
    }

    // Don't forget to clean up any possible remainders of removed types.
    purgeUnnamedAndUnreachableTypes(Model);
  }
};

pipeline::RegisterAnalysis<ConvertFunctionsToCABI> ToCABIAnalysis;
