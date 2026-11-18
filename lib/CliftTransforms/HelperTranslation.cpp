//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTHELPERTRANSLATION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

// Represents the kind of transformation applied to a QEMU helper parameter.
enum class HelperTypeKind : uint8_t {
  // TODO: Using `std::inplace_vector` (C++26) or equivalent would eliminate the
  //       need for the default none value.

  // No such parameter exists (default value).
  None,

  // Parameter is discarded (along with any initializing expressions).
  Discard,

  // Parameter is converted to a signed integer.
  Signed,

  // Parameter is converted to an unsigned integer.
  Unsigned,

  // Parameter is converted to floating point.
  Float,
};

// Represents the transformed type of a QEMU helper.
// Zero size specifies that only the kind is changed.
struct HelperType {
  HelperTypeKind Kind;
  uint8_t Size;
};

struct HelperDescriptor {
  std::string_view Identifier;
  HelperType ReturnType;
  HelperType ParameterTypes[5];
};

static constexpr HelperType Discard = { HelperTypeKind::Discard, 0 };

static constexpr HelperType Sxx = { HelperTypeKind::Signed, 0 };
static constexpr HelperType S16 = { HelperTypeKind::Signed, 2 };
static constexpr HelperType S32 = { HelperTypeKind::Signed, 4 };
static constexpr HelperType S64 = { HelperTypeKind::Signed, 8 };

static constexpr HelperType Uxx = { HelperTypeKind::Unsigned, 0 };
static constexpr HelperType U16 = { HelperTypeKind::Unsigned, 2 };
static constexpr HelperType U32 = { HelperTypeKind::Unsigned, 4 };
static constexpr HelperType U64 = { HelperTypeKind::Unsigned, 8 };

static constexpr HelperType F32 = { HelperTypeKind::Float, 4 };
static constexpr HelperType F64 = { HelperTypeKind::Float, 8 };

// Booleans are treated as unsigned. Using an explicit name is useful as a
// documentation device, however.
static constexpr HelperType Bool = Uxx;

// List of translated QEMU helper functions and the typing applied to them.
// NOTE: The helpers must be listed in alphabetical order.
static constexpr HelperDescriptor QEMUHelpers[] = {
  { "float32_abs", F32, { F32 } },
  { "float32_add", F32, { F32, F32, Discard } },
  { "float32_chs", F32, { F32 } },
  { "float32_compare", Sxx, { F32, F32, Discard } },
  { "float32_compare_quiet", Sxx, { F32, F32, Discard } },
  { "float32_default_nan", F32, { Discard } },
  { "float32_div", F32, { F32, F32, Discard } },
  { "float32_eq", Bool, { F32, F32, Discard } },
  { "float32_eq_quiet", Bool, { F32, F32, Discard } },
  { "float32_exp2", F32, { F32, Discard } },
  { "float32_is_any_nan", Bool, { F32 } },
  { "float32_is_denormal", Bool, { F32 } },
  { "float32_is_infinity", Bool, { F32 } },
  { "float32_is_neg", Bool, { F32 } },
  { "float32_is_normal", Bool, { F32 } },
  { "float32_is_quiet_nan", Bool, { F32, Discard } },
  { "float32_is_signaling_nan", Bool, { F32, Discard } },
  { "float32_is_zero", Bool, { F32 } },
  { "float32_is_zero_or_denormal", Bool, { F32 } },
  { "float32_is_zero_or_normal", Bool, { F32 } },
  { "float32_le", Bool, { F32, F32, Discard } },
  { "float32_le_quiet", Bool, { F32, F32, Discard } },
  { "float32_log2", F32, { F32, Discard } },
  { "float32_lt", Bool, { F32, F32, Discard } },
  { "float32_lt_quiet", Bool, { F32, F32, Discard } },
  { "float32_max", F32, { F32, F32, Discard } },
  { "float32_maximum_number", F32, { F32, F32, Discard } },
  { "float32_maxnum", F32, { F32, F32, Discard } },
  { "float32_maxnummag", F32, { F32, F32, Discard } },
  { "float32_min", F32, { F32, F32, Discard } },
  { "float32_minimum_number", F32, { F32, F32, Discard } },
  { "float32_minnum", F32, { F32, F32, Discard } },
  { "float32_minnummag", F32, { F32, F32, Discard } },
  { "float32_mul", F32, { F32, F32, Discard } },
  { "float32_muladd", F32, { F32, F32, F32, Uxx, Discard } },
  { "float32_rem", F32, { F32, F32, Discard } },
  { "float32_round_to_int", F32, { F32, Discard } },
  { "float32_scalbn", F32, { F32, Sxx, Discard } },
  { "float32_set_sign", F32, { F32, Uxx } },
  { "float32_silence_nan", F32, { F32, Discard } },
  { "float32_sqrt", F32, { F32, Discard } },
  { "float32_squash_input_denormal", F32, { F32, Discard } },
  { "float32_sub", F32, { F32, F32, Discard } },
  { "float32_to_float64", F64, { F32, Discard } },
  { "float32_to_int16", S16, { F32, Discard } },
  { "float32_to_int16_round_to_zero", S16, { F32, Discard } },
  { "float32_to_int16_scalbn", S16, { F32, Discard, Sxx, Discard } },
  { "float32_to_int32", S32, { F32, Discard } },
  { "float32_to_int32_round_to_zero", S32, { F32, Discard } },
  { "float32_to_int32_scalbn", S32, { F32, Discard, Sxx, Discard } },
  { "float32_to_int64", S64, { F32, Discard } },
  { "float32_to_int64_round_to_zero", S64, { F32, Discard } },
  { "float32_to_int64_scalbn", S64, { F32, Discard, Sxx, Discard } },
  { "float32_to_uint16", U16, { F32, Discard } },
  { "float32_to_uint16_round_to_zero", U16, { F32, Discard } },
  { "float32_to_uint16_scalbn", U16, { F32, Discard, Sxx, Discard } },
  { "float32_to_uint32", U32, { F32, Discard } },
  { "float32_to_uint32_round_to_zero", U32, { F32, Discard } },
  { "float32_to_uint32_scalbn", U32, { F32, Discard, Sxx, Discard } },
  { "float32_to_uint64", U64, { F32, Discard } },
  { "float32_to_uint64_round_to_zero", U64, { F32, Discard } },
  { "float32_to_uint64_scalbn", U64, { F32, Discard, Sxx, Discard } },
  { "float32_unordered", Bool, { F32, F32, Discard } },
  { "float32_unordered_quiet", Bool, { F32, F32, Discard } },
  { "float64_abs", F64, { F64 } },
  { "float64_add", F64, { F64, F64, Discard } },
  { "float64_chs", F64, { F64 } },
  { "float64_compare", Sxx, { F64, F64, Discard } },
  { "float64_compare_quiet", Sxx, { F64, F64, Discard } },
  { "float64_default_nan", F64, { Discard } },
  { "float64_div", F64, { F64, F64, Discard } },
  { "float64_eq", Bool, { F64, F64, Discard } },
  { "float64_eq_quiet", Bool, { F64, F64, Discard } },
  { "float64_is_any_nan", Bool, { F64 } },
  { "float64_is_denormal", Bool, { F64 } },
  { "float64_is_infinity", Bool, { F64 } },
  { "float64_is_neg", Bool, { F64 } },
  { "float64_is_normal", Bool, { F64 } },
  { "float64_is_quiet_nan", Bool, { F64, Discard } },
  { "float64_is_signaling_nan", Bool, { F64, Discard } },
  { "float64_is_zero", Bool, { F64 } },
  { "float64_is_zero_or_denormal", Bool, { F64 } },
  { "float64_is_zero_or_normal", Bool, { F64 } },
  { "float64_le", Bool, { F64, F64, Discard } },
  { "float64_le_quiet", Bool, { F64, F64, Discard } },
  { "float64_log2", F64, { F64, Discard } },
  { "float64_lt", Bool, { F64, F64, Discard } },
  { "float64_lt_quiet", Bool, { F64, F64, Discard } },
  { "float64_max", F64, { F64, F64, Discard } },
  { "float64_maximum_number", F64, { F64, F64, Discard } },
  { "float64_maxnum", F64, { F64, F64, Discard } },
  { "float64_maxnummag", F64, { F64, F64, Discard } },
  { "float64_min", F64, { F64, F64, Discard } },
  { "float64_minimum_number", F64, { F64, F64, Discard } },
  { "float64_minnum", F64, { F64, F64, Discard } },
  { "float64_minnummag", F64, { F64, F64, Discard } },
  { "float64_mul", F64, { F64, F64, Discard } },
  { "float64_muladd", F64, { F64, F64, F64, Uxx, Discard } },
  { "float64_rem", F64, { F64, F64, Discard } },
  { "float64_round_to_int", F64, { F64, Discard } },
  { "float64_scalbn", F64, { F64, Sxx, Discard } },
  { "float64_set_sign", F64, { F64, Uxx } },
  { "float64_silence_nan", F64, { F64, Discard } },
  { "float64_sqrt", F64, { F64, Discard } },
  { "float64_squash_input_denormal", F64, { F64, Discard } },
  { "float64_sub", F64, { F64, F64, Discard } },
  { "float64_to_float32", F32, { F64, Discard } },
  { "float64_to_int16", S16, { F64, Discard } },
  { "float64_to_int16_round_to_zero", S16, { F64, Discard } },
  { "float64_to_int16_scalbn", S16, { F64, Discard, Sxx, Discard } },
  { "float64_to_int32", S32, { F64, Discard } },
  { "float64_to_int32_modulo", S32, { F64, Discard, Discard } },
  { "float64_to_int32_round_to_zero", S32, { F64, Discard } },
  { "float64_to_int32_scalbn", S32, { F64, Discard, Sxx, Discard } },
  { "float64_to_int64", S64, { F64, Discard } },
  { "float64_to_int64_modulo", S64, { F64, Discard, Discard } },
  { "float64_to_int64_round_to_zero", S64, { F64, Discard } },
  { "float64_to_int64_scalbn", S64, { F64, Discard, Sxx, Discard } },
  { "float64_to_uint16", U16, { F64, Discard } },
  { "float64_to_uint16_round_to_zero", U16, { F64, Discard } },
  { "float64_to_uint16_scalbn", U16, { F64, Discard, Sxx, Discard } },
  { "float64_to_uint32", U32, { F64, Discard } },
  { "float64_to_uint32_round_to_zero", U32, { F64, Discard } },
  { "float64_to_uint32_scalbn", U32, { F64, Discard, Sxx, Discard } },
  { "float64_to_uint64", U64, { F64, Discard } },
  { "float64_to_uint64_round_to_zero", U64, { F64, Discard } },
  { "float64_to_uint64_scalbn", U64, { F64, Discard, Sxx, Discard } },
  { "float64_unordered", Bool, { F64, F64, Discard } },
  { "float64_unordered_quiet", Bool, { F64, F64, Discard } },
  { "float64r32_add", F64, { F64, F64, Discard } },
  { "float64r32_div", F64, { F64, F64, Discard } },
  { "float64r32_mul", F64, { F64, F64, Discard } },
  { "float64r32_muladd", F64, { F64, F64, F64, Uxx, Discard } },
  { "float64r32_sqrt", F64, { F64, Discard } },
  { "float64r32_sub", F64, { F64, F64, Discard } },
  { "int16_to_float32", F32, { S16, Discard } },
  { "int16_to_float32_scalbn", F32, { S16, Sxx, Discard } },
  { "int16_to_float64", F64, { S16, Discard } },
  { "int16_to_float64_scalbn", F64, { S16, Sxx, Discard } },
  { "int32_to_float32", F32, { S32, Discard } },
  { "int32_to_float32_scalbn", F32, { S32, Sxx, Discard } },
  { "int32_to_float64", F64, { S32, Discard } },
  { "int32_to_float64_scalbn", F64, { S32, Sxx, Discard } },
  { "int64_to_float32", F32, { S64, Discard } },
  { "int64_to_float32_scalbn", F32, { S64, Sxx, Discard } },
  { "int64_to_float64", F64, { S64, Discard } },
  { "int64_to_float64_scalbn", F64, { S64, Sxx, Discard } },
  { "uint16_to_float32", F32, { U16, Discard } },
  { "uint16_to_float32_scalbn", F32, { U16, Sxx, Discard } },
  { "uint16_to_float64", F64, { U16, Discard } },
  { "uint16_to_float64_scalbn", F64, { U16, Sxx, Discard } },
  { "uint32_to_float32", F32, { U32, Discard } },
  { "uint32_to_float32_scalbn", F32, { U32, Sxx, Discard } },
  { "uint32_to_float64", F64, { U32, Discard } },
  { "uint32_to_float64_scalbn", F64, { U32, Sxx, Discard } },
  { "uint64_to_float32", F32, { U64, Discard } },
  { "uint64_to_float32_scalbn", F32, { U64, Sxx, Discard } },
  { "uint64_to_float64", F64, { U64, Discard } },
  { "uint64_to_float64_scalbn", F64, { U64, Sxx, Discard } },
};

static_assert(std::ranges::is_sorted(QEMUHelpers,
                                     std::less(),
                                     &HelperDescriptor::Identifier));

static mlir::Type materializeType(mlir::MLIRContext *Context,
                                  mlir::Type Type,
                                  const HelperType &Descriptor) {
  switch (Descriptor.Kind) {
  case HelperTypeKind::None:
    break;

  case HelperTypeKind::Discard:
    return {};

  case HelperTypeKind::Signed: {
    uint64_t Size = Descriptor.Size ? Descriptor.Size : getObjectSize(Type);
    return IntegerType::get(Context, IntegerKind::Signed, Size);
  }

  case HelperTypeKind::Unsigned: {
    uint64_t Size = Descriptor.Size ? Descriptor.Size : getObjectSize(Type);
    return IntegerType::get(Context, IntegerKind::Unsigned, Size);
  }

  case HelperTypeKind::Float:
    revng_assert(Descriptor.Size == 4 or Descriptor.Size == 8);
    return FloatType::get(Context, Descriptor.Size);
  }

  revng_abort("Unsupported parameter type");
}

static FunctionType transformHelperType(FunctionType HelperType,
                                        const HelperDescriptor &Descriptor) {

  llvm::SmallVector<mlir::Type> ParameterTypes;
  for (auto [I, T] : llvm::enumerate(HelperType.getArgumentTypes())) {
    revng_assert(I < std::size(Descriptor.ParameterTypes));

    if (auto NewType = materializeType(HelperType.getContext(),
                                       T,
                                       Descriptor.ParameterTypes[I]))
      ParameterTypes.push_back(NewType);
  }

  return FunctionType::get(HelperType.getContext(),
                           HelperType.getHandle(),
                           HelperType.getMutableName(),
                           HelperType.getMutableComment(),
                           materializeType(HelperType.getContext(),
                                           HelperType.getReturnType(),
                                           Descriptor.ReturnType),
                           ParameterTypes,
                           HelperType.getCAttributeList());
}

static bool retypeHelperFunctions(mlir::ModuleOp Module) {
  llvm::DenseMap<FunctionOp, const HelperDescriptor *> HelperFunctions;

  // Loop over module functions looking for supported QEMU helpers. The type of
  // each recognized function is changed and the function is recorded in the
  // above map for subsequent processing of callsites.
  for (mlir::Operation &Op : Module.getOps()) {
    auto Function = mlir::dyn_cast<FunctionOp>(&Op);
    if (not Function or not Function.isExternal())
      continue;

    auto I = std::ranges::lower_bound(QEMUHelpers,
                                      std::string_view(Function.getSymName()),
                                      std::less(),
                                      &HelperDescriptor::Identifier);

    if (I == std::ranges::end(QEMUHelpers)
        or I->Identifier != std::string_view(Function.getSymName()))
      continue;

    HelperDescriptor const &Descriptor = *I;

    auto [MI, Inserted] = HelperFunctions.try_emplace(Function, &Descriptor);
    revng_assert(Inserted);

    auto TransformedType = transformHelperType(Function.getFunctionType(),
                                               Descriptor);

    llvm::SmallVector<mlir::Attribute> ArgAttrs;
    for (auto [I, T] : llvm::enumerate(Descriptor.ParameterTypes)) {
      if (T.Kind != HelperTypeKind::None and T.Kind != HelperTypeKind::Discard)
        ArgAttrs.push_back(Function.getArgAttrs(I).getDictionaryAttr());
    }

    Function.setType(TransformedType);
    Function.setArgAttrsAttr(mlir::ArrayAttr::get(Function.getContext(),
                                                  ArgAttrs));
  }

  if (HelperFunctions.empty())
    return false;

  mlir::OpBuilder Builder(Module->getContext());

  // Find each use of an affected helper function and transform the call by
  // erasing any discarded arguments, and by inserting casts on the result and
  // arguments.
  Module->walk([&](UseOp Use) {
    auto Function = mlir::dyn_cast<FunctionOp>(Use.getUsedGlobal()
                                                 .getOperation());
    if (not Function)
      return;

    auto I = HelperFunctions.find(Function);
    if (I == std::ranges::end(HelperFunctions))
      return;

    auto Call = getOnlyUser<CallOp>(Use);
    if (not Call)
      return;

    const HelperDescriptor &Descriptor = *I->second;

    // Erase discarded arguments:
    {
      llvm::BitVector Erase(Call->getNumOperands());
      for (auto [I, T] : llvm::enumerate(Descriptor.ParameterTypes)) {
        if (T.Kind == HelperTypeKind::Discard)
          Erase.set(I + 1);
      }

      Call->eraseOperands(Erase);
    }

    Builder.setInsertionPoint(Call);
    for (auto [I, T] : llvm::enumerate(Function.getArgumentTypes())) {
      mlir::Value Value = Call->getOperand(I + 1);

      if (not equivalent(Value.getType(), T))
        Call->setOperand(I + 1,
                         Builder.create<BitCastOp>(Call->getLoc(), T, Value));
    }

    if (auto T = Function.getReturnType(); Call.getType() != T) {
      Builder.setInsertionPointAfter(Call);
      getOnlyUse(Call)->set(Builder.create<BitCastOp>(Call->getLoc(),
                                                      Call.getType(),
                                                      Call));
      Call.getResult().setType(T);
    }

    Use.getResult().setType(Function.getType());
  });

  return true;
}

struct HelperTranslationPass
  : clift::impl::CliftHelperTranslationBase<HelperTranslationPass> {

  void runOnOperation() override {
    if (retypeHelperFunctions(getOperation())) {
      mlir::RewritePatternSet Patterns(&getContext());
      if (mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(Patterns))
            .failed())
        signalPassFailure();
    }
  }
};

} // namespace

PassPtr<mlir::ModuleOp> clift::createHelperTranslationPass() {
  return std::make_unique<HelperTranslationPass>();
}
