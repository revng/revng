//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Transforms/IPO/StripSymbols.h"

#include "revng/Lift/JumpTargetReason.h"
#include "revng/Lift/LibTcg.h"
#include "revng/Lift/Lift.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelperRegistry.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/NewPC.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/SimplePassManager.h"

#include "CodeGenerator.h"
#include "PostLiftVerifyPass.h"

using namespace llvm::cl;

namespace {
const char *EntryDescStr = "virtual address of the entry point where to start";
opt<unsigned long long> EntryPointAddress("entry",
                                          desc(EntryDescStr),
                                          value_desc("address"),
                                          cat(MainCategory));
alias A1("e",
         desc("Alias for -entry"),
         aliasopt(EntryPointAddress),
         cat(MainCategory));

} // namespace

struct ExternalFilePaths {
  std::string LibHelpers;
  std::string EarlyLinked;
};

static ExternalFilePaths
findExternalFilePaths(const model::Architecture::Values Architecture) {
  // What symbols from the revng namespace are actually used here?
  using namespace revng;

  const std::string ArchName = model::Architecture::getQEMUName(Architecture)
                                 .str();

  ExternalFilePaths Paths = {};

  // Note: here we use the declaration version of the helpers, i.e., where all
  //       helper functions are just declarations.
  const std::string LibHelpersName = "/share/revng/"
                                     "libtcg-helpers-declarations-only-"
                                     + ArchName + ".bc";
  auto OptionalHelpers = ResourceFinder.findFile(LibHelpersName);
  revng_assert(OptionalHelpers.has_value(), "Cannot find libtcg helpers");
  Paths.LibHelpers = OptionalHelpers.value();

  const std::string EarlyLinkedName = "/share/revng/early-linked-" + ArchName
                                      + ".ll";
  auto OptionalEarlyLinked = ResourceFinder.findFile(EarlyLinkedName);
  revng_assert(OptionalEarlyLinked.has_value(), "Cannot find early-linked.ll");

  Paths.EarlyLinked = OptionalEarlyLinked.value();

  return Paths;
}

/// Map describing the jump targets in the LLVM module, each one is identified
/// by its MetaAddress. The boolean value represents if the jump target has been
/// discovered through harvesting (false) or successively through the list of
/// model functions (true).
using JumpTargetMap = std::map<MetaAddress, bool>;

template<>
struct llvm::yaml::CustomMappingTraits<JumpTargetMap> {
  static void inputOne(IO &IO, StringRef Key, JumpTargetMap &Data) {
    MetaAddress Address = MetaAddress::fromString(Key);
    IO.mapRequired(Key.str().c_str(), Data[Address]);
  }

  static void output(IO &IO, JumpTargetMap &Data) {
    for (auto &[Key, Value] : Data)
      IO.mapRequired(Key.toString().c_str(), Value);
  }
};

static std::tuple<bool, std::map<MetaAddress, bool>>
collectJumpTargets(const llvm::Module &Module) {
  namespace JTR = JumpTargetReason;

  const llvm::Function *Root = Module.getFunction("root");
  std::optional NewPC = NewPCHelper.get(Module);

  if (Root == nullptr)
    return { false, {} };

  if (not NewPC.has_value())
    return { true, {} };

  // Collect all jump targets by inspecting calls to newpc and record whether it
  // was found after adding entry addresses of functions
  std::map<MetaAddress, bool> JumpTargets;
  for (IRHelperCall<NewPCArgument> Call : NewPC->callers()) {
    if (startsBasicBlock(Call)) {
      MetaAddress Address = addressFromNewPC(Call);

      // Detect if this jump targets has been discovered *after* recording the
      // entry addresses of functions

      // Be conservative and assume it is, in absence of information
      bool DependsOnModelFunction = true;
      const llvm::Instruction
        *Terminator = Call.call()->getParent()->getTerminator();
      if (Terminator->hasMetadata(JTR::MDName)) {
        uint32_t Reasons = JTR::getReasons(Terminator);
        DependsOnModelFunction = hasReason(Reasons,
                                           JTR::DependsOnModelFunction);
      }

      JumpTargets.emplace(Address, DependsOnModelFunction);
    }
  }

  return { true, JumpTargets };
}

static llvm::Error checkPrecondition(const model::Binary &Model) {
  if (Model.Architecture() == model::Architecture::Invalid) {
    return revng::createError("Cannot lift binary with architecture invalid.");
  }

  if (Model.DefaultABI() == model::ABI::Invalid
      and Model.DefaultPrototype().isEmpty()) {
    return revng::createError("Cannot lift binary with neither `DefaultABI` "
                              "nor `DefaultPrototype`.");
  }

  return llvm::Error::success();
}

static bool shouldInvalidateRoot(const std::map<MetaAddress, bool> &JumpTargets,
                                 const TupleTreeDiff<model::Binary> &Diff) {
  // Inspect the diff looking for newly added model::Functions
  using Fields = TupleLikeTraits<model::Binary>::Fields;
  size_t FunctionsIndex = static_cast<size_t>(Fields::Functions);
  for (const auto &Change : Diff.Changes) {
    bool IsAddition = not Change.Old.has_value() and Change.New.has_value();
    bool IsRemoval = Change.Old.has_value() and not Change.New.has_value();

    // Look for additions to /Functions
    auto &Path = Change.Path;
    if (Path.size() == 1 and Path[0].get<size_t>() == FunctionsIndex) {
      // Check the Entry address of the newly added model::Function
      MetaAddress ChangedAddress;
      if (IsAddition)
        ChangedAddress = std::get<model::Function>(*Change.New).Entry();
      else
        ChangedAddress = std::get<model::Function>(*Change.Old).Entry();

      auto It = JumpTargets.find(ChangedAddress);
      bool IsJumpTarget = It != JumpTargets.end();
      bool DependsOnModelFunction = IsJumpTarget and It->second;

      if (IsAddition and not IsJumpTarget) {
        // We're adding a function that was not a jump target
        return true;
      } else if (IsRemoval and DependsOnModelFunction) {
        // We're removing a function whose address was not discovered *before*
        // starting to take into account the entry addresses of model
        // functions
        return true;
      }
    }
  }

  return false;
}

namespace revng::pypeline::piperuns {

Lift::Lift(const class Model &Model,
           llvm::StringRef Config,
           llvm::StringRef DynamicConfig,
           const BinariesContainer &Binary,
           LLVMRootContainer &ModuleContainer) :
  TheModel(Model), Binary(Binary), ModuleContainer(ModuleContainer) {
}

CustomInvalidationData Lift::run() {
  llvm::Task T(6, "Lift");
  const TupleTree<model::Binary> &Model = TheModel.get();

  T.advance("findFiles", false);
  const auto Paths = findExternalFilePaths(Model->Architecture());

  // Look for the library in the system's paths
  T.advance("Load libtcg", false);
  auto TheLibTcg = LibTcg::get(Model->Architecture());

  // Get access to raw binary data
  revng_assert(Binary.size() == 1);
  llvm::ArrayRef<char> File = Binary.getFile(0);
  RawBinaryView RawBinary(*Model, { File.data(), File.size() });
  llvm::Module &Module = ModuleContainer.getModule();

  T.advance("Construct CodeGenerator", false);
  CodeGenerator Generator(RawBinary,
                          &Module,
                          Model,
                          Paths.LibHelpers,
                          Paths.EarlyLinked,
                          model::Architecture::x86_64);

  std::optional<uint64_t> EntryPointAddressOptional;
  if (EntryPointAddress.getNumOccurrences() != 0)
    EntryPointAddressOptional = EntryPointAddress;
  T.advance("Translate", true);

  Generator.translate(TheLibTcg, EntryPointAddressOptional);

  T.advance("Sort Module", true);
  sortModule(Module);

  T.advance("Verify Module", true);
  // TODO: convert this from a pass to a free-standing function
  PostLiftVerifyPass{}.runOnModule(Module);

  SimplePassManager PM;
  PM.addPass(llvm::StripDeadDebugInfoPass());
  PM.run(Module);

  // Compute invalidation data
  Buffer SerializedInvalidation;

  {
    auto [HasRoot, JumpTargets] = collectJumpTargets(Module);
    revng_assert(HasRoot);
    llvm::raw_svector_ostream OS(SerializedInvalidation.data());
    serialize(OS, JumpTargets);
  }

  return { {}, { { ObjectID(), SerializedInvalidation } } };
}

llvm::Error Lift::checkPrecondition(const class Model &Model) {
  const model::Binary &Binary = *Model.get().get();
  return revng::joinErrors(::checkPrecondition(Binary),
                           RawBinaryView::checkPrecondition(Binary));
}

bool Lift::requiresCustomInvalidation(const ModelDiff &Diff) {
  using Fields = TupleLikeTraits<model::Binary>::Fields;
  size_t FunctionsIndex = static_cast<size_t>(Fields::Functions);
  for (const auto &Change : Diff.get().Changes) {
    if (Change.Path.size() == 1
        and Change.Path[0].get<size_t>() == FunctionsIndex)
      return true;
  }
  return false;
}

std::vector<std::set<ObjectID>>
Lift::processCustomInvalidation(const InvalidationData &Data,
                                const ModelDiff &Diff) {
  auto LLVMModuleData = Data.at(1);
  revng_assert(LLVMModuleData.size() == 1);
  revng_assert(*std::get<0>(LLVMModuleData[0]) == ObjectID());

  auto DataBuffer = std::get<1>(LLVMModuleData[0]);
  llvm::StringRef String(reinterpret_cast<const char *>(DataBuffer.data()),
                         DataBuffer.size());
  auto JumpTargets = llvm::cantFail(fromString<JumpTargetMap>(String));

  if (shouldInvalidateRoot(JumpTargets, Diff.get()))
    return { {}, { ObjectID() } };
  else
    return {};
}

} // namespace revng::pypeline::piperuns
