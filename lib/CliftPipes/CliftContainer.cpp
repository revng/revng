//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/Helpers.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/Kinds.h"

using namespace revng;
using namespace revng::pipes;

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::SymbolOpInterface;
using mlir::SymbolTable;

using ContextPtr = std::unique_ptr<MLIRContext>;
using OwningModuleRef = mlir::OwningOpRef<ModuleOp>;

static mlir::Block &getModuleBlock(ModuleOp Module) {
  revng_assert(Module);
  revng_assert(Module->getNumRegions() == 1);
  return Module->getRegion(0).front();
}

static decltype(auto) getModuleOperations(ModuleOp Module) {
  return getModuleBlock(Module).getOperations();
}

static bool isTargetFunction(clift::FunctionOp F) {
  return getMetaAddress(F).isValid();
}

template<typename R, typename C, typename P>
static P visitHelper(R (C::*)(P));
template<typename R, typename C, typename P>
static P visitHelper(R (C::*)(P) const);

// Helper for visiting module operations non-recursively.
// Allows erasing the visited operation during visitation.
template<typename Visitor>
static void visit(ModuleOp Module, Visitor visitor) {
  // While the iterator guarantees of iplist are not explicitly documented,
  // it is expected that erasing an element from a doubly linked list does not
  // invalidate iterators to other elements. Therefore we can safely erase the
  // iterated element after first advancing the iterator to get an iterator to
  // the subsequent element or a past-the-end iterator. Because the iterator is
  // advanced in the middle of each iteration, as opposed to between iterations,
  // a range-for loop cannot be used here.
  llvm::iplist<Operation> &OpList = getModuleOperations(Module);

  auto Begin = OpList.begin();
  const auto End = OpList.end();

  while (Begin != End) {
    using RequestedOpT = decltype(visitHelper(&Visitor::operator()));
    Operation *const Op = &*Begin++;
    if constexpr (std::is_same_v<RequestedOpT, Operation *>) {
      visitor(Op);
    } else if (auto RequestedOp = mlir::dyn_cast<RequestedOpT>(Op)) {
      visitor(RequestedOp);
    }
  }
}

// Cloning MLIR from one module into another requires first serialising the
// source and then deserialising it again into the target module. We do this
// a) when cloning operations from one container to another, and
// b) when "garbage collecting" types and attributes during remove.
//
// Ideally at least for use case a) we could clone from one context into another
// directly, but that is not supported at this time.
static OwningModuleRef cloneModuleInto(ModuleOp SourceModule,
                                       MLIRContext &DestinationContext) {
  llvm::SmallString<1024> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  mlir::writeBytecodeToFile(SourceModule, Out);

  // Serialised MLIR contains multiple operations and as such must be
  // deserialised into a block. We use a temporary block stored on stack.
  mlir::Block OuterBlock;

  const mlir::LogicalResult
    R = mlir::readBytecodeFile(llvm::MemoryBufferRef(Buffer, "cloneModuleInto"),
                               &OuterBlock,
                               mlir::ParserConfig(&DestinationContext));
  revng_assert(mlir::succeeded(R));

  // The serialised code is known to contain exactly one ModuleOp operation.
  auto &Operations = OuterBlock.getOperations();
  revng_assert(Operations.size() == 1);
  auto DestinationModule = mlir::cast<ModuleOp>(Operations.front());

  if (VerifyLog.isEnabled()) {
    const size_t SrcOpsCount = getModuleOperations(SourceModule).size();
    const size_t NewOpsCount = getModuleOperations(DestinationModule).size();
    revng_assert(NewOpsCount == SrcOpsCount);
  }

  // Detach the module from the temporary block before returning it.
  DestinationModule->remove();

  // The module is now owned by the OwningModuleRef returned to the caller.
  return DestinationModule;
}

static pipeline::Target makeTarget(const MetaAddress &MA) {
  return pipeline::Target(MA.toString(), kinds::CliftFunction);
}

static void makeExternal(clift::FunctionOp F) {
  revng_assert(F->getNumRegions() == 1);

  // A function is made external by clearing its region.
  mlir::Region &Region = F.getBody();
  Region.dropAllReferences();
  Region.getBlocks().clear();
}

// Remove any unused non-target symbols.
// TODO: Investigate if the -symbol-dce pass can be used to replace this?
static void pruneUnusedSymbols(ModuleOp Module) {
  llvm::DenseSet<Operation *> UsedSymbols;

  visit(Module, [&](clift::FunctionOp F) {
    if (isTargetFunction(F))
      UsedSymbols.insert(F);

    if (F.isExternal())
      return;

    if (const auto &Uses = SymbolTable::getSymbolUses(F)) {
      for (const SymbolTable::SymbolUse &Use : *Uses) {
        const auto &SymbolRef = Use.getSymbolRef();
        revng_assert(SymbolRef.getNestedReferences().empty());

        Operation *const
          Symbol = SymbolTable::lookupSymbolIn(Module,
                                               SymbolRef.getRootReference());
        revng_assert(Symbol);

        UsedSymbols.insert(Symbol);
      }
    }
  });

  visit(Module, [&](SymbolOpInterface S) {
    if (not UsedSymbols.contains(S.getOperation()))
      S->erase();
  });
}

const char CliftFunctionContainer::ID = 0;

void CliftFunctionContainer::setModule(OwningModuleRef &&NewModule) {
  revng_assert(NewModule);
  revng_assert(clift::hasModuleAttr(NewModule.get()));

  // Make any non-target functions external.
  visit(*NewModule, [&](clift::FunctionOp F) {
    if (not F.isExternal() and not isTargetFunction(F))
      makeExternal(F);
  });

  pruneUnusedSymbols(*NewModule);
  Module = std::move(NewModule);
}

// 1. Temporarily clone the source module within the source context.
// 2. Filter from the temporary module operations that we are not interested in.
// 3. Clone the temporary module into the context of the new container.
//
// Filtering in the source module and not after cloning into the new context
// is important to avoid polluting the new context with types and attributes
// not used by the remaining functions.
std::unique_ptr<pipeline::ContainerBase>
CliftFunctionContainer::cloneFiltered(const pipeline::TargetsList &Filter)
  const {
  auto DestinationContainer = std::make_unique<CliftFunctionContainer>(name());

  if (getModuleBlock(*Module).empty())
    return DestinationContainer;

  // The temporary module is automatically deleted at end of scope.
  OwningModuleRef TemporaryModule(mlir::cast<ModuleOp>((*Module)->clone()));

  bool RemovedSome = false;
  visit(*TemporaryModule, [&](clift::FunctionOp F) {
    if (F.isExternal())
      return;

    MetaAddress MA = getMetaAddress(F);
    if (MA.isValid() and not Filter.contains(makeTarget(MA))) {
      makeExternal(F);
      RemovedSome = true;
    }
  });

  if (RemovedSome)
    pruneUnusedSymbols(*TemporaryModule);

  MLIRContext &DestinationContext = *DestinationContainer->Context;
  DestinationContext.appendDialectRegistry(Context->getDialectRegistry());
  DestinationContext.loadDialect<clift::CliftDialect>();

  OwningModuleRef &DestinationModule = DestinationContainer->Module;
  DestinationModule = cloneModuleInto(*TemporaryModule,
                                      *DestinationContainer->Context);

  return DestinationContainer;
}

void CliftFunctionContainer::mergeBackImpl(CliftFunctionContainer
                                             &&SourceContainer) {
  if (getModuleBlock(*SourceContainer.Module).empty())
    return;

  if (getModuleBlock(*Module).empty()) {
    Module = std::move(SourceContainer.Module);
    Context = std::move(SourceContainer.Context);
    return;
  }

  // Register the dialects of the other container in this container.
  Context->appendDialectRegistry(SourceContainer.Context->getDialectRegistry());
  Context->loadDialect<clift::CliftDialect>();

  // Clone the other container's module into this container's context.
  // This module is automatically erased at the end of scope.
  OwningModuleRef TemporaryModule = cloneModuleInto(*SourceContainer.Module,
                                                    *Context);

  mlir::Block &DestinationBlock = getModuleBlock(*Module);
  visit(*TemporaryModule, [&](SymbolOpInterface Symbol) {
    // Erase an existing symbol with the same name, if one exists.
    if (auto S = SymbolTable::lookupSymbolIn(*Module, Symbol.getName())) {
      if (auto F = mlir::dyn_cast<clift::FunctionOp>(Symbol.getOperation())) {
        if (F.isExternal())
          return;
      }
      S->erase();
    }

    // Move each new symbol from the temporary module to the container's module.
    Symbol->remove();
    DestinationBlock.push_back(Symbol);
  });

  // Assume that at least some symbols were copied over and always prune.
  pruneUnusedSymbols(*Module);
}

pipeline::TargetsList CliftFunctionContainer::enumerate() const {
  pipeline::TargetsList::List List;

  visit(Module.get(), [&](clift::FunctionOp F) {
    if (F.isExternal())
      return;

    MetaAddress MA = getMetaAddress(F);
    if (MA.isValid())
      List.push_back(makeTarget(MA));
  });

  // TargetsList requires ordering but does not itself sort the list.
  llvm::sort(List);

  return pipeline::TargetsList(std::move(List));
}

bool CliftFunctionContainer::removeImpl(const pipeline::TargetsList &List) {
  if (getModuleBlock(*Module).empty())
    return false;

  bool RemovedSome = false;
  visit(*Module, [&](clift::FunctionOp F) {
    if (F.isExternal())
      return;

    if (not isTargetFunction(F))
      return;

    makeExternal(F);
    RemovedSome = true;
  });

  if (RemovedSome) {
    // If any functions were removed, prune symbols and garbage collect types by
    // cloning the module into a new context.

    pruneUnusedSymbols(*Module);

    auto NewContext = clift::makeContext();
    auto NewModule = cloneModuleInto(*Module, *NewContext);

    Module = std::move(NewModule);
    Context = std::move(NewContext);
  }

  return RemovedSome;
}

void CliftFunctionContainer::clearImpl() {
  auto NewContext = clift::makeContext();
  Module = clift::makeModule(*NewContext);

  Context = std::move(NewContext);
}

llvm::Error CliftFunctionContainer::serialize(llvm::raw_ostream &OS) const {
  mlir::writeBytecodeToFile(*Module, OS);
  return llvm::Error::success();
}

llvm::Error
CliftFunctionContainer::deserializeImpl(const llvm::MemoryBuffer &Buffer) {
  auto NewContext = clift::makeContext();

  OwningModuleRef NewModule;
  if (Buffer.getBufferSize() == 0) {
    NewModule = clift::makeModule(*NewContext);

  } else {
    const mlir::ParserConfig Config(NewContext.get());
    NewModule = mlir::parseSourceString<ModuleOp>(Buffer.getBuffer(), Config);

    if (not NewModule)
      return revng::createError("Cannot load MLIR module.");

    if (not clift::hasModuleAttr(NewModule.get()))
      return revng::createError("MLIR module is not a Clift module.");
  }

  Module = std::move(NewModule);
  Context = std::move(NewContext);

  return llvm::Error::success();
}

llvm::Error
CliftFunctionContainer::extractOne(llvm::raw_ostream &OS,
                                   const pipeline::Target &Target) const {
  return cloneFiltered(pipeline::TargetsList::List{ Target })->serialize(OS);
}

static pipeline::RegisterDefaultConstructibleContainer<CliftFunctionContainer>
  X;

std::unique_ptr<pipeline::ContainerBase>
CliftContainer::cloneFiltered(const pipeline::TargetsList &Targets) const {
  auto DestinationContainer = std::make_unique<CliftContainer>(name());

  MLIRContext &DestinationContext = *DestinationContainer->Context;
  DestinationContext.appendDialectRegistry(Context->getDialectRegistry());
  DestinationContext.loadDialect<clift::CliftDialect>();

  OwningModuleRef &DestinationModule = DestinationContainer->Module;
  DestinationModule = cloneModuleInto(*Module, *DestinationContainer->Context);

  return DestinationContainer;
}

static bool isEmpty(mlir::ModuleOp Module) {
  revng_assert(Module->getAttr("clift.module"));

  if (not getModuleBlock(Module).empty()) {
    // Module contains at least one operation (function, global, etc).
    return false;
  }

  mlir::Attribute Types = Module->getAttr("clift.types");
  if (Types and mlir::cast<mlir::ArrayAttr>(Types).size() > 0) {
    // Module contains at least one type.
    return false;
  }

  return true;
}

void CliftContainer::mergeBackImpl(CliftContainer &&SourceContainer) {
  if (isEmpty(*SourceContainer.Module))
    return;

  if (isEmpty(*Module)) {
    Module = std::move(SourceContainer.Module);
    Context = std::move(SourceContainer.Context);
    return;
  }

  // Register the dialects of the other container in this container.
  Context->appendDialectRegistry(SourceContainer.Context->getDialectRegistry());
  Context->loadDialect<clift::CliftDialect>();

  // Clone the other container's module into this container's context.
  // This module is automatically erased at the end of scope.
  OwningModuleRef TemporaryModule = cloneModuleInto(*SourceContainer.Module,
                                                    *Context);

  mlir::Block &DestinationBlock = getModuleBlock(*Module);
  visit(*TemporaryModule, [&](SymbolOpInterface Symbol) {
    // Erase an existing symbol with the same name, if one exists.
    if (auto S = SymbolTable::lookupSymbolIn(*Module, Symbol.getName())) {
      if (auto F = mlir::dyn_cast<clift::FunctionOp>(Symbol.getOperation())) {
        if (F.isExternal())
          return;
      }
      S->erase();
    }

    // Move each new symbol from the temporary module to the container's module.
    Symbol->remove();
    DestinationBlock.push_back(Symbol);
  });

  for (mlir::NamedAttribute Attribute : (*SourceContainer.Module)->getAttrs()) {
    // TODO: something better to do then just override existing with incoming
    //       when there's a name clash?
    (*Module)->setAttr(Attribute.getName(), Attribute.getValue());
  }

  // Assume that at least some symbols were copied over and always prune.
  pruneUnusedSymbols(*Module);
}

pipeline::TargetsList CliftContainer::enumerate() const {
  if (isEmpty(*Module))
    return {};

  pipeline::TargetsList::List List;

  List.emplace_back(kinds::CliftModule);

  return pipeline::TargetsList(std::move(List));
}

void CliftContainer::clearImpl() {
  auto NewContext = clift::makeContext();
  Module = clift::makeModule(*NewContext);

  Context = std::move(NewContext);
}

llvm::Error CliftContainer::serialize(llvm::raw_ostream &OS) const {
  mlir::writeBytecodeToFile(*Module, OS);
  return llvm::Error::success();
}

llvm::Error CliftContainer::deserializeImpl(const llvm::MemoryBuffer &Buffer) {
  auto NewContext = clift::makeContext();

  OwningModuleRef NewModule;
  if (Buffer.getBufferSize() == 0) {
    NewModule = clift::makeModule(*NewContext);

  } else {
    const mlir::ParserConfig Config(NewContext.get());
    NewModule = mlir::parseSourceString<ModuleOp>(Buffer.getBuffer(), Config);

    if (not NewModule)
      return revng::createError("Cannot load MLIR module.");

    if (not clift::hasModuleAttr(NewModule.get()))
      return revng::createError("MLIR module is not a Clift module.");
  }

  Module = std::move(NewModule);
  Context = std::move(NewContext);

  return llvm::Error::success();
}

llvm::Error CliftContainer::extractOne(llvm::raw_ostream &OS,
                                       const pipeline::Target &Target) const {
  return cloneFiltered(pipeline::TargetsList::List{ Target })->serialize(OS);
}

const char CliftContainer::ID = 0;

static pipeline::RegisterDefaultConstructibleContainer<CliftContainer> Y;
