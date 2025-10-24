#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ModRef.h"

#include "revng/ADT/Concepts.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Tag.h"

template<typename T>
concept PointerToLLVMTypeOrDerived = std::derived_from<std::remove_pointer_t<T>,
                                                       llvm::Type>;

// TODO: make this work with the IR helper registry (see `IRHelperRegistry.h`),
//       as currently pulled functions go untracked.

template<typename KeyT>
class OpaqueFunctionsPool {
private:
  llvm::Module *M = nullptr;
  const bool PurgeOnDestruction = false;
  std::map<KeyT, llvm::Function *> Pool;
  llvm::AttributeList AttributeSets;
  llvm::MemoryEffects MemoryEffects = llvm::MemoryEffects::none();
  FunctionTags::TagsSet Tags;

public:
  OpaqueFunctionsPool(llvm::Module *M, bool PurgeOnDestruction) :
    M(M), PurgeOnDestruction(PurgeOnDestruction) {}

  ~OpaqueFunctionsPool() {
    if (PurgeOnDestruction) {
      for (auto &[Key, F] : Pool) {
        revng_assert(F->use_begin() == F->use_end());
        eraseFromParent(F);
      }
    }
  }

public:
  void addFnAttribute(llvm::Attribute::AttrKind Kind) {
    using namespace llvm;
    AttributeSets = AttributeSets.addFnAttribute(M->getContext(), Kind);
  }

  void setMemoryEffects(const llvm::MemoryEffects &NewMemoryEffects) {
    MemoryEffects = NewMemoryEffects;
  }

  void setTags(const FunctionTags::TagsSet &Tags) { this->Tags = Tags; }

public:
  auto begin() const { return Pool.begin(); }
  auto end() const { return Pool.end(); }

public:
  void record(KeyT Key, llvm::Function *F) {
    auto It = Pool.find(Key);
    if (It == Pool.end()) {
      Pool[Key] = F;
    } else {
      if (It->second != F) {
        dbg << "Same key, different function.\nOriginal:\n";
        It->second->dump();
        dbg << "New:\n";
        F->dump();
        revng_abort();
      }
    }
  }

public:
  llvm::Function *
  get(KeyT Key, llvm::FunctionType *FT, const llvm::Twine &Name = {}) {
    using namespace llvm;

    Function *F = nullptr;
    auto It = Pool.find(Key);
    if (It != Pool.end()) {
      F = It->second;
    } else {
      F = Function::Create(FT, GlobalValue::ExternalLinkage, Name, M);
      F->setAttributes(AttributeSets);
      F->setMemoryEffects(MemoryEffects);
      Tags.set(F);
      Pool.insert(It, { Key, F });
    }

    // Ensure the function we're returning is as expected
    revng_assert(F->getFunctionType() == FT);

    return F;
  }

  llvm::Function *get(KeyT Key,
                      llvm::Type *ReturnType = nullptr,
                      llvm::ArrayRef<llvm::Type *> Arguments = {},
                      const llvm::Twine &Name = {}) {
    using namespace llvm;
    if (ReturnType == nullptr)
      ReturnType = Type::getVoidTy(M->getContext());

    return get(Key, FunctionType::get(ReturnType, Arguments, false), Name);
  }

  /// Initialize the pool with all the functions in M that match the tag TheTag,
  /// using the return type as key.
  void initializeFromReturnType(const FunctionTags::Tag &TheTag)
    requires std::derived_from<std::remove_pointer_t<KeyT>, llvm::Type>
  {
    using TypeLike = std::remove_pointer_t<KeyT>;
    for (llvm::Function &F : TheTag.functions(M)) {
      auto *RetType = F.getFunctionType()->getReturnType();
      if (auto *KeyType = dyn_cast<TypeLike>(RetType))
        record(KeyType, &F);
    }
  }

  /// Initialize the pool with all the functions in M that match the tag TheTag,
  /// using the type of the ArgNo-th argument as key.
  void initializeFromNthArgType(const FunctionTags::Tag &TheTag, unsigned ArgNo)
    requires std::derived_from<std::remove_pointer_t<KeyT>, llvm::Type>
  {
    using TypeLike = std::remove_pointer_t<KeyT>;
    for (llvm::Function &F : TheTag.functions(M)) {
      auto ArgType = F.getFunctionType()->getParamType(ArgNo);
      if (auto *KeyType = dyn_cast<TypeLike>(ArgType))
        record(KeyType, &F);
    }
  }

  /// Initialize the pool with all the functions in M that match the tag TheTag,
  /// using the type of the ArgNo-th argument as key.
  void initializeFromName(const FunctionTags::Tag &TheTag)
    requires std::is_same_v<KeyT, std::string>
  {
    for (llvm::Function &F : TheTag.functions(M))
      record(F.getName().str(), &F);
  }
};

enum class InitializationMode {
  InitializeFromArgument0 = 0,
  InitializeFromArgument1 = 1,
  InitializeFromArgument2 = 2,
  InitializeFromArgument3 = 3,
  InitializeFromArgument4 = 4,
  InitializeFromArgument5 = 5,
  InitializeFromArgument6 = 6,
  InitializeFromReturnType,
  CustomInitialization
};

template<typename KeyT>
class FunctionPoolTag : public FunctionTags::Tag {
public:
  using AttributeVector = llvm::SmallVector<llvm::Attribute::AttrKind, 4>;

  using CustomInitializer = void (*)(OpaqueFunctionsPool<KeyT> &,
                                     llvm::Module &,
                                     const FunctionPoolTag &);

private:
  AttributeVector Attributes;
  llvm::MemoryEffects MemoryEffects;
  std::set<const Tag *> Tags;
  InitializationMode Initialization;
  CustomInitializer Initializer = nullptr;

private:
  FunctionPoolTag(llvm::StringRef Name,
                  AttributeVector Attributes,
                  llvm::MemoryEffects MemoryEffects,
                  std::set<const Tag *> OtherTags,
                  InitializationMode Initialization,
                  CustomInitializer Initializer) :
    Tag(Name),
    Attributes(Attributes),
    MemoryEffects(MemoryEffects),
    Tags(OtherTags),
    Initialization(Initialization),
    Initializer(Initializer) {
    Tags.insert(this);
  }

public:
  FunctionPoolTag(llvm::StringRef Name,
                  AttributeVector Attributes,
                  llvm::MemoryEffects MemoryEffects,
                  std::set<const Tag *> OtherTags,
                  InitializationMode Initialization) :
    FunctionPoolTag(Name,
                    Attributes,
                    MemoryEffects,
                    OtherTags,
                    Initialization,
                    nullptr) {

    revng_assert(Initialization != InitializationMode::CustomInitialization);
  }

  FunctionPoolTag(llvm::StringRef Name,
                  AttributeVector Attributes,
                  llvm::MemoryEffects MemoryEffects,
                  std::set<const Tag *> OtherTags,
                  CustomInitializer Initializer) :
    FunctionPoolTag(Name,
                    Attributes,
                    MemoryEffects,
                    OtherTags,
                    InitializationMode::CustomInitialization,
                    Initializer) {}

public:
  OpaqueFunctionsPool<KeyT> getPool(llvm::Module &M) const {
    OpaqueFunctionsPool<KeyT> Result(&M, false);

    // Set attributes
    for (auto Attribute : Attributes)
      Result.addFnAttribute(Attribute);

    Result.setMemoryEffects(MemoryEffects);

    // Set revng tags
    FunctionTags::TagsSet TagsSet;
    for (const Tag *TheTag : Tags)
      TagsSet.insert(*TheTag);
    Result.setTags(TagsSet);

    auto CustomInitialization = InitializationMode::CustomInitialization;
    bool HasInitializer = Initializer != nullptr;
    revng_check((Initialization == CustomInitialization) == HasInitializer);

    if constexpr (std::derived_from<std::remove_pointer_t<KeyT>, llvm::Type>) {
      switch (Initialization) {
      case InitializationMode::InitializeFromReturnType:
        Result.initializeFromReturnType(*this);
        return Result;

      case InitializationMode::InitializeFromArgument0:
      case InitializationMode::InitializeFromArgument1:
      case InitializationMode::InitializeFromArgument2:
      case InitializationMode::InitializeFromArgument3:
      case InitializationMode::InitializeFromArgument4:
      case InitializationMode::InitializeFromArgument5:
      case InitializationMode::InitializeFromArgument6: {
        auto ArgNumber = static_cast<unsigned>(Initialization);
        Result.initializeFromNthArgType(*this, ArgNumber);
        return Result;
      }

      case InitializationMode::CustomInitialization:
        // Handle below
        break;
      }
    }

    revng_check(Initialization == InitializationMode::CustomInitialization);
    Initializer(Result, M, *this);

    return Result;
  }
};
