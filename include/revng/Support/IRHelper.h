#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <optional>
#include <string>
#include <type_traits>

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Generator.h"
#include "revng/Support/IRHelperRegistry.h"
#include "revng/Support/IRHelpers.h"

/// The arguments of a helper that does not name them
enum class NoArgument {
};

/// A call to a helper
///
/// \tparam Argument the enum naming the arguments of the helper
/// \tparam Call `llvm::CallBase`, or `const llvm::CallBase` when the call has
///         been reached through a constant instruction
template<typename Argument, typename Call>
class IRHelperCallImpl {
private:
  Call *TheCall = nullptr;

public:
  explicit IRHelperCallImpl(Call *TheCall) : TheCall(TheCall) {
    revng_assert(TheCall != nullptr);
  }

  /// Widen a call reached through a mutable instruction into a constant one
  ///
  /// This is the only implicit conversion: it gives up the ability to set the
  /// arguments and nothing else. It is a template so that it never becomes the
  /// copy constructor.
  template<typename MutableCall>
    requires(std::is_const_v<Call>
             and std::same_as<MutableCall, std::remove_const_t<Call>>)
  IRHelperCallImpl(const IRHelperCallImpl<Argument, MutableCall> &Other) :
    TheCall(Other.call()) {}

public:
  Call *call() const { return TheCall; }

  llvm::Value *getArgument(Argument Index) const {
    return TheCall->getArgOperand(static_cast<unsigned>(Index));
  }

  void setArgument(Argument Index, llvm::Value *Value) const
    requires(not std::is_const_v<Call>)
  {
    TheCall->setArgOperand(static_cast<unsigned>(Index), Value);
  }
};

template<typename Argument>
using IRHelperCall = IRHelperCallImpl<Argument, llvm::CallBase>;

template<typename Argument>
using ConstIRHelperCall = IRHelperCallImpl<Argument, const llvm::CallBase>;

/// The declaration of a helper in a module
///
/// \tparam Argument the enum naming the arguments of the helper
template<typename Argument>
class IRHelperFunction {
private:
  llvm::Function *F = nullptr;

public:
  explicit IRHelperFunction(llvm::Function *F) : F(F) {
    revng_assert(F != nullptr);
  }

public:
  llvm::Function *function() const { return F; }

  llvm::Argument *getArgument(Argument Index) const {
    return F->getArg(static_cast<unsigned>(Index));
  }

  /// \return \p I as a call to this helper, or `std::nullopt` if \p I is
  ///         something else
  std::optional<IRHelperCall<Argument>> getCall(llvm::Instruction *I) const {
    auto *Call = llvm::dyn_cast<llvm::CallBase>(I);
    if (Call == nullptr or getCalledFunction(Call) != F)
      return std::nullopt;

    return IRHelperCall<Argument>(Call);
  }

  std::optional<ConstIRHelperCall<Argument>>
  getCall(const llvm::Instruction *I) const {
    const auto *Call = llvm::dyn_cast<llvm::CallBase>(I);
    if (Call == nullptr or getCalledFunction(Call) != F)
      return std::nullopt;

    return ConstIRHelperCall<Argument>(Call);
  }

  cppcoro::generator<IRHelperCall<Argument>> callers() const {
    return allCallers(F);
  }

  cppcoro::generator<IRHelperCall<Argument>>
  callersIn(llvm::Function *Container) const {
    return allCallersIn(F, Container);
  }

private:
  // These take everything they need as an argument so that the coroutine frame
  // holds a copy of it: an `IRHelperFunction` is usually a temporary, and a
  // coroutine capturing `this` would outlive it.
  static cppcoro::generator<IRHelperCall<Argument>>
  allCallers(llvm::Function *F) {
    for (llvm::CallBase *Call : ::callers(F))
      co_yield IRHelperCall<Argument>(Call);
  }

  static cppcoro::generator<IRHelperCall<Argument>>
  allCallersIn(llvm::Function *F, llvm::Function *Container) {
    for (llvm::CallBase *Call : ::callersIn(F, Container))
      co_yield IRHelperCall<Argument>(Call);
  }
};

/// \return the function \p Helper wraps, or `nullptr` if it holds nothing
template<typename Argument>
inline llvm::Function *
functionOrNull(const std::optional<IRHelperFunction<Argument>> &Helper) {
  return Helper.has_value() ? Helper->function() : nullptr;
}

/// The name of a helper
///
/// Declaring one registers its name, so that no two helpers can share it.
/// Helpers are only ever reached through one of these: nothing looks them up
/// by name.
///
/// \tparam Argument the enum naming the arguments of the helper
template<typename Argument = NoArgument>
class IRHelper {
private:
  std::string Name;
  RegisterIRHelper Registration;

public:
  explicit IRHelper(llvm::StringRef Name) :
    Name(Name.str()), Registration(Name.str()) {}

public:
  llvm::StringRef name() const { return Name; }

  /// \return the declaration of this helper in \p M, or `std::nullopt` if \p M
  ///         does not have one
  ///
  /// \note `llvm::Module::getFunction` is constant and still hands out a
  ///       mutable `llvm::Function`, so this does the same.
  std::optional<IRHelperFunction<Argument>> get(const llvm::Module &M) const {
    llvm::Function *F = M.getFunction(Name);
    if (F == nullptr)
      return std::nullopt;

    return IRHelperFunction<Argument>(F);
  }

  IRHelperFunction<Argument>
  create(llvm::Module &M,
         llvm::FunctionType *Type,
         llvm::GlobalValue::LinkageTypes Linkage) const {
    auto *F = llvm::Function::Create(Type, Linkage, Name, M);
    return IRHelperFunction<Argument>(F);
  }

  IRHelperFunction<Argument> getOrCreate(llvm::Module &M,
                                         llvm::FunctionType *Type) const {
    llvm::FunctionCallee Callee = M.getOrInsertFunction(Name, Type);
    auto *F = llvm::cast<llvm::Function>(Callee.getCallee());
    return IRHelperFunction<Argument>(F);
  }

  /// Create a helper taking no argument and returning \p ReturnType
  IRHelperFunction<Argument> getOrCreate(llvm::Module &M,
                                         llvm::Type *ReturnType) const {
    llvm::FunctionCallee Callee = M.getOrInsertFunction(Name, ReturnType);
    auto *F = llvm::cast<llvm::Function>(Callee.getCallee());
    return IRHelperFunction<Argument>(F);
  }

  /// \return \p I as a call to this helper, or `std::nullopt` if \p I is
  ///         something else
  ///
  /// This looks the helper up in the module of \p I: prefer `get` followed by
  /// `IRHelperFunction::getCall` when testing more than one instruction.
  std::optional<IRHelperCall<Argument>> getCall(llvm::Instruction *I) const {
    std::optional Function = get(*I->getModule());
    if (not Function.has_value())
      return std::nullopt;

    return Function->getCall(I);
  }

  std::optional<ConstIRHelperCall<Argument>>
  getCall(const llvm::Instruction *I) const {
    std::optional Function = get(*I->getModule());
    if (not Function.has_value())
      return std::nullopt;

    return Function->getCall(I);
  }
};
