#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "llvm/ADT/StringSet.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/Function.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/LocalIdentifier.h"
#include "revng/Model/NamingConfiguration.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/UnionDefinition.h"
#include "revng/Support/CommonOptions.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/StringOperations.h"

namespace model {

template<typename Inheritor>
class NameBuilder {
private:
  using RFT = model::RawFunctionDefinition;

public:
  const model::NamingConfiguration &Configuration;

public:
  NameBuilder(const model::Binary &Binary) :
    Configuration(Binary.Configuration().Naming()) {}

public:
  [[nodiscard]] llvm::Error isNameReserved(llvm::StringRef Name) const {
    return static_cast<const Inheritor &>(*this).isNameReserved(Name);
  }

private:
  void assertNameIsReserved(llvm::StringRef Name) const {
    if (llvm::Error Error = isNameReserved(Name)) {
      // All good, we want an error here.
      llvm::consumeError(std::move(Error));

    } else {
      std::string ActualError = "An automatic name `" + Name.str()
                                + "` is reserved and should therefore be "
                                  "banned in order to avoid collisions.";
      revng_abort(ActualError.c_str());
    }
  }

  [[nodiscard]] std::string automaticName(const model::Binary &Binary,
                                          const model::Segment &Segment) const {
    auto Iterator = Binary.Segments().find(Segment.key());
    auto Result = std::string(Configuration.unnamedSegmentPrefix())
                  + std::to_string(std::distance(Binary.Segments().begin(),
                                                 Iterator));
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::DynamicFunction &Function) const {
    // TODO: use something nicer to look at than hash, maybe punycode.
    std::string Result = Configuration.unnamedDynamicFunctionPrefix().str()
                         + revng::nameHash(Function.Name());
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::Function &Function) const {
    std::string Result = Configuration.unnamedFunctionPrefix().str()
                         + Function.Entry().toIdentifier();
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::TypeDefinition &Definition) const {
    auto K = model::TypeDefinitionKind::automaticNamePrefix(Definition.Kind());
    std::string Result = Configuration.unnamedTypeDefinitionPrefix().str()
                         + K.str() + std::to_string(Definition.ID());
    assertNameIsReserved(Result);
    return Result;
  }

  [[nodiscard]] std::string
  automaticName(const model::EnumDefinition &Definition,
                const model::EnumEntry &Entry) const {
    std::string Result = Configuration.unnamedEnumEntryPrefix().str()
                         + name(Definition) + "_"
                         + std::to_string(Entry.Value());
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::StructDefinition &Definition,
                const model::StructField &Field) const {
    std::string Result = Configuration.unnamedStructFieldPrefix().str()
                         + std::to_string(Field.Offset());
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::UnionDefinition &Definition,
                const model::UnionField &Field) const {
    std::string Result = Configuration.unnamedUnionFieldPrefix().str()
                         + std::to_string(Field.Index());
    assertNameIsReserved(Result);
    return Result;
  }

  [[nodiscard]] std::string
  automaticName(const model::CABIFunctionDefinition &Function,
                const model::Argument &Argument) const {
    std::string Result = Configuration.unnamedFunctionArgumentPrefix().str()
                         + std::to_string(Argument.Index());
    assertNameIsReserved(Result);
    return Result;
  }
  [[nodiscard]] std::string
  automaticName(const model::RawFunctionDefinition &Function,
                const model::NamedTypedRegister &Argument) const {
    std::string Result = Configuration.unnamedFunctionRegisterPrefix().str()
                         + std::string(getRegisterName(Argument.Location()));
    assertNameIsReserved(Result);
    return Result;
  }

public:
  /// These method (and its overloads) should be the only way you obtain any
  /// name you embed into an artifact (be it decompiled c, disassembly, or
  /// anything else).
  ///
  /// That ensures that the names always comply with our policy for names.
  /// Which is as follows:
  ///
  /// Users can give _any_ names to the objects they are allowed to rename
  /// in the model and as long as such names are unique, the model will be
  /// valid (*).
  /// BUT, such names can sometimes be suppressed and replaced by their
  /// corresponding automatic names (every user-renamable object always has
  /// one). That happens if:
  /// 1. The name contains symbols not allowed to be emitted by underlying
  ///    language, for example anything outside of `[A-Za-z0-9_]` for C.
  /// 2. It collides with something underlying language reserves, for example
  ///    we will not emit a function named `if` in C, or `rax` in assembly.
  /// 3. It collides with something we reserve, which includes:
  ///      1. automatic names for every renamable object, for example, struct
  ///         names: `struct_42`.
  ///      2. names for static stuff that cannot be renamed, for example,
  ///         primitive names: `uint64_t`.
  ///      3. names for dynamic stuff that cannot be renamed, for example,
  ///         helper names: `page_dump`.
  ///      4. some extra prefixes, like everything starting with `__builtin_`.
  ///
  /// ------------------------
  /// (*) Temporarily, there are further restrictions, namely you are not
  /// allowed to use `/`. But these will eventually be lifted.
  [[nodiscard]] std::string name(EntityWithName auto const &E) const {
    if (E.Name().empty()) {
      return automaticName(E);

    } else if (llvm::Error Error = isNameReserved(E.Name())) {
      // We don't care what the specific error is - if there is one,
      // just fall back on the automatic name.
      llvm::consumeError(std::move(Error));

      return automaticName(E);

    } else {
      return E.Name();
    }
  }

  [[nodiscard]] std::string name(const auto &Parent,
                                 EntityWithName auto const &E) const {
    if (E.Name().empty()) {
      return automaticName(Parent, E);

    } else if (llvm::Error Error = isNameReserved(E.Name())) {
      // We don't care what the specific error is - if there is one,
      // just fall back on the automatic name.
      llvm::consumeError(std::move(Error));

      return automaticName(Parent, E);

    } else {
      return E.Name();
    }
  }

  // This is just here for compatibility: it's nice to be able to pass anything
  // into the name builder and get the name back.
  std::string name(const model::PrimitiveType &Primitive) const {
    std::string Result = Primitive.getCName();
    assertNameIsReserved(Result);
    return Result;
  }

  /// This helper provides shared functionality for all the different types
  /// of counting name builders (those that emit `prefix_1`, `prefix_2`,
  /// `prefix_3`, and so on as default names on sequential requests).
  ///
  /// It ensures that:
  /// - a single *name* cannot appear more than once
  /// - a single *location* cannot appear more than once
  /// - the indexes (either a part of a name, or the corresponding locations)
  ///   are unique for each \ref nameImpl call.
  class CountingNameBuilder {
  private:
    const NameBuilder *Parent = nullptr;

    uint64_t NextIndex = 0;

    /// This is used to keep track of the *names* that were already emitted.
    llvm::DenseSet<llvm::StringRef> EmittedNames = {};

    /// This is used to keep track of the *locations* that were already emitted.
    std::set<SortedVector<MetaAddress>> EmittedLocations = {};

  protected:
    const NameBuilder &parent() const { return notNull(Parent); }

  public:
    CountingNameBuilder() = default;
    CountingNameBuilder(const NameBuilder &Parent) : Parent(&Parent) {}

  public:
    struct NamingResult {
      std::string Name;

      /// The index of this variable provided to enable `pipeline::Location`
      /// to use it as a key.
      uint64_t Index;

      /// This flag is set for variables that cannot be located (either no
      /// address set can be gathered or the gathered address set is already
      /// attached to another variable). Such variables should have any actions
      /// allowed on them.
      bool HasAddressAssociated;

      /// Since the same variable shouldn't be queried more then once,
      /// the warning appears warning here instead of using a dedicated query
      /// method (like the base `NameBuilder`).
      std::string Warning = {};
    };

    [[nodiscard]] NamingResult automaticName(llvm::StringRef Prefix,
                                             bool HasAddressAssociated) {
      uint64_t CurrentIndex = NextIndex++;
      std::string Result = Prefix.str() + std::to_string(CurrentIndex);
      parent().assertNameIsReserved(Result);
      return { Result, CurrentIndex, HasAddressAssociated };
    }

  protected:
    [[nodiscard]] NamingResult
    nameImpl(SortedVector<MetaAddress> const &UserLocationSet,
             RangeOf<const model::LocalIdentifier &> auto const &KnownNames,
             llvm::StringRef AutomaticPrefix) {
      if (UserLocationSet.empty()) {
        // No location is provided, emit an *untagged* automatic name.
        return automaticName(AutomaticPrefix, false);
      }

      bool IsLocationUnique = EmittedLocations.insert(UserLocationSet).second;
      if (not IsLocationUnique) {
        // This location was used for a different variable already,
        // emit an *untagged* automatic name.
        return automaticName(AutomaticPrefix, false);
      }

      auto Comparator = [&](const model::LocalIdentifier &Identifier) {
        return Identifier.Location() == UserLocationSet;
      };
      auto Iterator = std::ranges::find_if(KnownNames, Comparator);
      if (Iterator == KnownNames.end()) {
        // There's nothing in the model for the current location,
        // emit a *tagged* automatic name.
        return automaticName(AutomaticPrefix, true);
      }

      bool IsNameUnique = EmittedNames.insert(Iterator->Name()).second;
      if (not IsNameUnique) {
        // This name was already emitted, fall back on the automatic name.
        auto Result = automaticName(AutomaticPrefix, true);
        Result.Warning = "Name `" + Iterator->Name()
                         + "` cannot be used more than once, so it was "
                           "replaced by an automatic one (`"
                         + Result.Name + "`).";
        return Result;
      }

      if (llvm::Error Reason = parent().isNameReserved(Iterator->Name())) {
        // Current name cannot be used as its reserved for something else.
        auto Result = automaticName(AutomaticPrefix, true);
        Result.Warning = "Name `" + Iterator->Name()
                         + "` is not valid, so it was replaced by an automatic "
                           "one (`"
                         + Result.Name + "`) because "
                         + revng::unwrapError(std::move(Reason)) + ".";
        return Result;

      } else {
        // We don't want existing automatic names to "shift" when a rename
        // happens, as such we have to advance the index even if no automatic
        // name is produced for it.
        uint64_t CurrentIndex = NextIndex++;

        return { Iterator->Name(), CurrentIndex, true };
      }
    }

    [[nodiscard]] std::set<llvm::StringRef>
    homelessNamesImpl(RangeOf<const model::LocalIdentifier &> auto R) const {
      return R | std::views::filter([this](const auto &V) {
               return not EmittedNames.contains(V.Name());
             })
             | std::views::transform([](const auto &V) { return V.Name(); })
             | revng::to<std::set<llvm::StringRef>>();
    }
  };

  class VariableNameBuilder : public CountingNameBuilder {
    const model::Function *Function = nullptr;

  public:
    VariableNameBuilder() = default;
    VariableNameBuilder(const NameBuilder &Parent,
                        const model::Function &Function) :
      CountingNameBuilder(Parent), Function(&Function) {}

    const model::Function &function() const { return notNull(Function); }

    CountingNameBuilder::NamingResult
    name(SortedVector<MetaAddress> const &UserLocationSet) {
      auto Prefix = this->parent().Configuration.unnamedLocalVariablePrefix();
      return CountingNameBuilder::nameImpl(UserLocationSet,
                                           function().LocalVariables(),
                                           Prefix);
    }

    CountingNameBuilder::NamingResult
    name(TrackingSortedVector<MetaAddress> const &UserLocationSet) {
      auto Prefix = this->parent().Configuration.unnamedLocalVariablePrefix();
      return CountingNameBuilder::nameImpl(UserLocationSet.unwrap(),
                                           function().LocalVariables(),
                                           Prefix);
    }

    std::set<llvm::StringRef> homelessNames() const {
      return CountingNameBuilder::homelessNamesImpl(Function->LocalVariables());
    }
  };
  VariableNameBuilder localVariables(const model::Function &Function) const {
    return VariableNameBuilder(*this, Function);
  }

  class GotoLabelNameBuilder : public CountingNameBuilder {
    const model::Function *Function = nullptr;

  public:
    GotoLabelNameBuilder() = default;
    GotoLabelNameBuilder(const NameBuilder &Parent,
                         const model::Function &Function) :
      CountingNameBuilder(Parent), Function(&Function) {}

    const model::Function &function() const { return notNull(Function); }

    CountingNameBuilder::NamingResult
    name(SortedVector<MetaAddress> const &UserLocationSet) {
      auto Prefix = this->parent().Configuration.unnamedGotoLabelPrefix();
      return CountingNameBuilder::nameImpl(UserLocationSet,
                                           function().GotoLabels(),
                                           Prefix);
    }

    CountingNameBuilder::NamingResult
    name(TrackingSortedVector<MetaAddress> const &UserLocationSet) {
      auto Prefix = this->parent().Configuration.unnamedGotoLabelPrefix();
      return CountingNameBuilder::nameImpl(UserLocationSet.unwrap(),
                                           function().GotoLabels(),
                                           Prefix);
    }

    std::set<llvm::StringRef> homelessNames() const {
      return CountingNameBuilder::homelessNamesImpl(Function->GotoLabels());
    }
  };
  GotoLabelNameBuilder gotoLabels(const model::Function &Function) const {
    return GotoLabelNameBuilder(*this, Function);
  }

  [[nodiscard]] std::string llvmName(const model::Function &Function) const {
    if (DebugNames)
      return "local_" + name(Function);
    else
      return "local_" + Function.Entry().toString();
  }

private:
  [[nodiscard]] std::optional<std::string>
  warningImpl(const std::string &Name,
              const std::string &OriginalName,
              llvm::Error &&Reason) const {
    if (Name == OriginalName) {
      // Suppress warnings on appropriately placed automatic names in the model.
      //
      // TODO: get rid of this once `import-from-c` is fixed and we no longer
      //       put such names in.
      llvm::consumeError(std::move(Reason));
      return std::nullopt;
    }

    return "Name `" + OriginalName
           + "` is not valid, so it was replaced by an automatic one (`" + Name
           + "`) because " + revng::unwrapError(std::move(Reason)) + ".";
  }

public:
  [[nodiscard]] std::optional<std::string>
  warning(EntityWithName auto const &E) const {
    if (E.Name().empty())
      return std::nullopt;

    if (auto Reason = isNameReserved(E.Name()))
      return warningImpl(automaticName(E), E.Name(), std::move(Reason));

    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::string>
  warning(const auto &Parent, EntityWithName auto const &E) const {
    if (E.Name().empty())
      return std::nullopt;

    if (auto Reason = isNameReserved(E.Name()))
      return warningImpl(automaticName(Parent, E), E.Name(), std::move(Reason));

    return std::nullopt;
  }

public:
  [[nodiscard]] std::string paddingFieldName(uint64_t Offset) const {
    std::string Result = Configuration.structPaddingPrefix().str()
                         + std::to_string(Offset);
    assertNameIsReserved(Result);
    return Result;
  }

public:
  [[nodiscard]] std::string
  artificialReturnValueWrapperName(const RFT &Function) const {
    auto Result = Configuration.artificialReturnValuePrefix().str()
                  + name(Function);
    assertNameIsReserved(Result);
    return Result;
  }
};

struct CNameBuilder : public NameBuilder<CNameBuilder> {
public:
  using NameBuilder<CNameBuilder>::NameBuilder;

  [[nodiscard]] llvm::Error isNameReserved(llvm::StringRef Name) const;
};
struct AssemblyNameBuilder : NameBuilder<AssemblyNameBuilder> {
public:
  const model::Architecture::Values &Architecture;

public:
  AssemblyNameBuilder(const model::Binary &Binary) :
    NameBuilder<AssemblyNameBuilder>(Binary),
    Architecture(Binary.Architecture()) {}

  [[nodiscard]] llvm::Error isNameReserved(llvm::StringRef Name) const;
};

} // namespace model
