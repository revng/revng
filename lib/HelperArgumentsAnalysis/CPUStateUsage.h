#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/FastValuePrinter.h"

#include "ArgumentUsageAnalysis.h"

namespace aua {

class PointerSet {
private:
  std::set<int64_t> Offsets;
  std::set<int64_t> Strides;

public:
  const int64_t *getConstant() const {
    if (Strides.size() == 0 and Offsets.size() == 1)
      return &*Offsets.begin();
    return nullptr;
  }

  int64_t offset() const {
    revng_assert(Offsets.size() == 1);
    return *Offsets.begin();
  }

  auto offsets() const { return Offsets; }
  auto strides() const { return Strides; }

public:
  static PointerSet none() { return PointerSet(); }

  static PointerSet fromConstant(int64_t Offset) {
    PointerSet Result;
    Result.Offsets.insert(Offset);
    return Result;
  }

  static PointerSet fromStrided(int64_t Stride) {
    PointerSet Result = fromConstant(0);
    revng_assert(Stride > 0);
    Result.Strides.insert(Stride);
    return Result;
  }

  static PointerSet unknown() { return fromStrided(1); }

  static PointerSet fromValue(const Value &V);

public:
  void merge(const PointerSet &Other) {
    if (Strides != Other.Strides) {
      *this = unknown();
      return;
    }

    Offsets.insert(Other.Offsets.begin(), Other.Offsets.end());
  }

  SmallVector<PointerSet, 2> enumerate() const {
    SmallVector<PointerSet, 2> Result;

    for (int64_t Offset : Offsets) {
      PointerSet New;
      New.Strides = Strides;
      New.Offsets.insert(Offset);
      Result.push_back(std::move(New));
    }

    return Result;
  }

public:
  [[nodiscard]] PointerSet add(const PointerSet &Other) const;

  [[nodiscard]] PointerSet negate() const {
    PointerSet Result;
    Result.Strides = Strides;
    for (int64_t Value : Offsets)
      Result.Offsets.insert(-Value);
    return Result;
  }

public:
  [[nodiscard]] std::string toString() const;
};

struct GlobalAUAResults {
  /// The value of the map is the number of accesses it is expanded to
  std::map<MemoryAccess, unsigned> Accesses;
  std::set<EscapedArgument> EscapedArguments;

  void registerAccess(const MemoryAccess &Access) {
    if (Access.start().collect<ArgumentValue>().size() > 0)
      Accesses.insert({ Access, 0 });
  }
};

using OffsetAndSize = std::pair<uint64_t, uint64_t>;

struct CPUStateUsage {
  GlobalAUAResults RawAUAResults;

  llvm::DenseSet<OffsetAndSize> Reads;
  llvm::DenseSet<OffsetAndSize> Writes;
  bool Escapes = false;

public:
  template<typename O>
  void dump(O &Output, llvm::StringRef Prefix) const {
    Output << Prefix.str() << "CPU State " << (Escapes ? "does" : "does not")
           << " escape\n";

    Output << Prefix.str() << "Read offsets: {";
    const char *ListPrefix = "";
    for (const auto &[Offset, Size] : Reads) {
      Output << ListPrefix << "i" << (Size * 8) << " @ " << Offset;
      ListPrefix = ", ";
    }
    Output << " }\n";

    Output << Prefix.str() << "Written offsets: {";
    ListPrefix = "";
    for (const auto &[Offset, Size] : Writes) {
      Output << " i" << (Size * 8) << " @ " << Offset;
      ListPrefix = ", ";
    }
    Output << " }\n";

    if (RawAUAResults.Accesses.size() > 0) {
      Output << Prefix.str() << "Global accesses:\n";
      for (const auto &[Access, Count] : RawAUAResults.Accesses)
        Output << Prefix.str() << "  " << Access.toString() << " (" << Count
               << "x)"
               << "\n";
    }

    if (RawAUAResults.EscapedArguments.size() > 0) {
      Output << Prefix.str() << "Global escaped arguments: {";
      for (const EscapedArgument &EscapedArgument :
           RawAUAResults.EscapedArguments)
        Output << " " << EscapedArgument.toString();
      Output << " }\n";
    }
  }

public:
  static CPUStateUsage escapes() {
    CPUStateUsage Result;
    Result.Escapes = true;
    return Result;
  }
};

class StructPointers {
private:
  const llvm::Module &M;
  const llvm::DataLayout &DL;
  std::map<llvm::StructType *, SmallVector<uint64_t, 2>> OffsetsOfStructs;
  llvm::DenseMap<llvm::Value *, llvm::StructType *> Pointers;

public:
  StructPointers(const llvm::Module &M, llvm::StructType &Struct) :
    M(M), DL(M.getDataLayout()) {
    revng_log(Log, "Analyzing CPU struct");
    LoggerIndent Indent(Log);
    visitType(Struct, 0);
  }

public:
  void registerPointer(llvm::Value &Pointer, llvm::StructType &Pointee) {
    if (OffsetsOfStructs.contains(&Pointee))
      Pointers[&Pointer] = &Pointee;
  }

  void propagateFromActualArguments();

  bool pointsIntoStruct(llvm::Value &V) const {
    return Pointers.count(&V) != 0;
  }

  const SmallVector<uint64_t, 2> &getOffsetsFor(llvm::Value &V) const {
    static SmallVector<uint64_t, 2> Empty;
    auto It = Pointers.find(&V);
    if (It == Pointers.end())
      return Empty;
    else
      return OffsetsOfStructs.at(It->second);
  }

public:
  template<typename O>
  void dump(O &Output) {
    FastValuePrinter FVP(M);
    for (auto &&[V, Struct] : Pointers) {
      std::string Function = "";
      if (auto *I = dyn_cast<llvm::Instruction>(V))
        Function = " in function " + I->getFunction()->getName().str();
      else if (auto *Argument = dyn_cast<llvm::Argument>(V))
        Function = " in function " + Argument->getParent()->getName().str();

      Output << "Value " << FVP.toString(*V) << Function << " is a pointer to "
             << Struct->getName().str()
             << " which is present at the following offsets: ";
      for (uint64_t Offset : OffsetsOfStructs.at(Struct))
        Output << " " << Offset;
      Output << "\n";
    }
  }

  void dump() debug_function { dump(dbg); }

private:
  void visitType(llvm::Type &Type, uint64_t StartingOffset);
};

class CPUStateUsageAnalysis {
private:
  Context &TheContext;
  const ArgumentUsageAnalysis &AUA;
  const llvm::DataLayout &DL;
  llvm::Type &RootType;

  /// These are information about usage of CPU state by a specific helper.
  std::map<llvm::Function *, CPUStateUsage> HelperCPUStateUsage;

  /// These are the set of parts of the CPU state that each memory access could
  /// touch.
  std::map<const llvm::Use *, llvm::DenseSet<OffsetAndSize>>
    MemoryAccessOffsets;

  /// These are the set of memory accesses where the CPU state escapes.
  /// This takes precedence over MemoryAccessOffsets.
  llvm::DenseSet<llvm::Instruction *> Escaping;

  StructPointers Initializer;

public:
  CPUStateUsageAnalysis(Context &TheContext,
                        const ArgumentUsageAnalysis &AUA,
                        const llvm::DataLayout &DL,
                        llvm::Type &RootType,
                        StructPointers &&Initializer) :
    TheContext(TheContext),
    AUA(AUA),
    DL(DL),
    RootType(RootType),
    Initializer(std::move(Initializer)) {}

public:
  CPUStateUsage *get(llvm::Function &F) {
    auto It = HelperCPUStateUsage.find(&F);
    if (It == HelperCPUStateUsage.end())
      return nullptr;
    return &It->second;
  }

  const llvm::DenseSet<std::pair<uint64_t, uint64_t>> &
  getOffsets(const llvm::Use &U) const {
    static llvm::DenseSet<std::pair<uint64_t, uint64_t>> Empty;
    auto It = MemoryAccessOffsets.find(&U);
    if (It == MemoryAccessOffsets.end())
      return Empty;
    return It->second;
  }

  bool isEscaping(const llvm::Instruction &I) const {
    return Escaping.contains(&I);
  }

public:
  void analyze(llvm::Function &Function);

  void registerAsEscaping(llvm::Function &Function) {
    HelperCPUStateUsage[&Function] = CPUStateUsage::escapes();
  }

public:
  void annotate(llvm::Module &M) const;

public:
  template<typename O>
  void dumpStats(O &Stream, llvm::StringRef Prefix) const {
    for (auto &&[Function, Usage] : HelperCPUStateUsage) {
      Stream << Prefix.str() << Function->getName().str() << ": ";
      if (Usage.Escapes) {
        Stream << "escapes";
      } else {
        Stream << "reads " << Usage.Reads.size() << " fields and ";
        Stream << "writes " << Usage.Writes.size() << " fields.";
      }
      Stream << "\n";

      std::map<llvm::Function *, std::pair<unsigned, unsigned>> CalleeStats;
      for (auto &[Access, Count] : Usage.RawAUAResults.Accesses) {
        if (auto *I = dyn_cast<llvm::Instruction>(Access.location()
                                                    .getUser())) {
          if (Access.isWrite())
            CalleeStats[I->getFunction()].second += Count;
          else
            CalleeStats[I->getFunction()].first += Count;
        }
      }

      for (auto &[F, P] : CalleeStats) {
        auto [ReadCount, WriteCount] = P;
        Stream << Prefix.str() << "  " << F->getName().str() << ": reads "
               << ReadCount << " fields and writes " << WriteCount
               << " fields.\n";
      }
    }
  }

private:
  unsigned size(llvm::Type &Type) const {
    if (auto *IntegerType = dyn_cast<llvm::IntegerType>(&Type))
      return IntegerType->getIntegerBitWidth() / 8;
    else if (isa<llvm::PointerType>(&Type))
      return DL.getPointerTypeSize(&Type);
    revng_abort();
  }

  unsigned size(const aua::MemoryAccess &Access) const {
    llvm::User *U = Access.location().getUser();
    if (auto *Store = dyn_cast<llvm::StoreInst>(U)) {
      return size(*Store->getValueOperand()->getType());
    } else if (auto *Load = dyn_cast<llvm::LoadInst>(U)) {
      return size(*Load->getType());
    } else if (auto *Call = dyn_cast<llvm::CallInst>(U)) {
      // memcpy, memmove, memset
      return cast<llvm::ConstantInt>(Call->getArgOperand(2))->getLimitedValue();
    }

    revng_abort();
  }

  GlobalAUAResults collectGlobalAUAResults(const llvm::Function &Function);

  std::optional<llvm::DenseSet<uint64_t>>
  computeAccessesInRoot(const Value &Offset) const;
};

} // namespace aua
