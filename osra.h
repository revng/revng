#ifndef _OSRA_H
#define _OSRA_H

// Standard includes
#include <cstdint>
#include <stack>
#include <limits>
#include <map>
#include <set>
#include <vector>

// Forward declarations
namespace llvm {
class BasicBlock;
class formatted_raw_ostream;
class Function;
class Instruction;
class LLVMContext;
class LoadInst;
class Module;
class SwitchInst;
class StoreInst;
class Value;
}

class OSRAPass : public llvm::FunctionPass {
public:
  static char ID;

  OSRAPass() : llvm::FunctionPass(ID) { }

  bool runOnFunction(llvm::Function &F) override;

  void describe(llvm::formatted_raw_ostream &O,
                const llvm::Instruction *I) const;
  void describe(llvm::formatted_raw_ostream &O,
                const llvm::BasicBlock *BB) const;

public:
  class BoundedValue {
  public:
    BoundedValue(const llvm::Value *V) :
      Value(V),
      LowerBound(0),
      UpperBound(0),
      Sign(UnknownSignedness),
      Bottom(false),
      Negated(false),
      Weak(false) { }

  BoundedValue() :
      Value(nullptr),
      LowerBound(0),
      UpperBound(0),
      Sign(UnknownSignedness),
      Bottom(false),
      Negated(false),
      Weak(false) { }

    // TODO: update users in case the sign changed
    void setSignedness(bool IsSigned);

    void describe(llvm::formatted_raw_ostream &O) const;

    enum MergeType { And, Or };
    enum Bound { Lower, Upper };

    bool exclude(llvm::Constant *ToExclude,
                 const llvm::DataLayout &DL,
                 llvm::Type *Int64);

    bool isUninitialized() const { return Sign == UnknownSignedness; }

    template<MergeType MT=And>
    bool merge(const BoundedValue &Other,
               const llvm::DataLayout &DL,
               llvm::Type *Int64);

    template<Bound B, MergeType Type=And>
    bool setBound(llvm::Constant *NewValue, const llvm::DataLayout &DL);

    const llvm::Value *value() const { return Value; }

    bool isSigned() const {
      assert(Sign != UnknownSignedness && !Bottom);
      return Sign != Unsigned;
    }

    llvm::Constant *lower(llvm::Type *Int64) const {
      return llvm::ConstantInt::get(Int64, LowerBound, isSigned());
    }

    llvm::Constant *upper(llvm::Type *Int64) const {
      return llvm::ConstantInt::get(Int64, UpperBound, isSigned());
    }

    /// Returns the BoundedValue bounds considering negation
    std::pair<llvm::Constant *, llvm::Constant *>
    actualBoundaries(llvm::Type *Int64) const {

      using CI = llvm::ConstantInt;
      if (!Negated)
        return std::make_pair(lower(Int64), upper(Int64));
      else if (LowerBound == lowerExtreme())
        return std::make_pair(CI::get(Int64, UpperBound + 1, isSigned()),
                              CI::get(Int64, upperExtreme(), isSigned()));
      else if (UpperBound == upperExtreme())
        return std::make_pair(CI::get(Int64, lowerExtreme(), isSigned()),
                              CI::get(Int64, LowerBound - 1, isSigned()));

      assert(false && "The OSR is unlimited");
    }

    bool operator ==(const BoundedValue &Other) const {
      if (Bottom || Other.Bottom)
        return Bottom == Other.Bottom;

      return (Value == Other.Value
              && LowerBound == Other.LowerBound
              && UpperBound == Other.UpperBound
              && Sign == Other.Sign
              && Bottom == Other.Bottom
              && Negated == Other.Negated
              && Weak == Other.Weak);
    }

    bool operator !=(const BoundedValue &Other) {
      return !(*this == Other);
    }

    void flip() {
      Negated = !Negated;
      return;
    }

    void setBottom() {
      assert(!Bottom);
      Bottom = true;
    }

    bool isBottom() const { return Bottom; }

    bool isTop() const {
      return (isUninitialized()
              || (!Negated
                  && LowerBound == lowerExtreme()
                  && UpperBound == upperExtreme()));
    }

    void setWeak() {
      Weak = true;
    }

    bool isWeak() const { return Weak; }

    uint64_t size() const {
      if (!Negated)
        return UpperBound - LowerBound;
      else if (LowerBound == lowerExtreme())
        return upperExtreme() - (UpperBound + 1);
      else if (UpperBound == upperExtreme())
        return (LowerBound - 1) - lowerExtreme();

      assert(false && "The OSR is unlimited");
    }

    static BoundedValue createGE(const llvm::Value *V,
                                 uint64_t Value,
                                 bool Sign) {
      BoundedValue Result(V);
      Result.setSignedness(Sign);
      Result.LowerBound = Value;
      Result.UpperBound = Result.upperExtreme();
      return Result;
    }

    static BoundedValue createLE(const llvm::Value *V,
                                    uint64_t Value,
                                    bool Sign) {
      BoundedValue Result(V);
      Result.setSignedness(Sign);
      Result.LowerBound = Result.lowerExtreme();
      Result.UpperBound = Value;
      return Result;
    }

    static BoundedValue createEQ(const llvm::Value *V,
                                    uint64_t Value,
                                    bool Sign) {
      BoundedValue Result(V);
      Result.setSignedness(Sign);
      Result.LowerBound = Value;
      Result.UpperBound = Value;
      return Result;
    }

    static BoundedValue createNE(const llvm::Value *V,
                                 uint64_t Value,
                                 bool Sign) {
      BoundedValue Result(V);
      Result.setSignedness(Sign);
      Result.LowerBound = Value;
      Result.UpperBound = Value;
      Result.Negated = true;
      return Result;
    }

    void setTop() {
      if (isUninitialized())
        return;

      LowerBound = lowerExtreme();
      UpperBound = upperExtreme();
      Negated = false;
    }

    bool isSingleRange() const {
      if (!Negated)
        return true;
      else
        return LowerBound == lowerExtreme() || UpperBound == upperExtreme();
    }

  private:
    uint64_t lowerExtreme() const {
      switch (Sign) {
      case Unsigned:
        return std::numeric_limits<uint64_t>::min();
      case Signed:
        return std::numeric_limits<int64_t>::min();
      case InconsistentSignedness:
        return std::numeric_limits<uint64_t>::min();
      default:
        assert(false);
      }
    }

    uint64_t upperExtreme() const {
      switch (Sign) {
      case Unsigned:
        return std::numeric_limits<uint64_t>::max();
      case Signed:
        return std::numeric_limits<int64_t>::max();
      case InconsistentSignedness:
        return std::numeric_limits<int64_t>::max();
      default:
        assert(false);
      }
    }

  public:
    const llvm::Value *Value;
    uint64_t LowerBound;
    uint64_t UpperBound;

    enum Signedness : uint8_t {
      UnknownSignedness,
      Unsigned,
      Signed,
      InconsistentSignedness
    };

    Signedness Sign;
    uint8_t Bottom;
    uint8_t Negated;
    uint8_t Weak;
  };

  class OSR {
  public:
    OSR(const BoundedValue *Value) : Base(0), Factor(1), BV(Value) { }
    OSR(uint64_t Base) : Base(Base), Factor(0), BV(nullptr) { }
    OSR() : Base(0), Factor(1), BV(nullptr) { }
    OSR(const OSR &Other) :
      Base(Other.Base),
      Factor(Other.Factor),
      BV(Other.BV) { }

    bool combine(unsigned Opcode, llvm::Constant *Operand,
                 const llvm::DataLayout &DL);

    llvm::Constant *solveEquation(llvm::Constant *KnownTerm,
                                  bool CeilingRounding,
                                  const llvm::DataLayout &DL);

    bool isRelativeTo(const llvm::Value *V) const {
      return !isConstant() && BV->value() == V;
    }

    bool isWeak() const { return !isConstant() && BV->isWeak(); }

    void setBoundedValue(BoundedValue *NewBV) {
      BV = NewBV;
    }

    const BoundedValue *boundedValue() const {
      assert(BV != nullptr);
      return BV;
    }

    bool operator ==(const OSR& Other) const {
      return Base == Other.Base && Factor == Other.Factor && BV == Other.BV;
    }

    bool operator !=(const OSR& Other) const {
      return !(*this == Other);
    }

    void describe(llvm::formatted_raw_ostream &O) const;

    bool isConstant() const {
      return !(BV != nullptr && BV->isBottom()) && Factor == 0;
    }

    bool compare(unsigned short P,
                 llvm::Constant *C,
                 const llvm::DataLayout &DL,
                 llvm::Type *Int64);

    llvm::Constant *evaluate(llvm::Constant *Value,
                             llvm::Type *Int64) const;

    uint64_t absFactor(llvm::Type *Int64,
                       const llvm::DataLayout &DL) const;

    std::pair<llvm::Constant *,
              llvm::Constant *> boundaries(llvm::Type *Int64,
                                           const llvm::DataLayout &DL) const;

    uint64_t size() const { return BV->size(); }

    uint64_t base() const { return Base; }
    uint64_t factor() const { return Factor; }

  private:
    uint64_t Base;
    uint64_t Factor;
    const BoundedValue *BV;
  };

private:
  class BVMap {
  private:
    using MapIndex = std::pair<llvm::BasicBlock *, const llvm::Value *>;
    using BVWithOrigin = std::pair<llvm::BasicBlock *, BoundedValue>;
    struct MapValue {
      BoundedValue Summary;
      std::vector<BVWithOrigin> Components;
    };

  public:
    BVMap() : BlockBlackList(nullptr), DL(nullptr), Int64(nullptr) { }
    BVMap(std::set<llvm::BasicBlock *> *BlackList,
          const llvm::DataLayout *DL,
          llvm::Type *Int64) :
      BlockBlackList(BlackList), DL(DL), Int64(Int64) { }

    void describe(llvm::formatted_raw_ostream &O,
                  const llvm::BasicBlock *BB) const;

    BoundedValue &get(llvm::BasicBlock *BB, const llvm::Value *V) {
      auto Index = std::make_pair(BB, V);
      auto MapIt = TheMap.find(Index);
      if (MapIt == TheMap.end()) {
        MapValue NewBVOVector;
        NewBVOVector.Summary = BoundedValue(V);
        auto It = TheMap.insert(std::make_pair(Index, NewBVOVector)).first;
        return summarize(BB, &It->second);
      }

      MapValue &BVOs = MapIt->second;
      return BVOs.Summary;
    }

    BoundedValue &getWeak(llvm::BasicBlock *BB, const llvm::Value *V) {
      auto Index = std::make_pair(BB, V);
      auto MapIt = TheMap.find(Index);
      if (MapIt == TheMap.end()) {
        MapValue NewBVOVector;
        BoundedValue BV(V);
        BV.setWeak();
        NewBVOVector.Summary = BV;
        auto It = TheMap.insert(std::make_pair(Index, NewBVOVector)).first;
        return summarize(BB, &It->second);
      }

      MapValue &BVOs = MapIt->second;
      assert(BVOs.Summary.isWeak());
      for (auto &BV : BVOs.Components)
        assert(BV.second.isWeak());
      return BVOs.Summary;
    }

    void setSignedness(llvm::BasicBlock *BB,
                       const llvm::Value *V,
                       bool IsSigned) {
      auto Index = std::make_pair(BB, V);
      auto MapIt = TheMap.find(Index);
      assert(MapIt != TheMap.end());

      MapValue &BVOVector = MapIt->second;
      BVOVector.Summary.setSignedness(IsSigned);
      for (BVWithOrigin &BVO : BVOVector.Components)
        BVO.second.setSignedness(IsSigned);

      summarize(BB, &MapIt->second);
    }

    std::pair<bool, BoundedValue&> update(llvm::BasicBlock *Target,
                                          llvm::BasicBlock *Origin,
                                          BoundedValue NewBV);

    void prepareDescribe() const {
      BBMap.clear();
      for (auto Pair : TheMap) {
        auto *BB = Pair.first.first;
        if (BBMap.find(BB) == BBMap.end())
          BBMap[BB] = std::vector<MapValue> { Pair.second };
        else
          BBMap[BB].push_back(Pair.second);
      }
    }

  private:
    BoundedValue &summarize(llvm::BasicBlock *Target,
                            MapValue *BVOVectorLoopInfoWrapperPass);
  private:
    std::set<llvm::BasicBlock *> *BlockBlackList;
    const llvm::DataLayout *DL;
    llvm::Type *Int64;
    std::map<MapIndex, MapValue> TheMap;
    mutable std::map<const llvm::BasicBlock *, std::vector<MapValue>> BBMap;
  };

public:
  const OSR *getOSR(const llvm::Value *V) {
    auto *I = llvm::dyn_cast<llvm::Instruction>(V);

    if (I == nullptr)
      return nullptr;

    auto It = OSRs.find(I);
    if (It == OSRs.end())
      return nullptr;
    else
      return &It->second;
  }

  std::pair<llvm::Constant *, llvm::Value *>
    identifyOperands(const llvm::Instruction *I,
                     llvm::Type *Int64,
                     const llvm::DataLayout &DL);

private:
  OSR switchBlock(OSR Base, llvm::BasicBlock *BB) {
    if (!Base.isConstant())
      Base.setBoundedValue(&BVs.get(BB, Base.boundedValue()->value()));
    return Base;
  }

  /// Returns a copy of the OSR associated with the given value, or if it does
  /// not exist, create a new one. In both cases the return value will refer to
  /// a bounded value in the context of the given basic block.
  /// Note: after invoking this function you should always check if the result
  ///       is not expressed in terms of the instruction you're analyzing
  ///       itself, otherwise we could create (possibly infinite) loops we're
  ///       not really interested in.
  OSR createOSR(llvm::Value *V, llvm::BasicBlock *BB);

private:
  // TODO: why value and not instruction?
  std::map<const llvm::Value *, const OSR> OSRs;
  BVMap BVs;
  using BVVector = llvm::SmallVector<BoundedValue, 2>;
  std::map<const llvm::Instruction *, BVVector> Constraints;
};

#endif // _OSRA_H
