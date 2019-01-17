#ifndef OSRA_H
#define OSRA_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <map>

// LLVM includes
#include "llvm/Pass.h"

// Local libraries includes
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"
#include "revng/ReachingDefinitions/ReachingDefinitionsPass.h"
#include "revng/Support/IRHelpers.h"

// Local includes
#include "SimplifyComparisonsPass.h"

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
} // namespace llvm

class BVMap;
class BoundedValueHelpers;

/// \brief DFA to represent values as a + b * x, with c < x < d
class OSRAPass : public llvm::ModulePass {
public:
  static char ID;

  OSRAPass() : llvm::ModulePass(ID), BVs(nullptr) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<ConditionalReachedLoadsPass>();
    AU.addRequired<SimplifyComparisonsPass>();
    AU.addRequired<FunctionCallIdentification>();
    AU.setPreservesAll();
  }

public:
  /// \brief Represent an SSA value within a (negated) range and its signedness
  class BoundedValue {
  private:
    using BoundsVector = llvm::SmallVector<std::pair<uint64_t, uint64_t>, 3>;

  public:
    BoundedValue(const llvm::Value *V) :
      Value(V),
      Sign(UnknownSignedness),
      Bottom(false),
      Negated(false) {

      if (auto *Constant = llvm::dyn_cast<llvm::ConstantInt>(V)) {
        uint64_t Value = getLimitedValue(Constant);
        Bounds = BoundsVector{ { Value, Value } };
        Sign = AnySignedness;
      }
    }

    BoundedValue() :
      Value(nullptr),
      Sign(UnknownSignedness),
      Bottom(false),
      Negated(false) {}

    /// \brief Notify about a usage of the SSA value with a certain signedness
    ///
    /// This function can alter the signedness of the BV:
    /// `UnknownSignedness -- IsSigned --> Signed`;
    /// `UnknownSignedness -- !IsSigned --> Unsigned`;
    /// `Signed -- IsSigned --> Signed`;
    /// `Signed -- !IsSigned --> InconsistentSignedness`;
    /// `Unsigned -- !IsSigned --> Unsigned`;
    /// `Unsigned -- IsSigned --> InconsistentSignedness`;
    ///
    /// InconsistentSignedness is a sink state.
    // TODO: update users in case the sign changed
    void setSignedness(bool IsSigned);

    std::string describe() const;

    void dump() const debug_function { dump(dbg); }

    template<typename T>
    void dump(T &O) const {
      if (Negated)
        O << "NOT ";

      O << "(";
      O << getName(Value);
      O << ", ";

      switch (Sign) {
      case AnySignedness:
        O << "*";
        break;
      case UnknownSignedness:
        O << "?";
        break;
      case Signed:
        O << "s";
        break;
      case Unsigned:
        O << "u";
        break;
      case InconsistentSignedness:
        O << "x";
        break;
      }

      if (Bottom) {
        O << ", bottom";
      } else if (!isUninitialized()) {
        for (auto Bound : Bounds) {
          O << ", [";
          if (!isConstant() && hasSignedness()
              and Bound.first == lowerExtreme()) {
            O << "min";
          } else {
            O << Bound.first;
          }

          O << ", ";

          if (not isConstant() and hasSignedness()
              and Bound.second == upperExtreme()) {
            O << "max";
          } else {
            O << Bound.second;
          }

          O << "]";
        }
      }

      O << ")";
    }

    /// \brief Merge policies for BVs
    enum MergeType {
      And, ///< Intersection of the ranges
      Or ///< Union of the ranges
    };
    enum Bound { Lower, Upper };

    bool isUninitialized() const { return Sign == UnknownSignedness; }
    bool hasSignedness() const {
      return Sign != UnknownSignedness && Sign != AnySignedness;
    }

    bool isConstant() const {
      return !isUninitialized() && !Negated && !Bottom && Bounds.size() == 1
             && Bounds[0].first == Bounds[0].second;
    }

    uint64_t constant() const {
      revng_assert(isConstant());
      return Bounds[0].first;
    }

    /// \brief Merge \p Other using the \p MT policy
    template<MergeType MT = And>
    bool
    merge(BoundedValue Other, const llvm::DataLayout &DL, llvm::Type *Int64);

    /// \brief Accessor to the SSA value represent by this BV
    const llvm::Value *value() const { return Value; }

    bool isSigned() const {
      revng_assert(Sign != UnknownSignedness && Sign != AnySignedness
                   && !Bottom);
      return Sign != Unsigned;
    }

    bool isRightOpen() const {
      if (!hasSignedness() || Bounds.size() != 1)
        return false;

      if (Negated)
        return Bounds[0].first == lowerExtreme();
      else
        return Bounds[0].second == upperExtreme();
    }

    uint64_t lowerBound() const {
      revng_assert(isRightOpen());
      if (Negated)
        return Bounds[0].second;
      else
        return Bounds[0].first;
    }

    /// \brief If the BV is limited, return its bounds considering negation
    ///
    /// Do not invoke this method on unlimited BVs.
    // TODO: should this method perform a cast to the type of Value?
    std::pair<llvm::Constant *, llvm::Constant *>
    actualBoundaries(llvm::Type *Int64) const {
      revng_assert(!(Negated && isConstant()));
      revng_assert(Bounds.size() > 0);

      using CI = llvm::ConstantInt;
      uint64_t LowerBound = Bounds.front().first;
      uint64_t UpperBound = Bounds.back().second;
      if (!Negated) {
        return std::make_pair(CI::get(Int64, LowerBound, isSigned()),
                              CI::get(Int64, UpperBound, isSigned()));
      } else if (LowerBound == lowerExtreme()) {
        return std::make_pair(CI::get(Int64, UpperBound + 1, isSigned()),
                              CI::get(Int64, upperExtreme(), isSigned()));
      } else if (UpperBound == upperExtreme()) {
        return std::make_pair(CI::get(Int64, lowerExtreme(), isSigned()),
                              CI::get(Int64, LowerBound - 1, isSigned()));
      } else if (Negated) {
        return std::make_pair(CI::get(Int64, UpperBound + 1, isSigned()),
                              CI::get(Int64, LowerBound - 1, isSigned()));
      }

      revng_abort("The BV is unlimited");
    }

    BoundsVector bounds() const;

    BoundedValue &operator=(const BoundedValue &Other) {
      Value = Other.Value;
      Bounds = Other.Bounds;
      Sign = Other.Sign;
      Bottom = Other.Bottom;
      Negated = Other.Negated;
      return *this;
    }

    bool operator==(const BoundedValue &Other) const {
      // They must have the same Value
      if (Value != Other.Value)
        return false;

      // If one is bottom, both have to be bottom
      if (Bottom || Other.Bottom)
        return Bottom == Other.Bottom;

      // Same sign
      if (Sign != Other.Sign)
        return false;

      // Are they fully identical?
      if (Negated == Other.Negated && Bounds == Other.Bounds) {
        return true;
      } else if (hasSignedness() && Other.hasSignedness()
                 && Negated == !Other.Negated) {
        // One is negated, the other is not, normalize them and perform a
        // comparison
        return slowCompare(Other);
      }

      return false;
    }

    bool operator!=(const BoundedValue &Other) { return !(*this == Other); }

    /// \brief Negates this BV
    void flip() {
      Negated = !Negated;
      return;
    }

    /// \brief Set this BV to bottom
    void setBottom() {
      revng_assert(!Bottom);
      Bottom = true;
    }

    bool isBottom() const { return Bottom; }

    /// \brief Return if this BV is top
    ///
    /// A BV is top if it's uninitialized or it's not negated and both its
    /// boundaries are at their respective extremes.
    bool isTop() const {
      return (!isConstant() && Sign != AnySignedness
              && (isUninitialized()
                  || (!Negated && Bounds.size() == 1
                      && Bounds[0].first == lowerExtreme()
                      && Bounds[0].second == upperExtreme())));
    }

    /// \brief Return the size of the range constraining this BV
    ///
    /// Do not invoke this method on unlimited BVs.
    uint64_t size() const {
      revng_assert(!(Negated && isConstant()));
      revng_assert(!Bottom);
      revng_assert(Bounds.size() > 0);

      const uint64_t Max = std::numeric_limits<uint64_t>::max();
      uint64_t Result = 0;
      if (!Negated) {
        for (std::pair<uint64_t, uint64_t> Bound : Bounds) {
          if (Bound.second - Bound.first == Max)
            return Max;

          Result += Bound.second - Bound.first + 1;
        }
      } else {
        uint64_t Last = lowerExtreme() - 1;
        for (std::pair<uint64_t, uint64_t> Bound : Bounds) {
          Result += (Bound.first - 1) - (Last + 1) + 1;
          Last = Bound.second;
        }
        Result += ((upperExtreme() + 1) - 1) - (Last + 1) + 1;
      }

      return Result;
    }

    static BoundedValue
    createGE(const llvm::Value *V, uint64_t Value, bool Sign) {
      BoundedValue Result(V);
      Result.setSignedness(Sign);
      Result.Bounds = BoundsVector{ { Value, Result.upperExtreme() } };
      return Result;
    }

    static BoundedValue
    createLE(const llvm::Value *V, uint64_t Value, bool Sign) {
      BoundedValue Result(V);
      Result.setSignedness(Sign);
      Result.Bounds = BoundsVector{ { Result.lowerExtreme(), Value } };
      return Result;
    }

    static BoundedValue
    createEQ(const llvm::Value *V, uint64_t Value, bool Sign) {
      BoundedValue Result(V);
      Result.setSignedness(Sign);
      Result.Bounds = BoundsVector{ { Value, Value } };
      return Result;
    }

    static BoundedValue
    createNE(const llvm::Value *V, uint64_t Value, bool Sign) {
      BoundedValue Result = createEQ(V, Value, Sign);
      Result.flip();
      return Result;
    }

    static BoundedValue createConstant(const llvm::Value *V, uint64_t Value) {
      BoundedValue Result(V);
      Result.Bounds = BoundsVector{ { Value, Value } };
      Result.Sign = AnySignedness;
      return Result;
    }

    static BoundedValue
    createNegatedConstant(const llvm::Value *V, uint64_t Value) {
      BoundedValue Result = createConstant(V, Value);
      Result.flip();
      return Result;
    }

    static BoundedValue createBottom(const llvm::Value *V) {
      BoundedValue Result(V);
      Result.setBottom();
      return Result;
    }

    static BoundedValue createRange(const llvm::Value *V,
                                    uint64_t Lower,
                                    uint64_t Upper,
                                    bool Sign) {
      BoundedValue Result(V);
      Result.setSignedness(Sign);
      Result.Bounds = BoundsVector{ { Lower, Upper } };
      return Result;
    }

    static BoundedValue createNegatedRange(const llvm::Value *V,
                                           uint64_t Lower,
                                           uint64_t Upper,
                                           bool Sign) {
      BoundedValue Result = createRange(V, Lower, Upper, Sign);
      Result.flip();
      return Result;
    }

    /// \brief Set the BV to top
    ///
    /// Set the boundaries of the BV to their extreme values.
    void setTop() {
      if (isUninitialized())
        return;

      if (Sign == AnySignedness) {
        Bounds.clear();
        Sign = UnknownSignedness;
        Negated = false;
        Bottom = false;
        return;
      }

      Bounds = BoundsVector{ { lowerExtreme(), upperExtreme() } };
      Negated = false;
      Bottom = false;
    }

    /// Produce a new BV relative to \p V with boundaries multiplied by \p
    /// Multiplier and then adding \p Offset
    BoundedValue moveTo(llvm::Value *V,
                        const llvm::DataLayout &DL,
                        uint64_t Offset = 0,
                        uint64_t Multiplier = 0) const;

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
        revng_unreachable("Unexpected signedness");
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
        revng_unreachable("Unexpected signedness");
      }
    }

    /// \brief Performs a binary operation using signedness and type of the BV
    uint64_t performOp(uint64_t Op1,
                       unsigned Opcode,
                       uint64_t Op2,
                       const llvm::DataLayout &DL) const;

    template<BoundedValue::MergeType MT, typename T>
    BoundedValue mergeImpl(const BoundedValue &Other) const;

    /// \brief Perform a full comparison among two BoundedValues
    ///
    /// Use this in case you have a positive and a negative BoundedValue which
    /// might actually be the same
    bool slowCompare(const BoundedValue &Other) const;

    friend class BoundedValueHelpers;

  private:
    const llvm::Value *Value;
    BoundsVector Bounds;

    /// \brief Possible states for signedness
    enum Signedness : uint8_t {
      UnknownSignedness, ///< Nothing is known about the signedness
      AnySignedness, ///< Nothing is known about the signedness
      Unsigned,
      Signed,
      InconsistentSignedness ///< The BV is used both as signed and unsigned
    };

    Signedness Sign;
    uint8_t Bottom;
    uint8_t Negated;
  };

  /// \brief An OSR represents an expression a + b * x, x being a BoundedValue
  class OSR {
  public:
    /// \brief Constructor for a basic OSR
    OSR(const BoundedValue *Value) : Base(0), Factor(1), BV(Value) {}

    OSR() : Base(0), Factor(1), BV(nullptr) {}

    OSR(const OSR &Other) :
      Base(Other.Base),
      Factor(Other.Factor),
      BV(Other.BV) {}

    uint64_t constant() const {
      using CE = llvm::ConstantExpr;
      using CI = llvm::ConstantInt;
      using Constant = llvm::Constant;
      llvm::Type *T = BV->value()->getType();

      Constant *ConstantC = CI::get(T, BV->constant());
      Constant *FactorC = CI::get(T, Factor);
      Constant *BaseC = CI::get(T, Base);
      return getLimitedValue(CE::getAdd(CE::getMul(ConstantC, FactorC), BaseC));
    }

    /// \brief Combine this OSR with \p Operand through \p Opcode
    ///
    /// \p Operand is always assumed to be the second operand, never invoke this
    /// method with a non-commutative \p Opcode if \p Operand is the first
    /// operand.
    ///
    /// \param Opcode LLVM opcode describing the operation.
    /// \param Operand the constant Operand with which combine the OSR.
    /// \param FreeOpIndex the index of the non-constant operator.
    ///
    /// \return true if the OSR has been modified.
    bool combine(unsigned Opcode,
                 llvm::Constant *Operand,
                 unsigned FreeOpIndex,
                 const llvm::DataLayout &DL);

    /// \brief Compute the solution of integer equation `a + b * x = k`
    ///
    /// \param KnownTerm the right-hand side of the equation.
    /// \param CeilingRounding the rounding mode, round for excess if true.
    ///
    /// \return the solution of the integer equation using the specified
    ///         rounding mode.
    llvm::Constant *solveEquation(llvm::Constant *KnownTerm,
                                  bool CeilingRounding,
                                  const llvm::DataLayout &DL);

    /// \brief Checks if this OSR is relative to \p V
    bool isRelativeTo(const llvm::Value *V) const { return BV->value() == V; }

    /// \brief Change the BoundedValue associated to this OSR
    void setBoundedValue(BoundedValue *NewBV) { BV = NewBV; }

    /// \brief Accessor method to BoundedValue associated to this OSR
    const BoundedValue *boundedValue() const {
      revng_assert(BV != nullptr);
      return BV;
    }

    bool operator==(const OSR &Other) const {
      return Base == Other.Base && Factor == Other.Factor && BV == Other.BV;
    }

    bool operator!=(const OSR &Other) const { return !(*this == Other); }

    void describe(llvm::raw_ostream &O) const;

    std::string describe() const;

    void dump() const debug_function;

    /// \brief Return true if the OSR doesn't have a BoundedValue or is
    ///        ininfluent
    bool isConstant() const {
      revng_assert(BV != nullptr);
      return BV->isConstant();
    }

    /// \brief Helper function to performe the comparison \p P with \p C
    bool compare(unsigned short P,
                 llvm::Constant *C,
                 const llvm::DataLayout &DL,
                 llvm::Type *Int64);

    // TODO: are we sure we want to use Int64 here?
    /// \brief Compute `a + b * Value`
    llvm::Constant *evaluate(llvm::Constant *Value, llvm::Type *Int64) const;

    /// \brief Compute the boundaries value
    ///
    /// This method basically evaluates `a + b * c` and `a + b * d` being `c`
    /// and `d` the boundaries of the associated BoundedValue.
    ///
    /// \return a pair of `Constant` representing the lower and upper bounds.
    std::pair<llvm::Constant *, llvm::Constant *>
    boundaries(llvm::Type *Int64, const llvm::DataLayout &DL) const;

    class BoundsIterator {
    public:
      using bounds_pair = std::pair<uint64_t, uint64_t>;
      using container = llvm::SmallVector<bounds_pair, 3>;
      using inner_iterator = typename container::const_iterator;

      BoundsIterator(llvm::Type *T, const OSR &TheOSR, inner_iterator Start) :
        TheType(T),
        Current(Start),
        Index(0),
        TheOSR(TheOSR),
        DL(getModule(TheOSR.boundedValue()->value())->getDataLayout()) {}

      bool operator==(BoundsIterator &Other) const {
        revng_assert(TheOSR.Factor == Other.TheOSR.Factor);
        return std::tie(Current, Index) == std::tie(Other.Current, Other.Index);
      }

      bool operator!=(BoundsIterator &Other) const { return !(*this == Other); }

      BoundsIterator &operator++() {

        if ((Current->first + Index) >= Current->second) {
          ++Current;
          Index = 0;
        } else {
          Index++;
        }

        return *this;
      }

      uint64_t operator*() const;

    private:
      llvm::Type *TheType;
      inner_iterator Current;
      uint64_t Index;
      const OSR &TheOSR;
      const llvm::DataLayout &DL;
    };

    class Bounds {
    public:
      using bounds_pair = std::pair<uint64_t, uint64_t>;
      using container = llvm::SmallVector<bounds_pair, 3>;

      Bounds(llvm::Type *T, container TheBounds, const OSR &TheOSR) :
        TheType(T),
        TheBounds(TheBounds),
        TheOSR(TheOSR) {}

      BoundsIterator begin() const {
        return BoundsIterator(TheType, TheOSR, TheBounds.begin());
      }

      BoundsIterator end() const {
        return BoundsIterator(TheType, TheOSR, TheBounds.end());
      }

    private:
      llvm::Type *TheType;
      llvm::SmallVector<std::pair<uint64_t, uint64_t>, 3> TheBounds;
      const OSR &TheOSR;
    };

    Bounds bounds(llvm::Type *T) const {
      return Bounds(T, BV->bounds(), *this);
    }

    /// \brief Return the size of the associated BoundedValue
    uint64_t size() const { return BV->size(); }

    /// \brief Accessor to the factor value of this OSR (`b`)
    uint64_t base() const { return Base; }

    /// \brief Accessor to the factor value of this OSR (`b`)
    uint64_t factor() const { return Factor; }

    // TODO: bad name
    BoundedValue apply(const BoundedValue &Target,
                       llvm::Value *V,
                       const llvm::DataLayout &DL) const {
      if (Target.isBottom() || Target.isTop() || !Target.hasSignedness())
        return Target;

      return Target.moveTo(V, DL, Base, Factor);
    }

  private:
    uint64_t Base;
    uint64_t Factor;
    const BoundedValue *BV;
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
  identifyOperands(const llvm::Instruction *I, const llvm::DataLayout &DL) {
    return identifyOperands(OSRs, I, DL);
  }

  // TODO: make me private?
  static std::pair<llvm::Constant *, llvm::Value *>
  identifyOperands(std::map<const llvm::Value *, const OSR> &OSRs,
                   const llvm::Instruction *I,
                   const llvm::DataLayout &DL);

  virtual void releaseMemory() override;

private:
  ~OSRAPass() override;

private:
  // TODO: why value and not instruction?
  std::map<const llvm::Value *, const OSR> OSRs;
  BVMap *BVs;
};

#endif // OSRA_H
