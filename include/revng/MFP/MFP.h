#pragma once

#include "revng/ADT/STLExtras.h"
#include "revng/Support/Debug.h"

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <cstddef>
#include <map>
#include <queue>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/ReversePostOrderTraversal.h"

namespace mfp {

inline Logger NullLogger("");

/// @{
/// Tag types selecting how `MFPConfiguration::EntryLabels` is computed when
/// the caller does not provide an explicit `std::vector<Label>`.

/// Use `GT::getEntryNode(Flow)` as the only entry label. Default. Appropriate
/// for forward analyses where the graph has a single entry node.
///
/// WARNING: Do not use with `llvm::Inverse<...>` graphs. By LLVM convention
/// `GraphTraits<Inverse<llvm::Function*>>::getEntryNode` still returns the
/// forward entry node (and `GenericGraph` follows the same convention), so
/// RPOT seeded from that node in the inverse direction reaches nothing
/// useful. This is unfortunate, but it is what it is. Use `All` instead.
struct Entry {};

/// Enumerate every node via `GT::nodes_begin..nodes_end` and use them all as
/// entry labels. Appropriate for backward analyses on `llvm::Inverse<...>`,
/// where `getEntryNode` is unreliable.
struct All {};
/// @}

template<typename Label>
using EntryLabelsOptions = std::variant<Entry,
                                        All,
                                        /* non-nullptr */
                                        const std::vector<Label> *>;

template<typename T, typename StreamT>
void dump(StreamT &Stream, unsigned Indent, const T &Element) {
  for (unsigned I = 0; I < Indent; ++I)
    Stream << "  ";
  Stream << "(not implemented)\n";
}

template<typename T, typename StreamT>
void dumpLabel(StreamT &Stream, const T &Element) {
  Stream << "(not implemented)";
}

/// Position relative to a `Key` recorded into an `ExtraState`
enum class Position : unsigned char {
  Before,
  After
};

/// An hashmap that enables transfer functions to attach a LatticeElement to
/// sub-Label entities (e.g., instructions in a basic block), depending to what
/// the user is interested in
template<typename KeyT, typename LatticeElement>
class ExtraState {
public:
  using Key = KeyT;
  using KeyAndPosition = std::pair<Key, Position>;

private:
  struct Hash {
    size_t operator()(const KeyAndPosition &P) const noexcept {
      return llvm::hash_combine(P.first, static_cast<int>(P.second));
    }
  };

private:
  /// The presence of a `(Key, Position)` entry marks it as interesting; the
  /// stored value is the most recently recorded one.
  std::unordered_map<KeyAndPosition, LatticeElement, Hash> Map;

public:
  /// @{
  /// Mark a `(Key, Position)` pair as interesting. Caller-side.

  void registerAsInterestingBefore(const Key &K) {
    Map.try_emplace({ K, Position::Before });
  }

  void registerAsInterestingAfter(const Key &K) {
    Map.try_emplace({ K, Position::After });
  }
  /// @}

  /// @{
  /// Record `Value` for `(K, Position)`. No-op if not interesting. Called
  /// from inside `applyTransferFunction`.
  void registerBefore(const Key &K, const LatticeElement &Value) {
    auto It = Map.find({ K, Position::Before });
    if (It != Map.end())
      It->second = Value;
  }

  void registerAfter(const Key &K, const LatticeElement &Value) {
    auto It = Map.find({ K, Position::After });
    if (It != Map.end())
      It->second = Value;
  }
  /// @}

  /// @{
  /// Retrieve the recorded value. Caller-side, after `getMaximalFixedPoint`.
  const LatticeElement &getBefore(const Key &K) const {
    auto It = Map.find({ K, Position::Before });
    revng_assert(It != Map.end());
    return It->second;
  }

  const LatticeElement &getAfter(const Key &K) const {
    auto It = Map.find({ K, Position::After });
    revng_assert(It != Map.end());
    return It->second;
  }
  /// @}

public:
  template<typename T>
  void dump(T &Stream) const {
    Stream << Map.size() << " elements:\n";
    for (const auto &[Key, Element] : Map) {
      Stream << "  " << (Key.second == Position::Before ? "Before " : "After ");
      mfp::dumpLabel(Stream, Key.first);
      Stream << "\n";

      mfp::dump<LatticeElement>(Stream, 2, Element);
    }
  }
};

template<typename LatticeElement>
struct MFPResult {
  LatticeElement InValue;
  LatticeElement OutValue;
};

/// GT is an instance of llvm::GraphTraits e.g. llvm::GraphTraits<GraphType>
template<typename GT>
auto successors(typename GT::NodeRef From) {
  return llvm::make_range(GT::child_begin(From), GT::child_end(From));
}

/// Placeholder struct to be employed when an MFI does not need to track extra
/// state
struct NoExtraState {};

template<typename MFI>
concept HasExtraStateKey = requires { typename MFI::ExtraStateKey; };

template<typename MFI>
struct GetExtraStateKey {
  using type = typename MFI::ExtraStateKey;
};

template<typename T>
struct Identity {
  using type = T;
};

template<typename MFI>
using ExtraStateKey = std::conditional_t<HasExtraStateKey<MFI>,
                                         GetExtraStateKey<MFI>,
                                         Identity<void *>>::type;

template<typename MFI>
using ExtraStateType = std::conditional_t<
  HasExtraStateKey<MFI>,
  ExtraState<ExtraStateKey<MFI>, typename MFI::LatticeElement>,
  NoExtraState>;

template<typename MFI, typename LatticeElement = typename MFI::LatticeElement>
concept MonotoneFrameworkInstance = requires(const MFI &I,
                                             LatticeElement E1,
                                             LatticeElement E2,
                                             typename MFI::Label L,
                                             ExtraStateType<MFI> &ES) {
  /// To compute the reverse post order traversal of the graph starting from
  /// the extremal nodes, we need that the nodes also represent a subgraph
  typename llvm::GraphTraits<typename MFI::Label>::NodeRef;

  std::same_as<typename MFI::Label,
               typename llvm::GraphTraits<typename MFI::GraphType>::NodeRef>;
  { I.combineValues(E1, E2) } -> std::same_as<LatticeElement>;
  { I.isLessOrEqual(E1, E2) } -> std::same_as<bool>;
  { I.applyTransferFunction(L, E2, ES) } -> std::same_as<LatticeElement>;
};

template<typename GT>
concept HasNodeRange = requires() {
  { GT::nodes_begin };
  { GT::nodes_end };
};

template<typename Label, typename LatticeElement>
using ResultMap = std::map<Label, MFPResult<LatticeElement>>;

template<MonotoneFrameworkInstance MFI>
using MFIResultMap = ResultMap<typename MFI::Label,
                               typename MFI::LatticeElement>;

template<MonotoneFrameworkInstance MFIType>
struct MFPConfiguration {
  /// The monotone framework instance
  const MFIType *Instance = nullptr;

  /// The graph on which the monotone framework will run
  typename MFIType::GraphType Flow;

  /// The value that will be used to initialize all the non-extremal nodes.
  /// Defaults to default constructor.
  const typename MFIType::LatticeElement *Bottom = nullptr;

  /// The value that will be used to initialize all the extremal nodes
  /// Defaults to default constructor.
  const typename MFIType::LatticeElement *ExtremalValue = nullptr;

  /// The list of extremal labels
  /// Defaults to empty.
  const std::vector<typename MFIType::Label> *ExtremalLabels = nullptr;

  /// How to seed the worklist (priority RPOT). Defaults to `Entry{}`. Pass
  /// `All{}` to enumerate every node (required for backward analyses on
  /// `llvm::Inverse<...>` graphs, where `getEntryNode` returns the forward
  /// entry and is therefore unreliable). Pass `&vec` (non-null) for full
  /// control.
  EntryLabelsOptions<typename MFIType::Label> EntryLabels = Entry{};

  /// The extra state to populate during the analysis
  /// Defaults to empty.
  ExtraStateType<MFIType> *ExtraState = nullptr;

  /// A logger where the advancement of the MFP algorithm should be reported
  /// Defaults to NullLogger.
  Logger *Logger = nullptr;
};

/// Compute the solution to the given instance of a monotone framework.
///
/// Compute the maximum fixed points of an instance of monotone framework GT an
/// instance of llvm::GraphTraits that tells us how to visit the graph LGT a
/// graph type that tells us how to visit the subgraph induced by a node in the
/// graph. This is needed for the RPOT because for certain graph (e.g.
/// Inverse<...>) the nodes don't necessary carry all the information that
/// GraphType has.
template<MonotoneFrameworkInstance MFIType,
         typename GT = llvm::GraphTraits<typename MFIType::GraphType>>
MFIResultMap<MFIType>
getMaximalFixedPointImpl(MFPConfiguration<MFIType> &Configuration) {
  using Label = typename MFIType::Label;
  using LatticeElement = typename MFIType::LatticeElement;

  auto &Instance = notNull(Configuration.Instance);
  auto &Bottom = notNull(Configuration.Bottom);
  auto &ExtremalValue = notNull(Configuration.ExtremalValue);
  auto &ExtremalLabels = notNull(Configuration.ExtremalLabels);
  auto &ExtraState = notNull(Configuration.ExtraState);
  auto &Logger = notNull(Configuration.Logger);

  // Resolve the EntryLabels variant into a concrete list of seeds.
  std::vector<Label> EntryLabels;
  auto ResolveEntryLabels = [&](const auto &Option) {
    using T = std::decay_t<decltype(Option)>;
    if constexpr (std::is_same_v<T, Entry>) {
      auto Seed = GT::getEntryNode(Configuration.Flow);
      if (Seed != typename GT::NodeRef{}) {
        EntryLabels.push_back(Seed);
      }
    } else if constexpr (std::is_same_v<T, All>) {
      if constexpr (HasNodeRange<GT>) {
        auto &Flow = Configuration.Flow;
        for (auto Node :
             llvm::make_range(GT::nodes_begin(Flow), GT::nodes_end(Flow))) {
          EntryLabels.push_back(Node);
        }
      } else {
        revng_abort();
      }
    } else {
      static_assert(std::is_same_v<T, const std::vector<Label> *>);
      EntryLabels = notNull(Option);
    }
  };
  std::visit(ResolveEntryLabels, Configuration.EntryLabels);

  if (Logger.isEnabled()) {
    revng_log(Logger, "Initializing extremal labels");
    LoggerIndent Indent(Logger);
    Logger << "Extremal value:\n";
    mfp::dump(*Logger.getAsLLVMStream(), 1, Configuration.ExtremalValue);
    Logger << DoLog;

    Logger << "Extremal labels:" << DoLog;
    LoggerIndent Indent2(Logger);
    for (Label ExtremalLabel : ExtremalLabels) {
      mfp::dumpLabel(*Logger.getAsLLVMStream(), ExtremalLabel);
      Logger << DoLog;
    }

    revng_log(Logger, "Initializing initial nodes");
    LoggerIndent Indent3(Logger);
    Logger << "Initial value:\n";
    mfp::dump(*Logger.getAsLLVMStream(), 1, Bottom);
    Logger << DoLog;

    Logger << "Initial labels:" << DoLog;
    LoggerIndent Indent4(Logger);
    for (Label InitialNode : EntryLabels) {
      mfp::dumpLabel(*Logger.getAsLLVMStream(), InitialNode);
      Logger << DoLog;
    }
  }

  std::map<Label, MFPResult<LatticeElement>> AnalysisResult;

  // Initialize the state of the analysis: associate extremal labels to extremal
  // values
  for (Label ExtremalLabel : ExtremalLabels)
    AnalysisResult[ExtremalLabel].InValue = ExtremalValue;

  struct WorklistItem {
    size_t Priority;
    Label Item;

    std::weak_ordering operator<=>(const WorklistItem &) const = default;
  };
  std::set<WorklistItem> Worklist;
  std::map<Label, size_t> LabelPriority;

  //
  // Initialize the worklist with the nodes in reverse post order.
  // Also, record the visit order as priority.
  //
  // If the graph has multiple initial nodes, we perform a reverse post order
  // visit from each initial node, sharing the list of visited nodes with
  // previous visits.
  {
    using NodeSet = llvm::SmallSet<Label, 8>;
    NodeSet Visited;
    for (Label Start : EntryLabels) {

      if (Visited.contains(Start))
        continue;

      ReversePostOrderTraversalExt<Label, GT, NodeSet> RPOT(Start, Visited);
      for (Label Node : RPOT) {
        LabelPriority[Node] = LabelPriority.size();
        Worklist.insert({ LabelPriority.at(Node), Node });

        // Initialize the analysis value for non extremal nodes
        if (not AnalysisResult.contains(Node))
          AnalysisResult[Node].InValue = Bottom;
      }
    }
  }

  // Step 2 iterations
  revng_log(Logger, "Starting the iterations");
  LoggerIndent Indent(Logger);

  unsigned IterationIndex = 0;
  while (not Worklist.empty()) {
    // Fetch the next label from the worklist
    WorklistItem First = *Worklist.begin();
    Label Start = First.Item;
    Worklist.erase(First);

    auto &LabelAnalysis = AnalysisResult.at(Start);

    if (Logger.isEnabled()) {
      Logger << "Iteration #" << IterationIndex << " on ";
      mfp::dumpLabel(*Logger.getAsLLVMStream(), Start);
      Logger << DoLog;
    }

    LoggerIndent Indent(Logger);

    if (Logger.isEnabled()) {
      Logger << "Initial value:\n";
      mfp::dump(*Logger.getAsLLVMStream(), 1, LabelAnalysis.InValue);
      Logger << DoLog;

      Logger << "Final value:\n";
      mfp::dump(*Logger.getAsLLVMStream(), 1, LabelAnalysis.OutValue);
      Logger << DoLog;
    }

    // Run the transfer function.
    revng_log(Logger, "Running the transfer function");
    Logger.indent();
    const auto New = Instance.applyTransferFunction(Start,
                                                    LabelAnalysis.InValue,
                                                    ExtraState);
    Logger.unindent();

    if (Logger.isEnabled()) {
      LoggerIndent Indent(Logger);
      Logger << "New final value:\n";
      mfp::dump(*Logger.getAsLLVMStream(), 1, New);
      Logger << DoLog;
    }

    // TODO: assert that MFI.isLessOrEqual(LabelAnalysis.OutValue, New)

    // Save the new final value
    LabelAnalysis.OutValue = New;

    // Enqueue successors that need to be recomputed
    revng_log(Logger, "Processing successors:");
    LoggerIndent Indent2(Logger);
    for (Label Successor : successors<GT>(Start)) {
      auto &SuccessorResults = AnalysisResult.at(Successor);

      if (Logger.isEnabled()) {
        Logger << "Considering successor ";
        mfp::dumpLabel(*Logger.getAsLLVMStream(), Successor);
        Logger << DoLog;

        Logger << "Initial value:\n";
        LoggerIndent Indent(Logger);
        Logger << DoLog;
        mfp::dump(*Logger.getAsLLVMStream(), 1, SuccessorResults.InValue);
        Logger << DoLog;
      }
      LoggerIndent Indent(Logger);

      if (not Instance.isLessOrEqual(LabelAnalysis.OutValue,
                                     SuccessorResults.InValue)) {
        // We need to re-enqueue

        // Combine the old value with the new incoming value and update it
        SuccessorResults.InValue = Instance
                                     .combineValues(SuccessorResults.InValue,
                                                    LabelAnalysis.OutValue);

        if (Logger.isEnabled()) {
          Logger << "Enqueuing. New initial value:\n";
          mfp::dump(*Logger.getAsLLVMStream(), 1, SuccessorResults.InValue);
          Logger << DoLog;
        }

        // Enqueue in the list
        Worklist.insert({ LabelPriority.at(Successor), Successor });
      } else {
        revng_log(Logger, "Ignoring");
      }
    }

    ++IterationIndex;
  }

  return AnalysisResult;
}

template<MonotoneFrameworkInstance MFIType,
         typename GT = llvm::GraphTraits<typename MFIType::GraphType>>
MFIResultMap<MFIType>
getMaximalFixedPoint(MFPConfiguration<MFIType> Configuration) {
  std::optional<MFIType> DefaultInstance;
  if constexpr (std::is_default_constructible_v<MFIType>) {
    if (Configuration.Instance == nullptr)
      Configuration.Instance = &DefaultInstance.emplace();
  } else {
    revng_assert(Configuration.Instance != nullptr);
  }

  using LatticElement = typename MFIType::LatticeElement;
  std::optional<LatticElement> DefaultBottom;
  std::optional<typename MFIType::LatticeElement> DefaultExtremalValue;
  if constexpr (std::is_default_constructible_v<LatticElement>) {

    if (Configuration.Bottom == nullptr)
      Configuration.Bottom = &DefaultBottom.emplace();

    if (Configuration.ExtremalValue == nullptr)
      Configuration.ExtremalValue = &DefaultExtremalValue.emplace();

  } else {
    revng_assert(Configuration.Bottom != nullptr);
    revng_assert(Configuration.ExtremalValue != nullptr);
  }

  std::vector<typename MFIType::Label> DefaultExtremalLabels;
  if (Configuration.ExtremalLabels == nullptr)
    Configuration.ExtremalLabels = &DefaultExtremalLabels;

  if (Configuration.Logger == nullptr)
    Configuration.Logger = &NullLogger;

  ExtraStateType<MFIType> DefaultExtraState;
  if (Configuration.ExtraState == nullptr)
    Configuration.ExtraState = &DefaultExtraState;

  return getMaximalFixedPointImpl<MFIType, GT>(Configuration);
}

} // namespace mfp
