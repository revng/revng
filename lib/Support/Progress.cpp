/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#if defined(__linux__)
extern "C" {
#include "sys/ioctl.h"
}
#endif

#include <chrono>
#include <fstream>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Progress.h"
#include "llvm/Support/Signals.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"

using namespace llvm::cl;

static opt<bool> HighResolutionMemory("high-resolution-memory-tracing",
                                      desc("Use a slower, more precise, source "
                                           "for memory traces"),
                                      init(false));

template<StrictSpecializationOf<std::chrono::duration> T>
class DurationParser : public llvm::cl::parser<T> {
public:
  using llvm::cl::parser<T>::parser;

  bool parse(llvm::cl::Option &Option,
             llvm::StringRef ArgName,
             const llvm::StringRef ArgValue,
             T &Val) {
    uint64_t IntegerValue = 0;
    bool Error = ArgValue.getAsInteger(10, IntegerValue);
    if (Error)
      return Option.error(ArgValue + " is not a valid number");

    Val = T{ IntegerValue };
    return false;
  }
};

static opt<std::chrono::milliseconds,
           false,
           DurationParser<std::chrono::milliseconds>>
  MemoryProfilerInterval("memory-profiler-interval",
                         desc("Debouncing time (in ms) for emitting a memory "
                              "usage tracepoint"),
                         init(std::chrono::milliseconds{ 100 }));

static void destroyTraceProgressListener(void *OpaqueListener);

class TraceProgressListener : public llvm::ProgressListener {
private:
  std::error_code EC;

  // TODO: consider buffering in thread-local buffers
  llvm::sys::Mutex OutputMutex;
  llvm::raw_fd_ostream Output;
  bool Closed = false;

  std::chrono::time_point<std::chrono::high_resolution_clock> LastMemoryPoll;

public:
  static constexpr bool AllThreads = true;

public:
  TraceProgressListener(llvm::StringRef OutputPath) : Output(OutputPath, EC) {
    revng_assert(!EC);
    Output << "[\n";
    llvm::sys::AddSignalHandler(destroyTraceProgressListener, this);
  }

  ~TraceProgressListener() override { close("Graceful exit"); }

public:
  void close(llvm::StringRef ExitReason) {
    if (not Closed) {
      Closed = true;
      emitEvent<false>(ExitReason, "task", "i");
      Output << "]\n";
      Output.flush();
    }
  }

public:
  void handleNewTask(const llvm::Task *T) override {
    llvm::sys::ScopedLock Lock(OutputMutex);
    emitEvent(T->name(), "task", "B");
  }

  void handleTaskCompleted(const llvm::Task *T) override {
    llvm::sys::ScopedLock Lock(OutputMutex);
    if (T->stepIndex() != -1) {
      emitEvent(T->stepName(), "task", "E");
    }
    emitEvent(T->name(), "task", "E");
  }

  void handleTaskAdvancement(const llvm::Task *T,
                             llvm::StringRef PreviousStepName) override {
    llvm::sys::ScopedLock Lock(OutputMutex);
    if (T->stepIndex() != 0)
      emitEvent(PreviousStepName, "task", "E");
    emitEvent(T->stepName(), "task", "B");
  }

private:
  static unsigned long long getTimestamp() {
    using namespace std::chrono;
    using std::chrono::microseconds;
    auto Epoch = system_clock::now().time_since_epoch();
    return duration_cast<microseconds>(Epoch).count();
  }

  template<bool EmitTrailingComma = true>
  void emitEvent(llvm::StringRef Name,
                 llvm::StringRef Category,
                 llvm::StringRef Phase) {
    Output << "{";
    Output << "\"name\": \"" << Name.str() << "\", ";
    Output << "\"cat\": \"" << Category.str() << "\", ";
    Output << "\"ph\": \"" << Phase.str() << "\", ";
    Output << "\"ts\": " << getTimestamp() << ", ";
    Output << "\"pid\": " << getpid() << ", ";
    Output << "\"tid\": " << getpid();
    Output << "}";
    if (EmitTrailingComma)
      Output << ",";
    Output << "\n";

    if (EmitTrailingComma)
      emitMemoryUsage();
  }

  void emitMemoryUsage() {
    auto Now = std::chrono::high_resolution_clock::now();
    if (Now - LastMemoryPoll < MemoryProfilerInterval)
      return;

    unsigned long long RSS;
    if (HighResolutionMemory) {
      RSS = readMemorySmaps();
    } else {
      RSS = readMemoryStatm();
    }

    Output << "{\"ph\": \"C\", ";
    Output << "\"ts\": " << getTimestamp() << ", ";
    Output << "\"pid\": " << getpid() << ", ";
    Output << "\"args\": {\"memory (RSS)\": " << RSS << "}},\n";

    LastMemoryPoll = Now;
  }

  static unsigned long long readMemorySmaps() {
    // TODO: linux-specific and requires `smaps_rollup` which is linux 4.14+.
    // Use `std::ifstream` instead of `llvm::MemoryBuffer` as it avoids reading
    // the entire file.
    std::ifstream IS("/proc/self/smaps_rollup");
    std::string Buffer;
    unsigned long long RSSKbytes = 0ULL;

    while (not IS.eof()) {
      std::getline(IS, Buffer);
      if (Buffer.starts_with("Rss:")) {
        llvm::StringRef BufferRef(Buffer);
        BufferRef.consume_front("Rss:");
        BufferRef.consume_back("kB");
        bool Error = BufferRef.trim().getAsInteger(10, RSSKbytes);
        revng_assert(not Error);
        break;
      }
    }
    revng_assert(RSSKbytes != 0);
    return RSSKbytes * 1024;
  }

  static unsigned long long readMemoryStatm() {
    static long PageSize = sysconf(_SC_PAGE_SIZE);
    static constexpr llvm::StringRef Path = "/proc/self/statm";

    auto Buffer = revng::cantFail(llvm::MemoryBuffer::getFileAsStream(Path));
    llvm::SmallVector<llvm::StringRef> Parts;
    Buffer->getBuffer().split(Parts, " ");
    revng_assert(Parts.size() > 2);
    unsigned long long RSSPages;
    bool Error = Parts[1].getAsInteger(10, RSSPages);
    revng_assert(not Error);

    return RSSPages * PageSize;
  }
};

class PlainProgressListener : public llvm::ProgressListener {
private:
  llvm::raw_ostream &Output;

public:
  static constexpr bool AllThreads = false;

public:
  PlainProgressListener(llvm::raw_ostream &Output) : Output(Output) {}

public:
  void handleNewTask(const llvm::Task *T) override {
    indent(T->index());
    Output << "Starting " << T->name().str();
    auto MaybeStepsCount = T->totalSteps();
    if (MaybeStepsCount)
      Output << " (" << *MaybeStepsCount << ")";
    Output << "\n";
  }

  void handleTaskCompleted(const llvm::Task *T) override {
    if (T->stepIndex() != -1) {
      indent(T->index() + 1);
      Output << T->stepName() << "\n";
    }
    indent(T->index() + 1 - 1);
    Output << "Ending " << T->name().str() << "\n";
  }

  void handleTaskAdvancement(const llvm::Task *T,
                             llvm::StringRef PreviousStepName) override {
    if (T->stepIndex() == 0)
      return;
    indent(T->index() + 1);
    Output << PreviousStepName.str() << "\n";
  }

private:
  void indent(unsigned Size) {
    for (unsigned I = 0; I < Size; ++I)
      Output << "  ";
  }
};

inline std::string
pad(const llvm::Twine &String, const size_t Size, char PaddingChar = ' ') {
  std::string Result = String.str();

  if (Size > Result.size())
    Result.insert(0, Size - Result.size(), PaddingChar);

  return Result;
}

class TerminalBarsProgressListener : public llvm::ProgressListener {
private:
  using TimePoint = decltype(std::chrono::high_resolution_clock::now());

private:
  size_t MaxProgressBars = 0;
  llvm::raw_ostream &Output;
  std::chrono::time_point<std::chrono::high_resolution_clock> LastDraw;
  std::vector<TimePoint> StartTimes;

private:
  static constexpr auto drawThreshold() {
    using namespace std::chrono_literals;
    return 50ms;
  }

public:
  static constexpr bool AllThreads = false;

public:
  TerminalBarsProgressListener(llvm::raw_ostream &Output) : Output(Output) {}

public:
  void handleNewTask(const llvm::Task *T) override {
    if (T->stack().Tasks.size() > MaxProgressBars) {
      ++MaxProgressBars;
      Output << "\n";
    }

    auto Now = std::chrono::high_resolution_clock::now();
    StartTimes.push_back(Now);
    draw(T->stack(), Now);
  }

  void handleTaskCompleted(const llvm::Task *T) override {
    auto Now = std::chrono::high_resolution_clock::now();
    if (Now - StartTimes.back() >= drawThreshold())
      forceDraw(T->stack(), Now);
    else
      draw(T->stack(), Now);
    StartTimes.pop_back();
  }

  void handleTaskAdvancement(const llvm::Task *T,
                             llvm::StringRef PreviousStepName) override {
    draw(T->stack());
  }

  void draw(const llvm::TaskStack &Stack) {
    draw(Stack, std::chrono::high_resolution_clock::now());
  }

  void draw(const llvm::TaskStack &Stack, const TimePoint &Now) {
    if (Now - LastDraw < drawThreshold())
      return;
    LastDraw = Now;

    forceDraw(Stack, Now);
  }

  void forceDraw(const llvm::TaskStack &Stack, const TimePoint &Now) {
    std::optional<unsigned> TerminalSize;

#if defined(__linux__)
    struct winsize Winsize {};
    // TODO: it might not be stderr
    int Result = ioctl(fileno(stderr), TIOCGWINSZ, &Winsize);
    revng_assert(Result != -1);
    TerminalSize = Winsize.ws_col;
#endif

    revng_assert(Stack.Tasks.size() == StartTimes.size());
    std::vector<float> Advancements;
    Advancements.resize(Stack.Tasks.size());

    auto LastIndex = Stack.Tasks.size() - 1;

    auto StepIndex = [](const llvm::Task *T) -> float {
      return std::max<int64_t>(0, T->stepIndex());
    };

    if (Stack.Tasks[LastIndex]->completed()) {
      Advancements[LastIndex] = 1.0;
    } else if (Stack.Tasks[LastIndex]->totalSteps()) {
      float Index = StepIndex(Stack.Tasks[LastIndex]);
      Advancements[LastIndex] = Index / *Stack.Tasks[LastIndex]->totalSteps();
    }

    for (signed I = LastIndex - 1; I >= 0; --I) {
      auto *Task = Stack.Tasks[I];
      revng_assert(not Task->completed());

      auto MaybeStepsCount = Task->totalSteps();
      if (MaybeStepsCount) {
        auto StepsCount = *MaybeStepsCount;
        revng_assert(Task->stepIndex() < StepsCount);

        if (StepsCount != 0) {

          float Addendum = 0.0;

          if (Task->currentStepHasSingleSubtask())
            Addendum = Advancements[I + 1];

          Advancements[I] = (StepIndex(Task) + Addendum) / StepsCount;
        } else {
          Advancements[I] = 0.0;
        }
      }

      revng_assert(Advancements[I] <= 1.0);
    }

    std::string Lines;

    // Go back MaxProgressBars lines
    using llvm::Twine;
    Lines = (Twine("\r\033[") + Twine(MaxProgressBars) + Twine("A")).str();

    llvm::SmallVector<llvm::SmallString<6>> TaskLengths;
    unsigned Longest = 0;
    for (size_t I = 0; I < Stack.Tasks.size(); ++I) {
      auto Cents = (Now - StartTimes.at(I)).count() / (std::nano().den / 100);
      using llvm::Twine;
      (Twine(Cents / 100) + "." + (Cents % 100 < 10 ? "0" : "")
       + Twine(Cents % 100))
        .toVector(TaskLengths.emplace_back());
      Longest = std::max<unsigned>(Longest, TaskLengths.back().size());
    }

    for (size_t II = 0; II < MaxProgressBars; ++II) {
      std::string Buffer;
      llvm::raw_string_ostream Line(Buffer);

      size_t I = MaxProgressBars - II - 1;

      if (I < Stack.Tasks.size()) {
        auto *T = Stack.Tasks[I];
        unsigned Percent = 100 * Advancements[I];
        constexpr unsigned BarLength = 39;
        unsigned Bar = BarLength * Advancements[I];

        Line << "[";
        unsigned III = 0;
        for (; III < Bar; ++III) {
          Line << "=";
        }

        if (III < BarLength) {
          Line << ">";
          ++III;
        }

        for (; III < BarLength; ++III) {
          Line << " ";
        }
        Line << "] ";
        Line << pad(llvm::Twine(Percent), 3) << "%";

        for (unsigned IIII = 0; IIII < 1 + (Longest - TaskLengths[I].size());
             ++IIII)
          Line << " ";
        Line << TaskLengths[I] << "s";

        Line << " " << T->name().str();

        if (T->totalSteps())
          Line << " (" << *T->totalSteps() << ")";

        if (not T->stepName().empty())
          Line << ": " << T->stepName().str();
      }

      Line.flush();

      if (TerminalSize and Buffer.size() > *TerminalSize) {
        static llvm::StringLiteral Suffix = "...";
        auto CutTo = *TerminalSize - Suffix.size();
        Buffer = llvm::StringRef(Buffer).substr(0, CutTo).str() + Suffix.str();
      }

      Lines += std::string("\r\033[2K") + Buffer + std::string("\n");
    }

    Output << Lines;
  }
};

static void destroyTraceProgressListener(void *OpaqueListener) {
  auto *Listener = static_cast<TraceProgressListener *>(OpaqueListener);
  Listener->close("Exit due to signal");
}

static auto RegisterTraceProgressListener = [](const std::string &Value) {
  if (Value.size() > 0) {
    llvm::ProgressReport->registerListener<TraceProgressListener>(Value);
  }
};

static auto TPLCallback = callback(RegisterTraceProgressListener);

static opt<std::string> TraceProgress("trace", TPLCallback);

static auto RegisterTerminalBarsProgressListener = [](const bool &Value) {
  using namespace llvm;
  if (Value) {
    ProgressReport->registerListener<TerminalBarsProgressListener>(errs());
  }
};

static auto RTBPLCallback = callback(RegisterTerminalBarsProgressListener);

static opt<bool> ProgressBars("progress", RTBPLCallback);

static auto RegisterProgressPlain = callback([](const bool &Value) {
  if (Value)
    llvm::ProgressReport->registerListener<PlainProgressListener>(llvm::errs());
});

static opt<bool> ProgressPlain("progress-plain", RegisterProgressPlain);
