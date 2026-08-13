This document describes the iterative process of discovering new code in rev.ng.

### Overview

Lifting cannot know what are the targets of an indirect branch until it has analyzed the code computing the jump destination, and it cannot analyze that code until it has been translated.
Discovery is therefore iterative: translate what is known, look for more, translate again.

```{.python notest title="The outer loop"}
def discover(address):
    """Register a jump target and schedule its translation, if never seen."""
    if is_jump_target(address):
        return set()

    register_jump_target(address)
    worklist.append(address)
    return {address}


worklist = initial_jump_targets()
changed = set(worklist)

while worklist:
    while worklist:
        block = translate(worklist.pop())
        for address in direct_successors(block) + return_addresses(block):
            changed |= discover(address)

    # Nothing left to translate: try to discover more blocks using
    # progressively more aggressive strategies.
    for strategy in [..., analyze_values]:
        changed = strategy(changed)
        if worklist:
            break
```

Only the last strategy is expensive: it copies a part of the translated program, optimizes the copy, and runs a value analysis on every indirect branch in it.
Running such analysis over the entire program would be needlessly expensive, so each iteration analyzes a *region* of it.

The rest of this document is about the logic that decides what enters that region, and therefore also whether another iteration is worth running at all.

### The state that drives the iteration

One set drives everything: the addresses that *changed* since the last analysis (`ValueMaterializerPCWhiteList`).
Exactly three things make an address *change*, and each of them can happen only once per address.

1. **A jump target is registered for the first time.**
   Newly discovered code has never been analyzed, so it is worth looking at.
   Nothing that was registered is ever unregistered.

2. **A block's set of successors grows.**
   An indirect branch that just gained a destination is a different branch than the one previously analyzed, and so is everything downstream of it, which now sees one more incoming path.

3. **A call site is detected as `noreturn` for the first time.**
   This leads to *detach the edge from the call to the fallthrough address*.
   Conclusions drawn about a block stop holding when a path into it disappears, so the indirect branches **reachable from** the detached fallthrough *change* (`collectInvalidatedIndirectBranches`).

Nothing else in the pipeline retracts a conclusion.
A branch that an iteration failed to resolve is not remembered as "unresolved, try again": if a later iteration changes the CFG underneath it, it has to be marked as *changed* explicitly, or it is never looked at again.

### From *changed* addresses to a region

The *changed* addresses are not what gets analyzed.
An indirect branch cannot be resolved without the code computing its target, which lies upstream of it, so the region is obtained by expanding the *changed* addresses **backward** and keeping the jump targets met along the way (`inflateValueMaterializerWhitelist`).

The region is a set of jump targets, and it decides which of them the dispatcher may reach in the copy about to be analyzed.
Anything reachable *exclusively* through the dispatcher is left detached when the copy is taken.
What gets analyzed is therefore a subgraph of the program: every indirect branch in it, plus the context the expansion gathered for it.

```{.python notest title="The backward expansion"}
def backward_visit(block):
    # All the unresolved edges converge into the dispatcher: entering it would
    # pull in the whole program.
    seen = {dispatcher}
    queue = [block]

    while queue:
        b = queue.pop()
        if b in seen:
            continue
        seen.add(b)
        yield b

        queue += predecessors(b)
        # Cross call sites backward, from the return address to the caller.
        queue += calls_returning_to(b)
```

The last step deserves a note.
A call block has two outgoing edges of interest: one to the callee, one to the return address.

Walking backward over the first would mean walking into every caller of the enclosing function; that is the interprocedural direction, and this analysis does not go there.

Walking backward over the second stays inside a single function, and it is the edge the analysis actually sees, since the copy is taken with calls rewritten to fall through.
Without it, the expansion would stop at every call, and a branch inside a loop that calls anything would be analyzed with its inputs coming out of the dispatcher: in scope, but with no context.
A callee's entry still has no predecessors, so the visit remains intraprocedural.

### The value analysis

```{.python notest}
def analyze_values(changed):
    region = set()
    for block in changed:
        region |= {b for b in backward_visit(block) if is_jump_target(b)}

    restrict_dispatcher_to(region)
    detach_unreachable_blocks()

    # Point each direct call at its own return address.
    rewrite_calls_to_fall_through()

    # Undo that wherever the callee never returns.
    detached = detach_fallthroughs_of_calls_that_never_return()
    invalidated = indirect_branches_reachable_from(detached)

    copy = clone(program)
    undo_all_of_the_above()

    next_changed = set()
    for branch in indirect_branches(copy):
        targets = materialize(branch)
        if not is_complete(targets):
            continue

        pin(branch, targets)
        for target in targets:
            next_changed |= discover(target)
        if successors_grew(branch, targets):
            next_changed.add(address(branch))

    if not detached <= already_detached:
        already_detached |= detached
        next_changed |= invalidated

    return next_changed
```

`rewrite_calls_to_fall_through` replaces the edge from each direct call to its callee with an edge to the call's return address (`CFGForm::NoFunctionCalls`). Indirect calls are left alone.

`detach_fallthroughs_of_calls_that_never_return` removes that same edge wherever the callee never returns (`cutNoReturnFallthroughs`).
Code laid out after such a call is not reachable, and treating it as reachable merges distinct functions and pollutes the reaching definitions of everything live across the call.

The seeds are the blocks transferring control to a dynamic symbol the model marks as `NoReturn`; the set is then closed by alternating post-dominance and the call graph, so a function whose exits are all seeds becomes a seed in turn.

None of this touches the program permanently: the rewrites are applied, the copy is taken, and then they are undone.
The last three lines are the only lasting effect besides the newly registered jump targets.

### Termination

Each of the three ways in which an address can *change* is monotone, which is what makes the iteration stop.

* Jump targets accumulate and are never unregistered, and a program has finitely many addresses.
* Successor sets only grow, and each of them is bounded.
* Detached fallthroughs are registered at most once.

Each address can therefore *change* only a bounded number of times, and the iteration ends as soon as a round changes nothing.

The detaching itself is deliberately *not* part of the state.
It is recomputed from scratch every round, so the set of calls known never to return keeps growing as more code is discovered, even after the seeds have converged.
This is why the widening is keyed on the fallthroughs actually detached rather than on the seeds.

## Case study: a jump table whose base is clobbered

```asm title="The jump table"
  lea rbx, [rip + table]
loop:
  cmp eax, 1
  ja bail
  movsxd rax, dword ptr [rbx + rax * 4]
  add rax, rbx
  jmp rax

case_0:
  mov rbx, rdi
  jmp bail

bail:
  call abort
case_1:
  jmp loop
```

`rbx` is the base of the table and is set once, before the loop.
`abort` never returns, so `case_1` is not the continuation of `bail`: it is just whatever the assembler laid out after the call.

While that edge exists, `rbx` at `jmp rax` is a join of the base and of the value `case_0` clobbers it with, and nothing can be materialized.

### The call is recognized as never-returning one iteration late

No seed for the noreturn analysis can exist during the first iteration, because a PLT stub does not name the symbol it jumps to:

```asm title="abort@plt"
abort@plt:
  jmp [rip + abort@GOTPCREL]
```

That is an indirect jump, not "a call to `abort`".
The stub becomes recognizable as a jump to a *named symbol* only after some iteration materializes that GOT slot; only then is the block annotated as such (`jump_to_symbol`), and only then does `bail` look like a call site that never returns.

The seeds available to iteration N are what iteration N−1 discovered, and iteration 1 has none.
This is structural rather than an accident of a particular binary: the iteration that first corrects the CFG is never the iteration that discovered the code the correction is about.
By then, the set of *changed* addresses has moved on to whatever turned up most recently, and `jmp rax` is not in it.

### What detaching the fallthrough invalidates

Detaching the edge from `bail` to `case_1` makes `rbx` derivable at `jmp rax`, but only for whoever looks.
`jmp rax` was already analyzed, unsuccessfully, in an earlier iteration, and nothing would bring it back.

Hence the third source of *change*: from each detached fallthrough, the indirect branches it reaches are collected and marked as *changed*, so that the backward expansion gives each of them its context on the following iteration.

Note that `case_1` is both the bogus fallthrough *and* a legitimate arm of the jump table.
Detaching removes an edge, not a block.
