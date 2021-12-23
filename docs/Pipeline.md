AutoPipe
=

### glossary

`pipeline`: a object that can be provided with yaml serialized papelines. can be requested to create `Targets` and notified of `invalidations`. It is composed of a tree of named `steps`. Each step is interleaved by a set of named backing container in which the current step will end up writing.

`Step`: a list of `Pipes`.

`Pipe`: a atomic operation to be performed on containers, annotated with a contract that specify the side-effects on the target contained in those containers.

`Target`: A pair composed by a list of `PathComponents` and a `Kind`, futhermore the tag is annotated as either `Exact` or `Derived`.
The lenght of the PathComponents list must be equal to the `Granularity Depth` of the `granularity` associated to the `Kind` of Target.

`PathComponent`: Either a name or `*`, which rappresents "all names".

`Kind`: A node inside the the `Kind Forest`, each kind tree is associated to a `Granularity` (a granularity may be associated to more than one Kind tree).
Each Kind is associated to a ` Container`, that is the only container that is allowed to contain such Kind.

`Granularity`: A node inside the `Granularity Tree`. The depth of the node + 1 specifies the `Granularity Depth`. Conceptually this rappresents the quantity of identifiers required to identify a object at that granularity level.

As an example, consider the following granularity tree:
```graphviz
digraph g {
    root -> functions;
    root -> types;
    functions -> bb;

}
`````

The granularity Depth of root is 1, thus we require a single name to tell apart objects at the root level. (/root/, /translatedRoot/, /binary/,....)

The granularity Depth of functions is 2, thus we require 2 names to tell apart objects at the functions level. (/root/f1/, /root/f2/, /translated/f1/, /translated/f3/, ....)

The granularity Depth of bb is 3, thus we require 3 names to tell apart objects at the bb level. (/root/f1/bb1/, /root/f2/bb1/, /root/f1/bb2/, ...)

` Container`: A backing container is anything that derives form the Container base class. A backing container is conceptually a map from `Targets` to a erased type. Sincethe contained type is erased and thus it cannot be accessed without casting it to the original type, the backing container base class pretty much behaves like a set of `Targets`.

In particular a  Container is able to:

* check if it contains a Target
* remove a contained Target
* copy the current backing container and apply a filter that removes part of the targets
* merge a backing container with the same real type as the current one

`Contract`:
Each Pipe must specify a contract, which will inform the pipeline how input and output of the Pipe relate to each other.

## Overview
The core purpose of the pipeline is to substitute handwritten imperative decompilation pipelines, in favor of a declarative ones. Libraries declare what operations are able to perform, while the user only specifies the names of actions it wishes to perform, and and can enjoyi complex features such as automatic invalidation.

The pipeline system is designed with the following purposes:
* give a pipeline and a `Target` that must be produced, run the minimum required subset of the pipeline to be able to produce the given target. (that is, behave like a in-in memory build system).
* given a invalidated `Target`, invalidate as well all the `Target` that have been produced from it.
* cache the partial results for future use, both in memory and on disk.


We achieve this by placing strong restrictions on what the user is able to specify, while trying allow general purpose computation.

## Final user perspective

The objective of the pipeline is to provide a way to configure in a simple and strightforward way the pipelines involved into the production of rev.ng artifacts.

In particular, at the moment this is a example of pipeline that copies the content of a container to another:

```
Containers:
 - Name:            Strings1
   Type:            StringContainer
 - Name:            Strings2
   Type:            StringContainer
Steps:
 - Name:            FirstStep
   Pipes:
     - Name:             CopyPipe
       UsedContainers: [Strings1, Strings2]
 - Name:            EndStep
 ```

The Containers section declares the type and the name of each container used, as we can see we can have more than one container with the same type, but we cannot have more than one container with the same name.
In this example a string container is just a `Container` that can contains some strings.

The steps name is a list of `steps`, each step is composed of a name, and a list of Pipes.

In this example there is just one `Pipe`, called CopyPipe. The name 'CopyPipe' is simply the names that has been associated to a class that implements the requirements to be used as a Pipe.

This Pipes expresses which containers depends upon with the UsedContainers field.
In this example the list contains Strings1 and Strings2, therefore when executed the CopyPipe will recieve a reference to both those containers and no other.

The semantic of CopyPipe is that it will copy from the first container to the second container.

##  Containers
The `pipeline` syntax may give the impression that there exists a single copy of each ` Container`, this is not the case, there exists a set of backing containers for each step. This sets are each composed by one copy of each declared  Container.


## Execution Model

The way the pipeline operates from the perspective of the final user is that the user must provide a `target` and the name of the ` Container` in which that Target must be produced.

Suppose that the user provides the previously shown pipeline and requests the production of some Target K.

The pipeline will inspect each step starting from the last, and will inspect the backing container with the provided name to see if it contains K. If does contains it then no futher operation is required, other wise it look in the next step or will deduce the rules to produce it. (more on this later)

Once it has been caclulated wich is the first and last Pipe required to produce the provided Target, then the steps that include the mentionied Pipes and all the steps in between are executed.

### Example:
Let as assume that the user wishes to produce the string associated to the Target K, and that it wishes it to be produced in the container Strings2.

Let us assume that all Strings2 containers are empty, and the container strings1 associated to the first step contains the Target K, while the one at EndStep is empty.

If the user reqests the production of K at Strings2 the following will happen:
* The pipeline will check if Strings2 at EndStep contains K.
* Since it does not, it will inspect the Pipeline CopyStrings and understand that the requirements for its execution are that Strings1 must contain K.
* Thus it will check if Strings1 at FirstStep contains K.
* The pipeline has now a starting and ending Pipe that must be executed: CopyPipe alone.
* Thus it will execute all steps inbetween, that is FirstStep alone.

## Clonging and Filtering

When the `Pipeline` executes a subset of the steps it will do so without directly write inside the concerned backing containers.

Instead it will clone the  Containers associated to the current step, operate the transformations on it, and then merge the result into the backing container of the next step.

Furthermore the pipeline will perform a filtering while cloning so that only the minimal subset required to produce the requested pipeline targerts is actually produced.

### Example:
Continuining with the previous example, let us now suppose that Strings1 at step FirstStep contains two Targerts K and J. As before we wish to produce Target K in Container Strings2.

The execution deduction performs as before and executes FirstStep alone.
When the Pipeline executes the step will begin by cloning and filtering the backing containers. In this case it will clone strings1 and filter out J, so that only K remains. It will clone strings2 but it is currently empty.

CopyPipe will be executed on this two cloned containers, and will produce a K target inside Strings2.

At the end of the step, each backing container will be merged into the backing container with the same name at the next step.

So at the end of the execution the state will be the following:

* Strings1 at step FirstStep contains J and K
* Strings2 at step FirstStep contains nothing
* Strings1 at step EndStep contains K (due to copy Pipe not erasing the input container)
* Strings2 at step EndStep contains K (due to having been copied)

## Caching Points

There is a reason we require a name for each step, it is because it is at the start of each step that we trigger the on disk serialization, and the on disk serialization will produce a file for each container with the following path
```
./StepName/ContainerName
```

Thus, in previous example the cache on disk will produce 4 files, when they are all used.

```
./FirstStep/Strings1
./FirstStep/Strings2
./EndStep/Strings1
./EndStep/Strings2
```

## Programmer Perspective

Until now we have discussed how the final user will operate the pipeline, we now discuss how a programmer that wishes to expose a custom Pipe will have to provide it.

### wiriting a custom Pipe

An example custom `Pipe` is the following:

```c++20
class CopyPipe {
public:
  static constexpr auto Name = "Copy Pipe";

  std::array<ContractGroup, 1> getContract() const {
    return { ContractGroup(StringKind, KindExactness::Exact, 0, StringKind, 1)  };

  }

  void run(const StringContainer &S, StringContainer &T) { T = S;  }

};
`````

There are 3 components, the constexpr name that is used to dump informations about it when debugging.

The run method. This method paramethers MUST all be the most derived type of Containers. They must be reference, they can be const.

Futhermore, a Pipe MUST have the getContractMethod. The getContract method must return a iterable container of `ContractGroup`.


## Contracts
A `Pipe` operates transformations over a set of ` Containers`, the pipeline cannot what this transformations may entail, thus a Pipe must specify a contract, which is just a way to encode the side effect of the Pipe over the backing containers when considering it just as a container of `Targets`.

Each Atomic Contract will be evaluated one at the time, one after the other, so any Pipe with more than one atomic contract can be thought as a series of Pipe with one atomic contract each. This is usefull when one cannot separate a Pipe into two components, or wishes to run them toghether for performance reasons, such as llvm passes.

### Atomic Contracts:

Let's start with the previous example:
```c++20
ContractGroup(StringKind, KindExactness::Exact, 0, StringKind, 1)
`````

This atomic contract has the following arguments

```cpp
Kind& sourceKind,
KindExactness exactness,
size_t inputContainerIndex,
Kind& outputKind,
size_t outputContainerIndex
`````

* sourceKind: declares which Targets will be affected by this pipeline
* exactness: either Exact or Derived, it will declare if Targets with kind derived by from sourceKind are to be affected or not.
* inputContainerIndex: the index (left to right) of the argument of the function run that will be used to understand which backing container is the input container.
* outputKind: the kind in which all affected Targets will be transformed into.
* outputContainerIndex: same as inputContainerIndex, just for output.

So, what is the semantics of the ContractGroup example?
It is saying the following:

CopyPipe is a Pipe that transform all Auto Pipe Targets with kind StringKind contained in the container bound to the first argument of the run function, to Targets with kind string kind into the container bound to the second argument of the run function.

Why do we require this complex way to assert this inanity?
We require the indexes of the container because a Pipe may operate on more than one.

```c++20
class CopyMultiplePipe {
public:
  static constexpr auto Name = "Copy Multiple Pipe";

  std::array<ContractGroup, 2> getContract() const {
    return { ContractGroup(StringKind, KindExactness::Exact, 0, StringKind, 2),
             ContractGroup(StringKind, KindExactness::Exact, 1, StringKind, 3)  };

  }

  void run(const StringContainer &S1,
           const StringContainer &S1,
           StringContainer &T)
  {
             T += S1;
             T += S2;

  }

};
`````

For similar reasons we require to be able to express the output kind, since we may either preserve the input one, or simply create a new one.


### Granularity in Pipes
There is a big trick regarding input and output kinds that comes into effect when input and output kinds have different granularity depth.

When the input has a granularity that is less than the output, for each transformed Target it will add the missing `PathComponents` required to make sure that the ammount of quantifiers is equal to the output kind granularity depth. The quantifiers added are always the `All` quantifier.

Conceptually this rappresents transformations that turn a something into its components. As an example consider the function isolation enfocer, which takes as input root, and populates the output container with functions.
The kind associated to root is RootKind with has depth 1, thus only a name is required to identify it. The output is associated to FunctionKind which has depth 2, and thus two names are required to identify it, the name of the root that has produced the function and the name of the function itself.
Thus, when running the isolation Pipe it will produce expect the Target /root/ and create the Target /root/*, which rappresents all isolated functions.

Similarly we must handle the opposite case, that is if the depth of the input is larger than the output. Such as when compiling all the translated functions into a single binary.

In this situation the semantics of the transformation is that the last quantifier of the input MUST be `All`. In the example, ALL functions named by /root/* must be present to be able to produce /root/
