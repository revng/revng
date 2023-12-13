#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from asyncio import Queue
from typing import Generic, List, TypeVar
from weakref import ReferenceType, ref

T = TypeVar("T")


class Stream(Generic[T]):
    def __init__(self):
        self.running = True
        self.entered = False
        self.queue: Queue[T] = Queue()

    def __enter__(self):
        if self.entered:
            raise RuntimeError("Stream can be entered only once")

        self.entered = True
        return self

    def __exit__(self, *args):
        self.running = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.entered:
            raise RuntimeError("Stream only works within a context (using 'with')")

        return await self.queue.get()


class MultiQueue(Generic[T]):
    """Implements a multi-consumer, multi-producer queue
    Each consumer uses steam to get a queue instance to iterate over
    while each producer uses send"""

    def __init__(self):
        self.listeners: List[ReferenceType[Stream[T]]] = []

    def stream(self) -> Stream[T]:
        stream: Stream[T] = Stream()
        self.listeners.append(ref(stream))
        return stream

    async def send(self, message: T):
        listeners_copy = self.listeners[:]
        self.listeners.clear()
        for listener_ref in listeners_copy:
            listener = listener_ref()
            if listener is not None and listener.running:
                self.listeners.append(listener_ref)
                await listener.queue.put(message)
