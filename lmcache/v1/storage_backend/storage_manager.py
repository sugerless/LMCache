# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
import time
from collections import OrderedDict
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
)
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.mooncakestore_connector import MooncakestoreConnector

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        allocator: MemoryAllocatorInterface,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()

        self.allocator_backend = LocalCPUBackend(
            config, allocator, lookup_server, lmcache_worker
        )

        self.store = MooncakestoreConnector(
            "",
            0,
            "",
            self.loop,
            self.allocator_backend,
            config,
        )

        # Set up metadata context for the store
        self.store.metadata = metadata
        self.store._setup_metadata_context()

        self.prefetch_tasks: Dict[CacheEngineKey, Future] = {}

        self.manager_lock = threading.Lock()

        self.lookup_server = lookup_server

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.worker_id = metadata.worker_id

        self.stream = torch.cuda.Stream()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
    ) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        return self.allocator_backend.allocate(shape, dtype, fmt, eviction=eviction)

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
    ) -> Optional[MemoryObj]:
        """
        Batched allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        return self.allocator_backend.batched_allocate(
            shape, dtype, batch_size, fmt, eviction=eviction
        )

    # FIXME: Should be deprecated
    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Non-blocking function to put the memory object into the storages.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        start = time.perf_counter()
        logger.info(f"[blankdebug] put() start: key={key}")

        # TODO(Jiayi): currently, the entire put task will be cancelled
        # if one of the backend is already storing this cache.
        # This might not be ideal. We need a caching policy to
        # configure caching policies (e.g., write-through,
        # write-back, etc.)
        if self.store:
            future = asyncio.run_coroutine_threadsafe(
                self.store.put(key, memory_obj), self.loop
            )
            future.result()  # Wait for completion

        memory_obj.ref_count_down()
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"[blankdebug] put() end: key={key}, elapsed={elapsed:.2f} ms")

    def batched_put(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        # FIXME(Jiayi): fix docstring
        """
        Non-blocking function to batched put the memory objects into the
        storage backends.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """

        # TODO(Jiayi): currently, the cache is stored to a certain
        # backend if this backend does not have this cache.
        # There's no way to configure a global caching policy
        # among different storage backends.
        if self.store:
            future = asyncio.run_coroutine_threadsafe(
                self.store.batch_put(keys, memory_objs), self.loop
            )
            future.result()  # Wait for completion

        for memory_obj in memory_objs:
            memory_obj.ref_count_down()

    def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Blocking function to get the memory object from the storages.
        """
        return self.store.get(key)

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storages.
        """
        # TODO (Jiayi): incorporate prefetching here

        # Search all backends for non-blocking get
        if self.store:
            return asyncio.run_coroutine_threadsafe(self.store.get(key), self.loop)
        return None

    def batched_get(self, keys: List[CacheEngineKey]) -> List[Optional[MemoryObj]]:
        """
        Batched blocking function to get multiple memory objects from the storages.

        :param List[CacheEngineKey] keys: The keys to retrieve.
        :return: List of MemoryObj or None for each key.
        """
        if self.store:
            return self.store.batch_get(keys)
        return [None] * len(keys)

    def layerwise_batched_get(
        self,
        keys: List[List[CacheEngineKey]],
    ) -> Generator[List[Future], None, None]:
        """
        Non-blocking function to get the memory objects into the storages
        in a layerwise manner.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        :param List[List[CacheEngineKey]] keys: The keys to get. The first
            dimension corresponds to the number of layers, and the second
            dimension corresponds to the number of chunks.

        :return: A generator that yields a list of futures for each layer.
        """
        for keys_multi_chunk in keys:
            # Store all chunks for one layer
            tasks = []
            for key in keys_multi_chunk:
                task = self.get_non_blocking(key)
                assert task is not None
                tasks.append(task)
            yield tasks

    # TODO(Jiayi): we need to consider eviction in prefetch
    def prefetch_callback(self, future, key):
        """
        Update metadata after prefetch.
        """
        self.manager_lock.acquire()
        self.manager_lock.release()
        try:
            memory_obj = future.result()
        except Exception as e:
            logger.error(f"Exception captured from future in prefetch_callback: {e}")
            raise e

        if memory_obj is None:
            logger.warning("Prefetch returned None")
            return

        # The object is already allocated and contains the data.
        # We just need to put it into the hot cache.
        self.manager_lock.acquire()
        self.allocator_backend.submit_put_task(key, memory_obj)
        self.manager_lock.release()

    def prefetch(self, key: CacheEngineKey) -> None:
        """Launch a prefetch request in the storage backend. Non-blocking"""
        # Check if already in hot cache
        if self.allocator_backend.contains(key):
            return

        self.manager_lock.acquire()
        if key in self.prefetch_tasks:
            self.manager_lock.release()
            return
        self.manager_lock.release()

        if self.store:
            prefetch_task = self.get_non_blocking(key)
            if prefetch_task is None:
                return

            lambda_callback = lambda f: self.prefetch_callback(f, key)

            self.manager_lock.acquire()
            self.prefetch_tasks[key] = prefetch_task
            prefetch_task.add_done_callback(lambda_callback)
            self.manager_lock.release()
            
    def batch_contains(
        self,
        keys: List[CacheEngineKey],
    ) -> List[bool]:
        """
        Check whether the keys exist in the storage backend.

        :param List[CacheEngineKey] keys: The keys to check.

        :param Optional[List[str]] search_range: The range of storage backends
        """
        return self.store.batch_exists(keys)

    # TODO(Jiayi): Currently, search_range is only used for testing.
    def contains(
        self,
        key: CacheEngineKey,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> bool:
        """
        Check whether the key exists in the storage backend.

        :param CacheEngineKey key: The key to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: True if the key exists in the specified storage backends.
        """
        return self.store.exists(key)

    def remove(
        self,
        key: CacheEngineKey,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Remove the key and the corresponding cache in the specified
        locations.

        :param CacheEngineKey key: The key to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """

        num_removed = 0
        if self.allocator_backend:
            num_removed += self.allocator_backend.remove(key)
        return num_removed

    def batched_unpin(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> None:
        """
        Unpin the keys in the specified locations.

        :param List[CacheEngineKey] keys: The keys to unpin.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `unpin` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `unpin` in all backends.
        """
        if self.allocator_backend:
            for key in keys:
                self.allocator_backend.unpin(key)

    def clear(
        self,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Clear all caches in the specified locations.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `clear` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `clear` in all backends.

        return: Total number of cleared caches in the specified
        storage backends.
        """

        num_cleared = 0
        if self.allocator_backend:
            num_cleared += self.allocator_backend.clear()
        return num_cleared

    def close(self):
        if self.store:
            self.store.close()
        if self.allocator_backend:
            self.allocator_backend.close()

        # using threadsafe method here as stop modifies
        # the internal state of the loop (in another thread)
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()

        logger.info("Storage manager closed.")
