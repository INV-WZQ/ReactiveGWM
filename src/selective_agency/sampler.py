"""Checkpointable deterministic rank-local sampler without prefetch cursor drift."""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class StatefulDistributedSampler(Sampler[tuple[int, int, int]]):
    def __init__(
        self,
        dataset_size: int,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool = True,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank")
        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.cursor = 0  # committed samples, never DataLoader-prefetch position
        self.samples_per_rank = math.ceil(self.dataset_size / self.num_replicas)
        self.total_size = self.samples_per_rank * self.num_replicas

    def _rank_indices(self) -> list[int]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        if len(indices) < self.total_size:
            indices += (indices * math.ceil((self.total_size - len(indices)) / len(indices)))[
                : self.total_size - len(indices)
            ]
        return indices[self.rank : self.total_size : self.num_replicas]

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        indices = self._rank_indices()
        start = self.cursor
        epoch = self.epoch
        for position in range(start, len(indices)):
            yield indices[position], epoch, position

    def __len__(self) -> int:
        return self.samples_per_rank - self.cursor

    def mark_consumed(self, count: int) -> None:
        if count <= 0 or self.cursor + count > self.samples_per_rank:
            raise ValueError("invalid committed sampler increment")
        self.cursor += int(count)
        if self.cursor == self.samples_per_rank:
            self.epoch += 1
            self.cursor = 0

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "dataset_size": self.dataset_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict) -> None:
        locked = ("dataset_size", "num_replicas", "rank", "seed", "shuffle")
        for name in locked:
            if state[name] != getattr(self, name):
                raise RuntimeError(
                    f"sampler topology/config mismatch for {name}: {state[name]} != {getattr(self,name)}"
                )
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if epoch < 0 or not 0 <= cursor < self.samples_per_rank:
            raise RuntimeError("invalid sampler checkpoint cursor")
        self.epoch = epoch
        self.cursor = cursor

    def seek_to_samples_seen(
        self, samples_seen: int, *, effective_global_batch: int
    ) -> dict[str, int]:
        """Seek an elastic sampler to the nearest earlier legal update boundary.

        ``samples_seen`` is global progress.  The returned signed delta uses a
        negative value for replay and a positive value for skip.  Because an
        effective batch is always ``num_replicas * accumulation`` in this
        trainer, the chosen boundary maps to an integer cursor on every rank.
        """

        samples_seen = int(samples_seen)
        effective_global_batch = int(effective_global_batch)
        if samples_seen < 0 or effective_global_batch <= 0:
            raise ValueError("sampler seek values must be non-negative/positive")
        if effective_global_batch % self.num_replicas:
            raise ValueError("effective global batch must be divisible by sampler world size")
        sampler_start = samples_seen // effective_global_batch * effective_global_batch
        self.epoch, global_position = divmod(sampler_start, self.total_size)
        if global_position % self.num_replicas:
            raise RuntimeError("elastic sampler boundary has no rank-local cursor")
        self.cursor = global_position // self.num_replicas
        if not 0 <= self.cursor < self.samples_per_rank:
            raise RuntimeError("elastic sampler cursor is outside the rank shard")
        delta = sampler_start - samples_seen
        return {
            "samples_seen_before_resume": samples_seen,
            "sampler_start_samples_seen": sampler_start,
            "sample_replay_or_skip": delta,
            "replayed_samples": max(-delta, 0),
            "skipped_samples": max(delta, 0),
            "sampler_epoch": self.epoch,
            "sampler_rank_cursor": self.cursor,
        }
