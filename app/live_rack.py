import queue
import numpy as np
from app.rack_builder import RackItem

class LiveRack:
    def __init__(self):
        self.items = tuple()
        self.commands = queue.SimpleQueue()

    def enqueue_add(self, item: RackItem):
        self.commands.put(("add", item))

    def enqueue_remove(self, item_id: str):
        self.commands.put(("remove", item_id))

    def enqueue_toggle(self, item_id: str, enabled: bool):
        self.commands.put(("toggle", item_id, enabled))

    def enqueue_param(self, item_id: str, name: str, value):
        self.commands.put(("param", item_id, name, value))

    def enqueue_move(self, item_id: str, direction: int):
        self.commands.put(("move", item_id, direction))
    
    def enqueue_eq_band(self, item_id, index, **kwargs):
       self.commands.put(("eq_band", item_id, index, kwargs))
    
    def enqueue_graphic_eq_band(self, item_id, index, gain):
        self.commands.put(("graphic_eq_band", item_id, index, gain))

    def _apply_commands(self, max_commands=64):
        for _ in range(max_commands):
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                break

            kind = cmd[0]

            if kind == "add":
                _, item = cmd
                self.items = self.items + (item,)

            elif kind == "remove":
                _, item_id = cmd
                self.items = tuple(
                    item for item in self.items
                    if item.id != item_id
                )

            elif kind == "toggle":
                _, item_id, enabled = cmd
                for item in self.items:
                    if item.id == item_id:
                        item.effect.enabled = bool(enabled)
                        break

            elif kind == "param":
                _, item_id, name, value = cmd
                for item in self.items:
                    if item.id == item_id:
                        item.effect.set_param(name, value)
                        break

            elif kind == "eq_band":
                _, item_id, index, kwargs = cmd

                for item in self.items:
                    if item.id == item_id:
                        item.effect.set_parametric_eqband(index, **kwargs)
                        break


            elif kind == "graphic_eq_band":
                _, item_id, index, gain = cmd

                for item in self.items:
                    if item.id == item_id:
                        item.effect.set_graphic_band(index, gain)
                        break

            elif kind == "move":
                _, item_id, direction = cmd
                items = list(self.items)
                idx = next(
                    (i for i, item in enumerate(items) if item.id == item_id),
                    None,
                )

                if idx is not None:
                    new_idx = idx + direction

                    if 0 <= new_idx < len(items):
                        items[idx], items[new_idx] = items[new_idx], items[idx]
                        self.items = tuple(items)

    def reset(self):
        for item in self.items:
            item.effect.reset()

    def process_block(self, x):
        self._apply_commands()

        y = x

        # Important: tuple snapshot for this block.
        items = self.items

        for item in items:
            y = item.effect.process_block(y)

            if y is None:
                raise RuntimeError(f"{item.name} returned None")

            y = np.asarray(y, dtype=np.float32)

            if not np.all(np.isfinite(y)):
                raise RuntimeError(f"{item.name} produced NaN/Inf")

        return y