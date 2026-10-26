"""Actions shared by streaming decisions and replay."""
from enum import IntEnum


class Action(IntEnum):
    ACCEPT = 0
    REJECT = 1
    DEFER = 2
