from asyncio import iscoroutinefunction, ensure_future
from logging import debug

from models.config import config


def dispatch_event(name: str):
    event_listeners = config("event_listeners")
    if name in event_listeners:
        function = event_listeners[name]
        if iscoroutinefunction(function):
            return ensure_future(function())
        else:
            return function()
