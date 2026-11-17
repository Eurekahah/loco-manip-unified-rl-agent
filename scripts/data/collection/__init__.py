"""Task-isolated demonstration collection framework."""

from .profiles import CollectionTaskProfile, register_collection_profile, resolve_profile

__all__ = [
    "CollectionTaskProfile",
    "register_collection_profile",
    "resolve_profile",
]
