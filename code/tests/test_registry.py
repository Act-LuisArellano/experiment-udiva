"""
TDD tests for the registry pattern.

Tests register/get cycle, duplicate detection, missing key errors,
listing, and containment checks.
"""

import pytest

from src.core.registry import Registry


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test")

        @reg.register("my_component")
        class MyComponent:
            pass

        assert reg.get("my_component") is MyComponent

    def test_duplicate_registration_raises(self):
        reg = Registry("test")

        @reg.register("dup")
        class First:
            pass

        with pytest.raises(ValueError, match="Duplicate registration"):
            @reg.register("dup")
            class Second:
                pass

    def test_get_unknown_raises(self):
        reg = Registry("test")

        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_get_unknown_shows_available(self):
        reg = Registry("test")

        @reg.register("alpha")
        class Alpha:
            pass

        with pytest.raises(KeyError, match="alpha"):
            reg.get("beta")

    def test_list_registered(self):
        reg = Registry("test")

        @reg.register("b_comp")
        class B:
            pass

        @reg.register("a_comp")
        class A:
            pass

        assert reg.list_registered() == ["a_comp", "b_comp"]

    def test_contains(self):
        reg = Registry("test")

        @reg.register("exists")
        class Exists:
            pass

        assert "exists" in reg
        assert "missing" not in reg

    def test_repr(self):
        reg = Registry("my_registry")

        @reg.register("item")
        class Item:
            pass

        assert "my_registry" in repr(reg)
        assert "item" in repr(reg)

    def test_empty_registry(self):
        reg = Registry("empty")
        assert reg.list_registered() == []
        assert "nothing" not in reg
