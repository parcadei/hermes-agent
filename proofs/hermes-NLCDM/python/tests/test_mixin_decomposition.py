"""Discriminative test: Python mixin decomposition preserves self.* access."""


class StorageMixin:
    def store(self, key, value):
        self.data[key] = value

    @staticmethod
    def normalize_key(key):
        return key.strip().lower()


class RetrievalMixin:
    def retrieve(self, key):
        self.store(key, self.data.get(key, "default"))  # calls StorageMixin method
        return self.data[key]


class Engine(StorageMixin, RetrievalMixin):
    def __init__(self):
        self.data = {}


def test_mixin_attr_access():
    e = Engine()
    e.store("x", 42)
    assert e.data == {"x": 42}


def test_cross_mixin_call():
    e = Engine()
    e.data["x"] = 99
    assert e.retrieve("x") == 99


def test_mro_no_diamond():
    mro = [c.__name__ for c in Engine.__mro__]
    assert mro == ["Engine", "StorageMixin", "RetrievalMixin", "object"]


def test_isinstance():
    e = Engine()
    assert isinstance(e, StorageMixin)
    assert isinstance(e, RetrievalMixin)
    assert isinstance(e, Engine)


def test_staticmethod_on_mixin():
    assert Engine.normalize_key("  Foo ") == "foo"
    assert StorageMixin.normalize_key("  Foo ") == "foo"
