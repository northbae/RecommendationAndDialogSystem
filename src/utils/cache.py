import pickle
from pathlib import Path
from functools import wraps
from typing import Callable, Any
import hashlib
import json

from .config import config


class CacheManager:

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = config.get('data.cache_dir', 'data/cache/')

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = config.get('cache.enabled', True)

    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, cache_key: str) -> Any:
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Ошибка чтения кэша: {e}")
                return None

        return None

    def set(self, cache_key: str, value: Any):
        if not self.enabled:
            return

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Ошибка записи в кэш: {e}")

    def clear(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def cache_result(self, cache_name: str = None):

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = cache_name or func.__name__
                cache_key = self._get_cache_key(func_name, args, kwargs)

                cached_value = self.get(cache_key)
                if cached_value is not None:
                    #print(f"Загружено из кэша: {func_name}")
                    return cached_value

                result = func(*args, **kwargs)

                self.set(cache_key, result)

                return result

            return wrapper

        return decorator

cache_manager = CacheManager()


def cache_matrix(name: str):
    return cache_manager.cache_result(cache_name=name)