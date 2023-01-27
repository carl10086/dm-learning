package com.ysz.dm.base.cache.decorator

import com.ysz.dm.base.cache.CustomCache
import com.ysz.dm.base.cache.exceptions.CustomCacheException

/**
 * @author carl
 * @since 2023-01-22 12:18 AM
 **/
class TailDecorator<K, V>(cache: CustomCache<K, V>) : BaseDecorator<K, V>(cache) {

    override fun get(k: K): V? {
        return try {
            super.get(k)
        } catch (e: Exception) {
            throw CustomCacheException(e)
        }
    }

    override fun multiGet(keys: Set<K>): Map<K, V> {
        return try {
            super.multiGet(keys)
        } catch (e: Exception) {
            throw CustomCacheException(e)
        }
    }

    override fun rm(k: K) {
        try {
            super.rm(k)
        } catch (e: Exception) {
            throw CustomCacheException(e)
        }
    }

    override fun put(k: K, v: V) {
        try {
            super.put(k, v)
        } catch (e: Exception) {
            throw CustomCacheException(e)
        }
    }

    override fun multiPut(map: Map<K, V>) {
        try {
            super.multiPut(map)
        } catch (e: Exception) {
            throw CustomCacheException(e)
        }
    }

    override fun multiRm(keys: Set<K>) {
        try {
            super.multiRm(keys)
        } catch (e: Exception) {
            throw CustomCacheException(e)
        }
    }

    override fun load(k: K, loader: (k: K) -> V, fallback: ((k: K) -> V)?): V? {
        return try {
            super.load(k, loader, fallback)
        } catch (e: Exception) {
            throw CustomCacheException(e)
        }
    }

    override fun multiLoad(keys: Set<K>, loader: (keys: Set<K>) -> Map<K, V>): Map<K, V> {
        return try {
            super.multiLoad(keys, loader)
        } catch (e: Exception) {
            throw CustomCacheException(e)
        }
    }
}