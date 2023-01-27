package com.ysz.dm.base.cache

import com.ysz.dm.base.cache.exceptions.CustomCacheException
import org.slf4j.LoggerFactory
import java.io.Closeable

/**
 * 1. we need k implement hashCodes and equals method .
 * @author carl
 * @since 2023-01-21 4:03 PM
 **/
interface CustomCache<K, V> : Closeable {
    fun get(k: K): V?

    fun multiGet(keys: Set<K>): Map<K, V>

    fun rm(k: K)

    fun put(k: K, v: V)

    fun multiPut(map: Map<K, V>) {/*可以用 lua 脚本优化, 但是意义不一定大, 要判断, 因为 瓶颈可以能在于 value 很大*/
        map.forEach(::put)
    }

    fun multiRm(keys: Set<K>) {
        keys.forEach(::rm)
    }


    fun load(k: K, loader: (k: K) -> V, fallback: ((k: K) -> V)? = null): V? {
        return try {
            val v: V? = try {
                this.get(k);
            } catch (e: Exception) {
                log.error("get from cache failed , we trying to get from loader, key:{}", k, e)
                /*return here*/
                return loader(k)
            }

            v ?: loader(k)?.apply { put(k, this) }
        } catch (e: Exception) {
            if (fallback == null) throw CustomCacheException.of(e)
            else fallback(k)
        }
    }

    fun multiLoad(keys: Set<K>, loader: (keys: Set<K>) -> Map<K, V>): Map<K, V> {
        if (keys.isNotEmpty()) {


        }
        return emptyMap()
    }


    companion object {
        private val log = LoggerFactory.getLogger(CustomCache::class.java)
    }

}