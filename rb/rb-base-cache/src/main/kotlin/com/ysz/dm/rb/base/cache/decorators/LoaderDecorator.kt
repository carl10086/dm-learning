package com.ysz.dm.rb.base.cache.decorators

import com.ysz.dm.rb.base.cache.CustomCache
import java.util.function.Function

/**
 * @author carl
 * @create 2022-11-17 6:46 PM
 **/
class LoaderDecorator<K, V>(
    cache: CustomCache<K, V>,
    /*loader for single key*/
    private val loader: Function<K, V?>,/*default generate by loader*/
    /*loader for multi keys*/
    private val multiLoader: Function<Collection<K>, Map<K, V>> = Function { keys ->
        buildMap {
            for (key in keys) {
                loader.apply(key)?.let { v -> put(key, v) }
            }
        }
    }
) : BaseDecorator<K, V>(cache) {
    override fun get(key: K): V? {
        return cache.get(key) ?: let {
            /*2. cache miss load from sor*/
            return loader.apply(key)?.let {
                cache.put(key, it)
                return it
            }
        }
    }


    override fun multiGet(keys: Collection<K>): Map<K, V> {
        if (keys.isNullOrEmpty()) return emptyMap()

        /*1. get from cache*/
        val fromCache = cache.multiGet(keys)

        /*2. miss keys = all - result*/
        val miss = keys subtract fromCache.keys

        /*3. no miss keys , direct return*/
        if (miss.isNullOrEmpty()) return fromCache

        /*4. fetch miss from loader*/
        val fromSor = multiLoader.apply(keys)

        if (fromSor.isNotEmpty()) cache.multiPut(fromSor)

        /*5. merge results*/
        return fromSor + fromCache
    }
}