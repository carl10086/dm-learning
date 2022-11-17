package com.ysz.dm.rb.v2.decorators

import com.ysz.dm.rb.v2.CustomCache
import java.util.function.Function

/**
 * @author carl
 * @create 2022-11-17 6:46 PM
 **/
class LoaderDecorator<K, V>(
    private val cache: CustomCache<K, V>, private val loader: Function<K, V?>,/*default generate by loader*/
    private val multiLoader: Function<Collection<K>, Map<K, V?>>? = Function {
        it.associateWith {
            loader.apply(it)
        }
    }
) : CustomCache<K, V> by cache {
    override fun get(key: K): V? {
        return cache.get(key) ?: let {
            /*2. cache miss load from sor*/
            return loader.apply(key)?.let {
                cache.put(key, it)
                return it
            }
        }
    }
}