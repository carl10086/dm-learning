package com.ysz.dm.rb.base.cache

/**
 * <pre>
 *     this cache design like guava , but use decorator design model for every thing
 * </pre>
 * @author carl
 * @create 2022-11-17 6:37 PM
 **/
interface CustomCache<K, V> {

    /**
     * load from cache
     */
    fun get(key: K): V?

    fun multiGet(keys: Collection<K>): Map<K, V>

    fun invalidate(k: K)

    fun multiInvalidate(keys: Collection<K>) = run { keys.forEach { invalidate(it) } }

    fun put(k: K, v: V)

    fun multiPut(map: Map<K, V>) = run { map.forEach { (t, u) -> put(t, u) } }

}


