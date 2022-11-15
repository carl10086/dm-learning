package com.ysz.dm.rb

import java.util.function.Function

/**
 * @author carl
 * @create 2022-11-15 3:15 PM
 **/
interface CustomCache<K, V> {

    /***
     * <pre>
     *    load data from cache .
     * </pre>
     */
    fun get(k: K, sorLoader: Function<K, V>, penetrationProtect: Boolean = false, fallback: Function<K, V>? = null): V?

    fun multiGet(keys: Collection<K>, loader: Function<Collection<K>, Map<K, V>>, fallback: Function<K, V>? = null)

    fun rm(k: K): Boolean

    fun rmAll(keys: Collection<K>): Map<K, Boolean> = emptyMap()
}