package com.ysz.dm.rb.base.cache.decorators

import com.ysz.dm.rb.base.cache.CustomCache
import org.slf4j.LoggerFactory
import java.util.function.Function

/**
 * @author carl
 * @create 2022-11-17 6:46 PM
 **/
class FallbackDecorator<K, V>(
    cache: CustomCache<K, V>,
    /*fallback*/
    private val fallback: Function<K, V>,/*default generate by loader*/
) : BaseDecorator<K, V>(cache) {

    override fun get(key: K): V? = try {
        cache.get(key)
    } catch (e: Exception) {
        log.error("error occurred when get key, k:{}", key, e)
        fallback.apply(key)
    }


    override fun multiGet(keys: Collection<K>): Map<K, V> = try {
        cache.multiGet(keys)
    } catch (e: Exception) {
        log.error("error occurred when multi get key:{}", keys)
        keys.associateWith { fallback.apply(it) }
    }

    companion object {
        private val log = LoggerFactory.getLogger(FallbackDecorator::class.java)
    }
}