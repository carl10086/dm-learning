package com.ysz.dm.rb.base.cache.decorators

import com.ysz.dm.rb.base.cache.CustomCache

/**
 * key-Prefix  + decorator
 * @author carl
 * @create 2022-11-17 6:46 PM
 **/
class KeyPrefixDecorator<V>(
    cache: CustomCache<String, V>,
    private val prefix: String
) : BaseDecorator<String, V>(cache) {

    override fun get(key: String): V? {
        return cache.get(prefix + key)
    }

    override fun multiGet(keys: Collection<String>): Map<String, V> {
        return cache.multiGet(keys.map { prefix + it })
    }

    override fun invalidate(k: String) {
        super.invalidate(prefix + k)
    }

    override fun multiInvalidate(keys: Collection<String>) {
        super.multiInvalidate(keys.map { prefix + it })
    }

    override fun put(k: String, v: V) {
        super.put(prefix + k, v)
    }

    override fun multiPut(map: Map<String, V>) {
        super.multiPut(map.entries.associateBy(
            keySelector = {
                prefix + it.key
            },
            valueTransform = {
                it.value
            }
        ))
    }
}