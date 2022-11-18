package com.ysz.dm.rb.base.cache.decorators

import com.ysz.dm.rb.base.cache.CustomCache

/**
 * @author carl
 * @create 2022-11-18 11:55 AM
 **/
class TransformKeyDecorator<K, V>(
    cache: CustomCache<K, V>, private val transformer: (k: K) -> K
) : BaseDecorator<K, V>(cache) {


    override fun get(key: K): V? {
        return super.get(transformer.invoke(key))
    }

    override fun multiGet(keys: Collection<K>): Map<K, V> {
        return super.multiGet(keys.map {
            transformer.invoke(it)
        })
    }

    override fun invalidate(k: K) {
        super.invalidate(transformer.invoke(k))
    }

    override fun multiInvalidate(keys: Collection<K>) {
        super.multiInvalidate(keys.map {
            transformer.invoke(it)
        })
    }

    override fun put(k: K, v: V) {
        super.put(transformer.invoke(k), v)
    }

    override fun multiPut(map: Map<K, V>) {
        super.multiPut(map.entries.associateBy(
            keySelector = {
                transformer.invoke(it.key)
            },
            valueTransform = {
                it.value
            }
        ))
    }
}