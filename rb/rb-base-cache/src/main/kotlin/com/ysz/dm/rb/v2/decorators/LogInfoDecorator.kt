package com.ysz.dm.rb.v2.decorators

import com.ysz.dm.rb.v2.CustomCache
import org.slf4j.Logger

/**
 * @author carl
 * @create 2022-11-17 6:46 PM
 **/
class LogInfoDecorator<K, V>(
    private val log: Logger,
    private val cache: CustomCache<K, V>
) :
    CustomCache<K, V> by cache {
    override fun get(key: K): V? {
        log.info("start get :{}", key)
        return cache.get(key).apply { log.info("finish get :{}", key) }
    }

    override fun multiGet(keys: Collection<K>): Map<K, V> {
        log.info("start multi Get:{}", keys)
        return cache.multiGet(keys).apply { log.info("finish multiGet keys:{}", keys) }
    }

}