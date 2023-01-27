package com.ysz.dm.base.cache.cfg

import com.ysz.dm.base.cache.CustomCache
import com.ysz.dm.base.cache.decorator.TailDecorator
import com.ysz.dm.base.cache.impl.LettuceCacheImpl

/**
 * @author carl
 * @since 2023-01-22 12:23 AM
 **/
class CustomCacheTools {

    companion object {

        fun <K, V> build(cfg: CustomCacheConfig<K, V>): CustomCache<K, V> {
            val cache: CustomCache<K, V> = LettuceCacheImpl(cfg)
            return TailDecorator(cache)
        }
    }
}