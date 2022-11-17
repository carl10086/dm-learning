package com.ysz.dm.rb.base.cache.impl.lettuce

import com.ysz.dm.rb.base.cache.BaseCustomCache
import com.ysz.dm.rb.base.cache.CustomCacheCfg
import io.lettuce.core.api.sync.RedisCommands

/**
 * @author carl
 * @create 2022-11-15 4:36 PM
 **/
class CustomCacheLettuceImpl<K, V>(
    cfg: CustomCacheCfg<K, V>,/*this connection maybe shared in different cache*/
    conn: CustomLettuceConnection<K, V>
) : BaseCustomCache<K, V>(cfg) {

    private val sync: RedisCommands<K, V> = conn.conn.sync()
    override fun get(key: K): V? {
        TODO("Not yet implemented")
    }

    override fun multiGet(keys: Collection<K>): Map<K, V> {
        TODO("Not yet implemented")
    }

    override fun invalidate(k: K) {
        TODO("Not yet implemented")
    }

    override fun put(k: K, v: V) {
        TODO("Not yet implemented")
    }

    override fun multiInvalidate(keys: Collection<K>) {
        super.multiInvalidate(keys)
    }

    override fun multiPut(map: Map<K, V>) {
        super.multiPut(map)
    }
}