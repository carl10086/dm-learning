package com.ysz.dm.base.cache.impl

import com.ysz.dm.base.cache.CustomCache
import com.ysz.dm.base.cache.cfg.CustomCacheConfig
import com.ysz.dm.base.redis.core.conn.CustomLettuceConn
import io.lettuce.core.api.sync.RedisCommands
import java.util.*

/**
 * @author carl
 * @since 2023-01-21 4:28 PM
 **/

class LettuceCacheImpl<K, V>(
    private val cfg: CustomCacheConfig<K, V>,
    private val conn: CustomLettuceConn<K, V> = CustomLettuceConn(cfg.conn),
    private val sync: RedisCommands<K, V> = conn.conn.sync()
) : CustomCache<K, V> {


    override fun get(k: K): V? = sync.get(k)
    override fun multiGet(keys: Set<K>): Map<K, V> {
        if (keys.isNotEmpty()) {
            val result = sync.mget(*toTypedArray(keys))

            return buildMap {
                for (keyValue in result) {
                    if (keyValue.hasValue()) {
                        put(keyValue.key, keyValue.value)
                    }
                }
            }
        }


        return Collections.emptyMap()
    }


    @Suppress("UNCHECKED_CAST")
    private fun toTypedArray(keys: Collection<K>): Array<K> {
        val array = arrayOfNulls<Any>(keys.size)

        var i = 0;

        for (key in keys) {
            array[i++] = key
        }

        return array as Array<K>
    }

    override fun rm(k: K) {
        this.sync.del(k)
    }

    override fun put(k: K, v: V) {
        this.sync.setex(k, this.cfg.expireAfterWrite.seconds, v)
    }

    override fun close() {
        conn.close()
    }


}