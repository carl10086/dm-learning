package com.ysz.dm.rb.lettuce

import com.ysz.dm.rb.AbstractCustomCache
import com.ysz.dm.rb.CustomCacheCfg
import io.lettuce.core.api.sync.RedisCommands
import java.util.function.Function

/**
 * @author carl
 * @create 2022-11-15 4:36 PM
 **/
class CustomCacheLettuceImpl<K, V>(
    cfg: CustomCacheCfg,
    /*this connection maybe shared in different cache*/
    private val conn: CustomLettuceConnection<K, V>
) : AbstractCustomCache<K, V>(cfg) {

    private val sync: RedisCommands<K, V> = conn.conn.sync()


    override fun get(k: K, sorLoader: Function<K, V>, penetrationProtect: Boolean, fallback: Function<K, V>?): V? {
        TODO("Not yet implemented")
    }

    fun close() {
        /*true force to close current commands*/
        this.sync.shutdown(false)
    }

    override fun multiGet(keys: Collection<K>, loader: Function<Collection<K>, Map<K, V>>, fallback: Function<K, V>?) {
        TODO("Not yet implemented")
    }

    override fun rm(k: K): Boolean {
        TODO("Not yet implemented")
    }

}