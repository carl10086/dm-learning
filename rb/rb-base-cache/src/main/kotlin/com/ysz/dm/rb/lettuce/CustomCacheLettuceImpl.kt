package com.ysz.dm.rb.lettuce

import com.ysz.dm.rb.AbstractCustomCache
import com.ysz.dm.rb.CustomCacheCfg
import io.lettuce.core.api.sync.RedisCommands

/**
 * @author carl
 * @create 2022-11-15 4:36 PM
 **/
class CustomCacheLettuceImpl<K, V>(
    cfg: CustomCacheCfg,/*this connection maybe shared in different cache*/
    conn: CustomLettuceConnection<K, V>
) : AbstractCustomCache<K, V>(cfg) {

    private val sync: RedisCommands<K, V> = conn.conn.sync()

    fun close() {/*true force to close current commands*/
        this.sync.shutdown(false)
    }


}