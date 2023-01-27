package com.ysz.dm.base.cache.cfg

import com.ysz.dm.base.redis.core.conn.CustomLettuceConnCfg
import java.time.Duration

/**
 * @author carl
 * @since 2023-01-21 4:20 PM
 **/
data class CustomCacheConfig<K, V>(
    val expireAfterWrite: Duration = Duration.ofMinutes(15),
    val conn: CustomLettuceConnCfg<K, V>

)