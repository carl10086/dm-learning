package com.ysz.dm.base.redis.core.conn

import com.ysz.dm.base.core.validate.ApiParamValidateTools.Companion.notNull
import com.ysz.dm.base.core.validate.Validatable
import io.lettuce.core.ReadFrom
import io.lettuce.core.RedisClient
import io.lettuce.core.RedisURI
import io.lettuce.core.codec.RedisCodec

/**
 *<pre>
 *
 *</pre>
 *@author carl.yu
 *@since 2023/1/2
 **/
class CustomLettuceConnCfg<K, V>(
    /*this decide the inner netty connection factory ?*/
    val client: RedisClient,
    /*prefer to read from*/
    val readFrom: ReadFrom? = null,
    val uris: List<RedisURI>,
    val codec: RedisCodec<K, V>
) : Validatable {

    fun onlySingleNode(): Boolean = uris.size == 1

    override fun validate() {
        if (!onlySingleNode()) {
            notNull(readFrom, "if uris is not single , readFrom can not be null")
        }
    }
}