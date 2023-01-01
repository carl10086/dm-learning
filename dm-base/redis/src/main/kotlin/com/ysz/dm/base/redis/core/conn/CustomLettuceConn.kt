package com.ysz.dm.base.redis.core.conn

import io.lettuce.core.api.StatefulRedisConnection
import io.lettuce.core.masterreplica.MasterReplica
import java.io.Closeable

/**
 *<pre>
 *  a custom connection for lettuce, sometimes we may both need :
 *
 *  1. master replica connection
 *  2. single node replication
 *  3. cluster
 *  4. ...
 *</pre>
 *@author carl.yu
 *@since 2023/1/2
 **/
class CustomLettuceConn<K, V>(
    private val cfg: CustomLettuceConnCfg<K, V>
) : Closeable {

    val conn: StatefulRedisConnection<K, V> =
        if (cfg.onlySingleNode()) cfg.client.connect(cfg.codec, cfg.uris[0])
        else MasterReplica.connect(
            cfg.client,
            cfg.codec,
            cfg.uris
        ).apply { this.readFrom = cfg.readFrom }


    override fun close() {
        conn.close()
    }
}