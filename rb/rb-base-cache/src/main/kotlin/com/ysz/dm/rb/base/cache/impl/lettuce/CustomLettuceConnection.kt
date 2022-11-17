package com.ysz.dm.rb.base.cache.impl.lettuce

import io.lettuce.core.api.StatefulRedisConnection
import io.lettuce.core.masterreplica.MasterReplica
import org.slf4j.LoggerFactory
import java.io.Closeable

/**
 * @author carl
 * @create 2022-11-15 5:23 PM
 **/
class CustomLettuceConnection<K, V>(private val cfg: CustomLettuceConnCfg<K, V>) : Closeable {

    /*may it common settings ok .*/
    val conn: StatefulRedisConnection<K, V> = if (cfg.onlySingleNode()) {
        /*only single node*/
        cfg.client.connect(cfg.codec, cfg.masterUri)
    } else {
        /*only for static master replica .*/
        MasterReplica.connect(cfg.client, cfg.codec, listOf(cfg.masterUri, cfg.replicaUri))
    }

    init {
        log.info("connection is success created , cfg:{}", cfg)
    }


    companion object {
        private val log = LoggerFactory.getLogger(CustomLettuceConnection::class.java)
    }

    override fun close() {
        conn?.let {
            log.info("connection is closing , cfg:{}", cfg)
            conn.close()
        }
    }

}