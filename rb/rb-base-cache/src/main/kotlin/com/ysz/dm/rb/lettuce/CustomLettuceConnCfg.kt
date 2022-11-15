package com.ysz.dm.rb.lettuce

import io.lettuce.core.ReadFrom
import io.lettuce.core.RedisClient
import io.lettuce.core.RedisURI
import io.lettuce.core.codec.RedisCodec

/**
 * <pre>
 *     a common connection encapsulation of lettuce inner connection . why ? may connection is different from envs .
 *
 *     - like dev or test , we need  a single node redis
 *     - but in production ,it may be master slave with sentitial , cluster , static cluster
 *
 *     what is support now  only support :
 *
 *     - single node
 *     - static  master slave
 *
 *
 * </pre>
 * @author carl
 * @create 2022-11-15 5:18 PM
 **/
data class CustomLettuceConnCfg<K, V>(
    /*default client with default options*/
    val client: RedisClient = RedisClient.create(),
    /*read from prefer*/
    val readFrom: ReadFrom = ReadFrom.ANY,
    val masterUri: RedisURI,
    val replicaUri: RedisURI = masterUri,
    val codec: RedisCodec<K, V>
) {

    /**
     * <pre>
     *     if master is same with replica, then we need to construct a single node connection
     *
     *     @return true if master is as same as replica
     * </pre>
     */
    fun onlySingleNode() = this.masterUri == replicaUri
}