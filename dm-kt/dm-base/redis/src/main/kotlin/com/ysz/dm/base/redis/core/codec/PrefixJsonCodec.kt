package com.ysz.dm.base.redis.core.codec

import com.ysz.dm.base.core.tools.json.JsonTools
import io.lettuce.core.codec.RedisCodec
import io.lettuce.core.codec.StringCodec
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

/**
 *<pre>
 * prefix + json codec
 *</pre>
 *@author carl.yu
 *@since 2023/1/2
 **/
class PrefixJsonCodec<K, V>(
    private val prefix: String,
    private val keyClass: Class<K>,
    private val valueClass: Class<V>,
) : RedisCodec<K, V> {

    private val stringCodec = StringCodec(StandardCharsets.UTF_8)

    override fun decodeKey(bytes: ByteBuffer?): K =
        JsonTools.mapper.readValue(stringCodec.decodeKey(bytes).substring(prefix.length), keyClass)

    override fun decodeValue(bytes: ByteBuffer?): V =
        JsonTools.mapper.readValue(stringCodec.decodeValue(bytes), valueClass)

    override fun encodeValue(value: V): ByteBuffer = stringCodec.encodeValue(JsonTools.mapper.writeValueAsString(value))

    override fun encodeKey(key: K): ByteBuffer =
        stringCodec.encodeKey(prefix + JsonTools.mapper.writeValueAsString(key))
}