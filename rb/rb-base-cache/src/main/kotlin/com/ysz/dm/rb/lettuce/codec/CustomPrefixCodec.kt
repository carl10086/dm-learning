package com.ysz.dm.rb.lettuce.codec

import com.ysz.dm.rb.base.core.tools.json.JsonTools
import io.lettuce.core.codec.RedisCodec
import io.lettuce.core.codec.StringCodec
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

/**
 * prefix means string
 *
 * @author carl
 * @create 2022-11-15 5:55 PM
 **/
class CustomPrefixCodec<V>(private val prefix: String, private val clz: Class<V>) : RedisCodec<String, V> {

    private val utf8 = StringCodec(StandardCharsets.UTF_8)

    override fun decodeKey(bytes: ByteBuffer) = utf8.decodeKey(bytes).substring(prefix.length)

    override fun decodeValue(bytes: ByteBuffer): V = JsonTools.mapper.readValue(utf8.decodeValue(bytes), clz)

    override fun encodeValue(value: V): ByteBuffer = utf8.encodeValue(JsonTools.mapper.writeValueAsString(value))

    override fun encodeKey(key: String): ByteBuffer = utf8.encodeKey(prefix + key)

}