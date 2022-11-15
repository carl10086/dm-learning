package com.ysz.dm.rb.lettuce.codec

import io.lettuce.core.codec.RedisCodec
import java.nio.ByteBuffer

/**
 * prefix means string
 *
 * @author carl
 * @create 2022-11-15 5:55 PM
 **/
class CustomPrefixCodec<V> : RedisCodec<String, V> {

    override fun decodeKey(bytes: ByteBuffer?): String {
        TODO("Not yet implemented")
    }

    override fun decodeValue(bytes: ByteBuffer?): V {
        TODO("Not yet implemented")
    }

    override fun encodeValue(value: V): ByteBuffer {
        TODO("Not yet implemented")
    }

    override fun encodeKey(key: String): ByteBuffer {
        TODO("Not yet implemented")
    }
}