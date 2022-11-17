package com.ysz.dm.rb.base.cache.lettuce.codec

import com.ysz.dm.rb.base.cache.impl.lettuce.codec.CustomPrefixCodec
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @create 2022-11-16 6:18 PM
 */
internal class CustomPrefixCodecTest {

    private val stringCodec: CustomPrefixCodec<String> = CustomPrefixCodec("default:", String::class.java)

    @Test
    internal fun `test_encodeThenDecodeKey`() {
        val src = "this is my"
        Assertions.assertEquals(stringCodec.decodeKey(stringCodec.encodeKey(src)), src)
    }


    @Test
    internal fun `test_encodeThenDecodeValue`() {
        val src = "this is my"
        Assertions.assertEquals(stringCodec.decodeValue(stringCodec.encodeValue(src)), src)
    }


}