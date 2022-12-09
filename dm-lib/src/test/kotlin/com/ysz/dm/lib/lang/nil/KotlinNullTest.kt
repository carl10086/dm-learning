package com.ysz.dm.lib.lang.nil

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.junit.platform.commons.logging.LoggerFactory

/**
 * @author carl
 * @create 2022-11-16 6:42 PM
 **/
internal class KotlinNullTest {

    private fun nullSupplier(arg: Int): String? = if (arg == 1) null else arg.toString()

    @Test
    internal fun test_returnIfNull() {
        Assertions.assertEquals(nullSupplier(1) ?: "null", "null")
        Assertions.assertEquals(nullSupplier(2) ?: "null", "2")
    }


    @Test
    internal fun test_logicAndReturnIfNull() {
        val len: Int? = nullSupplier(2)?.length
        println("$len")
    }

    companion object {
        private val log = LoggerFactory.getLogger(KotlinNullTest::class.java)
    }
}