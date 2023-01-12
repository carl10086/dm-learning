package com.ysz.dm.lib.lang.nil

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.platform.commons.logging.LoggerFactory

/**
 * @author carl
 * @create 2022-11-16 6:42 PM
 **/
internal class KotlinNullDm {

    private fun nullSupplier(arg: Int): String? = if (arg == 1) null else arg.toString()

    @Test
    internal fun test_returnIfNull() {
        assertEquals(nullSupplier(1) ?: "null", "null")
        assertEquals(nullSupplier(2) ?: "null", "2")
    }


    @Test
    internal fun test_logicAndReturnIfNull() {
        val len: Int? = nullSupplier(2)?.length
        println("$len")
    }


    private fun checkLength(s: String?, expected: Int?) {
        val length1 = s?.length
        assertEquals(expected, length1)
    }

    @Test
    fun `test checkLength`() {
        checkLength("abc", 3)
        checkLength(null, null)
    }

    @Test
    fun `test whenNull`() {
        val s1: String? = "abc"
        assertEquals("abc", (s1 ?: "---"))

        val s2: String? = null
        assertEquals("---", (s2 ?: "---"))
    }

    companion object {
        private val log = LoggerFactory.getLogger(KotlinNullDm::class.java)
    }
}