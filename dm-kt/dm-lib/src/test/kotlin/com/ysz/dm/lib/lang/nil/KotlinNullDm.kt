package com.ysz.dm.lib.lang.nil

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

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


    private fun String?.isNullOrLessThen2(): Boolean = this == null || this.length < 2

    @Test
    fun `test nullExtension`() {
        null.isNullOrLessThen2() eq true
        "".isNullOrLessThen2() eq true
        " ".isNullOrLessThen2() eq true
        "  ".isNullOrLessThen2() eq false
    }
}