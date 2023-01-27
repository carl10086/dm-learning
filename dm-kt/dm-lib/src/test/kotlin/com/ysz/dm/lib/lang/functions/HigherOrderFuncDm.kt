package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import com.ysz.dm.lib.common.atomictest.trace
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-15 1:18 AM
 **/
internal class HigherOrderFuncDm {

    fun <T> List<T>.any(
        /*predicate 是一个 function*/
        predicate: (T) -> Boolean
    ): Boolean {
        for (element in this) {
            if (predicate(element)) {
                return true;
            }
        }

        return false;
    }

    @Test
    fun `test customAny`() {
        val ints = listOf(1, 2, -3)
        ints.any { it > 0 } eq true

        val strings = listOf("abc", " ")
        strings.any { it.isBlank() } eq true
        strings.any(String::isNotBlank) eq true
    }


    @Test
    fun `test Repeat`() {
        repeat(4) {
            trace("hi")
        }

        trace eq """
            hi
            hi
            hi
            hi
        """.trimIndent()
    }

    fun customRepeat(
        times: Int,
        action: (Int) -> Unit
    ) {
        for (index in 0 until times) {
            action(index)
        }
    }


    @Test
    fun `test autoRemoveNullable`() {

        val transform: (String) -> Int? = { s: String ->
            s.toIntOrNull()
        }

        transform("112") eq 112
        transform("abc") eq null

        val x = listOf("112", "abc")

        x.mapNotNull(transform) eq "[112]"
        x.mapNotNull { it.toIntOrNull() } eq "[112]"

    }
}