package com.ysz.dm.lib.lang.other

import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-12 12:54 PM
 **/
internal class DestructureDm {

    private fun compute(input: Int): Pair<Int, String> =
        if (input > 5) Pair(input * 2, "High") else Pair(input * 2, "Low")

    @Test
    fun `test compute`() {
        val (a, b) = compute(1)
    }

    @Test
    fun `test iterate`() {
        val list = listOf('a', 'b', 'c')

        for ((index, value) in list.withIndex()) {
            println("$index:$value")
        }
    }
}