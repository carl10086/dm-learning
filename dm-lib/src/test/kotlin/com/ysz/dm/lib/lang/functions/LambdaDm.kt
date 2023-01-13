package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import com.ysz.dm.lib.common.atomictest.trace
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-14 1:38 AM
 **/
internal class LambdaDm {

    @Test
    fun `test basic`() {
        val list = listOf(1, 2, 3, 4)
        val result = list.map { n: Int -> "[$n]" }
        result eq listOf("[1]", "[2]", "[3]", "[4]")
    }

    @Test
    fun `test joinToString`() {
        val list = listOf(1, 2, 3, 4)

        list.joinToString(separator = " ", transform = {
            "[$it]"
        }) eq "[1] [2] [3] [4]"


        list.joinToString(" ") { "[$it]" } eq "[1] [2] [3] [4]"
    }

    @Test
    fun `test moreThanOneParam`() {
        listOf(1, 2).mapIndexed { index, t ->
            "[$index:$t]"
        } eq listOf("[0:1]", "[1:2]")
    }

    @Test
    fun `test zeroParam`() {
        kotlin.run { -> trace("A lambda") }
        kotlin.run { trace("without args") }

        trace eq """
            A lambda
            without args
        """.trimIndent()
    }


    @Test
    fun `test closures`() {
        val list = listOf(1, 5, 7, 10)
        val divider = 5
        list.filter { it % divider == 0 } eq listOf(5, 10)
    }

    @Test
    fun `test closesure2`() {
        val list = listOf(1, 5, 7, 10)

        var sum = 0
        val divider = 5
        list.filter { it % divider == 0 }.forEach { sum += it }

        sum eq 15

    }
}