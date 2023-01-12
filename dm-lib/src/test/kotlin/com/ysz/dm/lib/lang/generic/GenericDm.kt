package com.ysz.dm.lib.lang.generic

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-13 2:07 AM
 **/
internal class GenericDm {

    class GenericHolder<T>(private val value: T) {
        fun getValue() = value
    }

    @Test
    fun `test generic`() {
        val h1 = GenericHolder(Automobile("Ford"))
        val a: Automobile = h1.getValue()

        a eq "Automobile(brand=Ford)"


        val h2 = GenericHolder(1)
        val i = h2.getValue()
        i eq 1
    }


    /*定义了一个 extension properties 而不是 functions . 其实都可以, 区别在于可读性*/
    private val List<*>.indices: IntRange
        get() = 0 until this.size

    @Test
    fun `test starProjection`() {
        listOf(1).indices eq 0..0
        listOf('a', 'b', 'c', 'd').indices eq 0..3
        emptyList<Int>().indices eq IntRange.EMPTY
    }
}

data class Automobile(val brand: String)
