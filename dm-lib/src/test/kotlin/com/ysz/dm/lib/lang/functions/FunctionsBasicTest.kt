package com.ysz.dm.lib.lang.functions

import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory

/**
 *<pre>
 *
 *
 * https://medium.com/mobile-app-development-publication/kotlin-dont-just-use-let-7e91f544e27f
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/17
 **/
internal class FunctionsBasicTest {
    val a = fun(vararg ts: Int): List<Int> {
        val result = ArrayList<Int>()
        for (t in ts) // ts is an Array
            result.add(t)
        return result
    }


    @Test
    internal fun `test_varargs`() {

        fun <T> asList(vararg ts: T): List<T> {
            val result = ArrayList<T>()
            for (t in ts) // ts is an Array
                result.add(t)
            return result
        }

        log.info("{}", asList(1, 2, 3))
    }


    fun <T, R> Collection<T>.fold(
        initial: R, combine: (acc: R, nextElement: T) -> R
    ): R {
        var accumulator: R = initial

        for (element: T in this) {
            accumulator = combine(accumulator, element)
        }
        return accumulator
    }

    @Test
    internal fun `test_highOrder`() {
        val items = listOf(1, 2, 3)

        /*lambdas are code blocks enclosed in curly braces*/
        items.fold(0, { acc: Int, i: Int ->
            acc + i
        })


        /*parameter types in a lambda are optional if they can be inferred:*/
        val joinedToString = items.fold("Elements:", { acc, i -> acc + "" + i })


        /*function reference can also be used for higher-order function calls: */
        val product = items.fold(1, Int::times)
    }


    /*how to declare a function type ? */
    class IntTransformer : (Int) -> Int {
        override fun invoke(p1: Int): Int {
            TODO("Not yet implemented")
        }
    }

    val intFunction: (Int) -> Int = IntTransformer()


    @Test
    internal fun `test_highOrderFunc`() {

        /*no has receiver*/
        val stringPlus: (String, String) -> String = String::plus

        /*has receiver*/
        val intPlus: Int.(Int) -> Int = Int::plus


        println(stringPlus.invoke("<-", "->"))
        println(stringPlus("Hello", "world!"))

        println(intPlus.invoke(1, 1))
        println(2.intPlus(3)) // extension-like call
    }


    @Test
    internal fun `test_lambda`() {
//        max(strings, { a, b -> a.length < b.length })
    }

    companion object {
        private val log = LoggerFactory.getLogger(FunctionsBasicTest.javaClass)
    }


    data class Item(var username: String?)


    inline fun <T> Item.applyIfNotNull(
        item: T?,
        consumer: (t: T?, target: Item) -> Unit,
    ) {
        item?.let {
            consumer.invoke(item, this)
        }
    }

    @Test
    internal fun `test_updateIfNotNull`() {
        val item = Item("111")

        var item2 = Item("222")




    }
}