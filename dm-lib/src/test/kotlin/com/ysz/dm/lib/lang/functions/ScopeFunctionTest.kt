package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import com.ysz.dm.lib.common.atomictest.trace
import org.junit.jupiter.api.Test
import kotlin.random.Random

/**
 * @author carl
 * @create 2022-11-21 5:42 PM
 **/
internal class ScopeFunctionTest {


    data class Tag(var n: Int = 0) {
        var s: String = ""
        fun increment() = ++n;
    }

    @Test
    fun `test difference`() {
        // let : access object with 'it
        // returns last expression in lambda
        Tag(1).let {
            it.s = "let: ${it.n}"
            it.increment()
        } eq 2


        // let with a named argument
        Tag(2).let { tag ->
            tag.s = "let: ${tag.n}"
            tag.increment()
        } eq 3


        // access object with 'this'
        // returns last expression in lambda
        Tag(3).run {
            s = "run: $n"
            increment()
        } eq 4


        // with() : Access object with 'this
        // returns last expression in lambda
        with(Tag(4)) {
            s = "with: $n"
            increment()
        } eq 5


        // apply access with 'this'
        // returns modified object
        Tag(5).apply {
            s = "apply :$n"
            increment()
        } eq "Tag(n=6)"


        Tag(6).also {
            it.s = "also: ${it.n}"
            it.increment()
        } eq "Tag(n=7)"
    }

    @Test
    fun `test gets with null`() {
        fun gets(): String? = if (Random.nextBoolean()) "str!" else null

        gets()?.let { it.removeSuffix("!") + it.length }?.eq("str4")
    }


    data class Plumbus(var id: Int)

    fun display(map: Map<String, Plumbus>) {
        trace("displaying $map")

        val pb1: Plumbus = map["main"]?.let {
            /*如果存在, 则对象 内部 + 10*/
            it.id += 10
            it
        } ?: return /*如果为空, 则直接返回不继续了. 是方法直接返回. 这就 lambda 感觉要特别小心的地方*/
        trace(pb1)

        val pb2: Plumbus? = map["main"]?.run {
            id += 9
            this
        }
        trace(pb2)

        val pb3: Plumbus? = map["main"]?.apply {
            id += 8
        }
        trace(pb3)

        val pb4: Plumbus? = map["main"]?.also { it.id += 7 }
        trace(pb4)
    }

    @Test
    fun `test display`() {
        display(mapOf("main" to Plumbus(1)))
        display(mapOf("none" to Plumbus(2)))

        trace eq """
             displaying {main=Plumbus(id=1)}
             Plumbus(id=11)
             Plumbus(id=20)
             Plumbus(id=28)
             Plumbus(id=35)
             displaying {none=Plumbus(id=2)}
        """.trimIndent()
    }
}