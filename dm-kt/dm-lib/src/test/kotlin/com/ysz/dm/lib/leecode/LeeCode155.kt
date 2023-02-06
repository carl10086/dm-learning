package com.ysz.dm.lib.leecode


/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/2/7
 **/
class LeeCode155

class MinStack(
) {

    private val stack: java.util.LinkedList<Int> = java.util.LinkedList<Int>()
    private var min: Int? = null

    fun push(`val`: Int) {
        stack.add(`val`)
        min = if (min == null) {
            `val`
        } else {
            kotlin.math.min(`val`, min!!)
        }
    }

    fun pop() {
        val removed = stack.removeLast()
        if (removed == min!!) {
            this.min = this.stack.min()
        }
    }

    fun top(): Int {
        return stack.last()
    }

    fun getMin(): Int {
        return min!!
    }
}

fun main(args: Array<String>) {

    val stack = MinStack()

    stack.push(-2)
    stack.push(-3)
    stack.pop()
    println(stack.getMin())
}
