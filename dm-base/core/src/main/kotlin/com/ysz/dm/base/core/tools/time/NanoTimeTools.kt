package com.ysz.dm.base.core.tools.time

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/3
 **/
class NanoTimeTools {

    data class Time(
        val ms: Long,
        val nano: Long
    )

    companion object {
        private val startMs = System.currentTimeMillis()
        private val startNano = System.nanoTime()

        private const val MS_NANOS = 1_000_000L


        fun current(): Time {
            val nanoTime = System.nanoTime()
            val msTime = startMs + (nanoTime - startNano) / MS_NANOS
            return Time(msTime, nanoTime)
        }

        fun currentMs(): Long = current().ms
        fun currentNano(): Long = current().nano
    }

}